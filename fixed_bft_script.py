"""
Balanced Fine-Tuning (BFT) Training Script - Full Prompt Loss Configuration (4-bit QLoRA)

Adjustments applied:
  1. Reverted to Full Prompt Loss calculation (labels copy input_ids).
  2. Restored DataCollatorForSeq2Seq instead of completion-only masking.
  3. Preserved the token-level scaling fix to prevent the double-division bug.
  4. Added 4-bit model quantization via BitsAndBytesConfig (QLoRA).
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,  # Added for 4-bit stability
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,  # Added for 4-bit configuration
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from transformers import TrainerCallback


class SaveAdapterPerEpochCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(state.epoch)
        save_path = f"./bft_big_{epoch}"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Adapter saved to {save_path}")


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 4096
LORA_RANK = 32
LORA_ALPHA = 64
OUTPUT_DIR = "./checkpoints_bft"
SAVE_DIR = "./bft_big"

# BFT hyperparameters
BFT_GROUP_SIZE = int(os.getenv("BFT_GROUP_SIZE", "64"))
BFT_CONF_GAMMA = float(os.getenv("BFT_CONF_GAMMA", "1.0"))

# ─────────────────────────────────────────────
#  BFT Loss Functions
# ─────────────────────────────────────────────


def lowest_group_confidence(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    group_size: int | None = None,
) -> torch.Tensor:
    """
    Compute per-sample "Lowest Group Confidence" p_conf in [0,1].
    Evaluates across all unmasked tokens in the sequence.
    """
    if group_size is None:
        group_size = BFT_GROUP_SIZE

    with torch.no_grad():
        B, T, V = logits.shape
        logits = logits.float()

        # Shift labels to align with predictions (predict token t+1 at position t)
        labels_pad = F.pad(labels, (0, 1), value=ignore_index)  # [B, T+1]
        shift_labels = labels_pad[..., 1:].contiguous()[:, :T]  # [B, T]
        valid_mask = shift_labels != ignore_index  # [B, T]

        # Log-probs of correct tokens
        log_probs = logits.log_softmax(dim=-1)  # [B, T, V]
        gather_idx = shift_labels.masked_fill(~valid_mask, 0).unsqueeze(-1)
        per_tok_logp = torch.gather(log_probs, dim=-1, index=gather_idx).squeeze(
            -1
        )  # [B, T]

        p_conf_list = []
        for b in range(B):
            v = per_tok_logp[b][valid_mask[b]]
            if v.numel() == 0:
                p_conf_list.append(torch.tensor(0.0, device=logits.device))
                continue

            if v.numel() < group_size:
                min_avg_logp = v.mean()
            else:
                # conv1d sliding window average
                x = v.view(1, 1, -1)
                w = torch.ones(1, 1, group_size, device=logits.device) / group_size
                y = F.conv1d(x, w, stride=1)
                min_avg_logp = y.min().squeeze()

            p_conf = torch.exp(min_avg_logp).clamp(0.0, 1.0)
            p_conf_list.append(p_conf)

        p_conf = torch.stack(p_conf_list)
        return p_conf


def bft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_items_in_batch: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    BFT Loss: Sample-adaptive reweighting over token-level DFT Cross-Entropy.
    Handles global batch token division accurately without sequence-level double-averaging.
    """
    # ── Sample-level confidence ──
    p_conf = lowest_group_confidence(logits, labels, ignore_index=-100)
    p_conf = (p_conf**BFT_CONF_GAMMA).detach()  # [B]

    # ── Shift labels for causal LM ──
    B, T, V = logits.shape
    logits_flat = logits.float().view(-1, V)

    labels_pad = F.pad(labels, (0, 1), value=-100)
    shift_labels = labels_pad[..., 1:].contiguous()[:, :T]
    shift_labels_flat = shift_labels.reshape(-1)

    # ── Per-token cross-entropy loss ──
    per_token_loss_flat = F.cross_entropy(
        logits_flat,
        shift_labels_flat,
        ignore_index=-100,
        reduction="none",
    )
    per_token_loss = per_token_loss_flat.view(B, T)  # [B, T]

    # ── DFT token-level weights (detached) ──
    with torch.no_grad():
        dft_weights = torch.exp(-per_token_loss)  # [B, T]

    # ── Token-level weighted DFT loss ──
    weighted_loss = per_token_loss * dft_weights  # [B, T]
    valid_mask = (shift_labels != -100).float()  # [B, T]

    # ── BFT sample reweighting ──
    sample_weight = (1.0 - p_conf).to(weighted_loss.dtype)  # [B]

    if num_items_in_batch is not None:
        # Sum the weighted token losses across the batch instead of pre-averaging over sequence lengths
        total_loss = (sample_weight.unsqueeze(1) * weighted_loss * valid_mask).sum()
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(total_loss.device)
        loss = total_loss / num_items_in_batch
    else:
        # Fallback reduction strategy
        sample_dft_loss = (weighted_loss * valid_mask).sum(dim=1) / (
            valid_mask.sum(dim=1) + 1e-8
        )  # [B]
        loss = (sample_weight * sample_dft_loss).mean()

    return loss


# ─────────────────────────────────────────────
#  BFT Trainer
# ─────────────────────────────────────────────


class BFTTrainer(Trainer):
    """
    HuggingFace Trainer subclass optimized to inject custom sample-adaptive BFT losses.
    """

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss = bft_loss(logits, labels, num_items_in_batch)
        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────────
#  Model + Tokenizer Initialization
# ─────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = MAX_SEQ_LENGTH
tokenizer.padding_side = "right"

# 4-bit Quantization Config (NF4 with double quantization)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Crucial step for k-bit models when using gradient checkpointing
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ─────────────────────────────────────────────
#  Dataset Processing
# ─────────────────────────────────────────────

df = pd.read_csv("./rare_diseases_sft_variations.csv")
df["messages"] = df["messages"].apply(json.loads)

ds = df.copy()
ds["text"] = None
for i in range(len(ds)):
    ds.loc[i, "text"] = tokenizer.apply_chat_template(
        ds.loc[i, "messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
ds = Dataset.from_pandas(ds[["text"]])


def tokenize(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )
    result["labels"] = result["input_ids"].copy()  # Full prompt loss configuration
    return result


tokenized_ds = ds.map(tokenize, remove_columns=["text"], num_proc=4)

gc.collect()
torch.cuda.empty_cache()

# ─────────────────────────────────────────────
#  Execution Configuration
# ─────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    warmup_ratio=0.05,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=1,
    optim="paged_adamw_8bit",
    weight_decay=0.001,
    lr_scheduler_type="cosine",
    seed=3407,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    report_to="none",
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8,
    label_pad_token_id=-100,
)

trainer = BFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=data_collator,
    callbacks=[SaveAdapterPerEpochCallback()],
)

trainer.train(resume_from_checkpoint=False)

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Saved successfully to {SAVE_DIR}")
