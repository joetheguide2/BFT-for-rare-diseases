import unsloth
import numpy as np
import pandas as pd
import gc
import torch
import torch.nn.functional as F
import json
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

max_seq_length = 4096
lora_rank = 128

# ─────────────────────────────────────────────
#  BFT loss components
# ─────────────────────────────────────────────

def compute_per_token_confidence(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    valid_labels = labels.clone()
    valid_labels[valid_labels == -100] = 0
    conf = probs.gather(-1, valid_labels.unsqueeze(-1)).squeeze(-1)
    conf = conf * (labels != -100).float()
    return conf


def _longest_run_per_window(low_windows: torch.Tensor) -> torch.Tensor:
    B, T, g = low_windows.shape
    x = low_windows.long()
    runs = torch.zeros(B, T, device=low_windows.device)
    current = torch.zeros(B, T, device=low_windows.device, dtype=torch.long)
    for i in range(g):
        current = (current + x[..., i]) * x[..., i]
        runs = torch.max(runs, current.float())
    return runs


def sliding_window_stats(conf: torch.Tensor, mask: torch.Tensor, g: int = 256):
    B, T = conf.shape

    med = []
    for b in range(B):
        valid = conf[b][mask[b].bool()]
        med.append(valid.median() if valid.numel() > 0 else torch.tensor(0.5, device=conf.device))
    med = torch.stack(med)

    threshold = (0.5 * med).unsqueeze(-1)
    is_low = ((conf < threshold) & mask.bool()).float()

    pad_len = g - 1
    is_low_pad = F.pad(is_low, (0, pad_len))
    mask_pad = F.pad(mask.float(), (0, pad_len))

    low_windows = is_low_pad.unfold(-1, g, 1)
    mask_windows = mask_pad.unfold(-1, g, 1)

    window_valid_count = mask_windows.sum(-1).clamp(min=1)
    r = low_windows.sum(-1) / window_valid_count

    with torch.no_grad():
        L = _longest_run_per_window(low_windows)

    return r, L, med


def compute_token_weights(conf, mask, g=256, eps=1e-8):
    r, L, med = sliding_window_stats(conf, mask, g)

    group_a = ((r >= 0.05) & (r <= 0.20) & (L <= 4)).float()

    threshold = (0.5 * med).unsqueeze(-1)
    is_low = (conf < threshold).float() * mask.float()
    is_high = (1.0 - is_low) * mask.float()

    mean_conf = conf.mean(-1, keepdim=True).clamp(min=eps)
    suppressed = conf / mean_conf
    w_a = is_high + suppressed * is_low
    w = torch.ones_like(conf)
    w = w * (1 - group_a) + w_a * group_a
    w = w * mask.float()

    n_valid = mask.float().sum().clamp(min=1)
    w = w * (n_valid / w.sum().clamp(min=eps))
    return w


def compute_sample_weights(conf, mask, g=256, s_min=0.5, s_max=2.0, eps=1e-8):
    pad_len = g - 1
    conf_pad = F.pad(conf * mask.float(), (0, pad_len))
    mask_pad = F.pad(mask.float(), (0, pad_len))
    conf_windows = conf_pad.unfold(-1, g, 1)
    mask_windows = mask_pad.unfold(-1, g, 1)

    window_mean = (
        (conf_windows * mask_windows).sum(-1)
        / mask_windows.sum(-1).clamp(min=1)
    )

    large = torch.full_like(window_mean, 1e9)
    valid_window_mean = torch.where(mask.bool(), window_mean, large)
    min_conf = valid_window_mean.min(dim=-1).values.clamp(min=eps)

    s = (1.0 / min_conf).clamp(s_min, s_max)
    s = s / s.mean().clamp(min=eps)
    return s


def bft_loss(logits, labels, g=256, s_min=0.5, s_max=2.0):
    mask = (labels != -100).float()

    with torch.no_grad():
        conf = compute_per_token_confidence(logits.detach(), labels)
        token_w = compute_token_weights(conf, mask, g)
        sample_w = compute_sample_weights(conf, mask, g, s_min, s_max)

    B, T, V = logits.shape
    ce = F.cross_entropy(
        logits.view(B * T, V),
        labels.clamp(min=0).view(B * T),
        reduction="none",
    ).view(B, T)

    ce = ce * mask
    weighted_ce = ce * token_w * sample_w.unsqueeze(-1)
    loss = weighted_ce.sum() / mask.sum().clamp(min=1)
    return loss


# ─────────────────────────────────────────────
#  BFT Trainer
# ─────────────────────────────────────────────

class BFTTrainer(SFTTrainer):
    def __init__(self, *args, bft_window_size=256, bft_s_min=0.5, bft_s_max=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.bft_g = bft_window_size
        self.bft_s_min = bft_s_min
        self.bft_s_max = bft_s_max

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss = bft_loss(logits, labels, g=self.bft_g, s_min=self.bft_s_min, s_max=self.bft_s_max)
        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────────
#  Model + tokeniser
# ─────────────────────────────────────────────

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=False,
    max_lora_rank=lora_rank,
)
tokenizer.model_max_length = max_seq_length
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_rank * 2,
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────

df = pd.read_csv("./rare_diseases_what_is_only.csv")
df["messages"] = df["messages"].apply(json.loads)
tokenizer = get_chat_template(tokenizer, chat_template="qwen2.5")

ds = df.copy()
ds["text"] = None
for i in range(len(ds)):
    messages = ds.loc[i, "messages"]
    ds.loc[i, "text"] = tokenizer.apply_chat_template(messages, tokenize=False)
ds = Dataset.from_pandas(ds[["text"]])

def check_lengths(example):
    tokens = tokenizer(example["text"], truncation=True, max_length=max_seq_length)
    return len(tokens["input_ids"])

lengths = ds.map(lambda x: {"length": check_lengths(x)})
print(f"Max token length after forced truncation: {max(lengths['length'])}")

gc.collect()
torch.cuda.empty_cache()

# ─────────────────────────────────────────────
#  Train
# ─────────────────────────────────────────────

trainer = BFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    bft_window_size=256,
    bft_s_min=0.5,
    bft_s_max=2.0,
    args=SFTConfig(
        output_dir="./checkpoints_bft",
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        num_train_epochs=20,
        learning_rate=2e-4,
        logging_steps=1,
        optim="paged_adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        seed=3407,
        packing=True,
        gradient_checkpointing="unsloth",
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        logging_dir="./runs_bft",
    ),
)

trainer.train(resume_from_checkpoint=False)

model.save_pretrained("finetuned_1.5_bft")
tokenizer.save_pretrained("finetuned_1.5_bft")

gc.collect()
torch.cuda.empty_cache()
