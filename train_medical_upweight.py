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

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────

MEDICAL_TOKEN_UPSCALE_WEIGHT = 1.5   # ← change this to tune upweighting strength

# Add your medical entities / terms here.
# These are matched against decoded tokens, case-insensitively.
# Include gene symbols, HPO terms, disease names, key clinical terms, etc.
MEDICAL_ENTITIES = [
    # Example rare disease entities — replace / extend with your ZebraMap vocabulary
    "fibromyalgia", "gaucher", "pompe", "fabry", "niemann-pick",
    "huntington", "tay-sachs", "phenylketonuria", "pku",
    "mucopolysaccharidosis", "mps", "glycogen", "lysosomal",
    "autosomal", "recessive", "dominant", "x-linked",
    "haploinsufficiency", "missense", "nonsense", "frameshift",
    "heterozygous", "homozygous", "hemizygous",
    "enzyme", "deficiency", "substrate", "accumulation",
    "hepatosplenomegaly", "splenomegaly", "hepatomegaly",
    "neuropathy", "cardiomyopathy", "myopathy",
    "anemia", "thrombocytopenia", "leukopenia",
    "hpо", "orpha", "omim", "gene", "allele", "mutation", "variant",
    # Add more from your ZebraMap field vocabulary
]

max_seq_length = 4096
lora_rank = 128

# ─────────────────────────────────────────────
#  Medical entity token mask builder
# ─────────────────────────────────────────────

def build_medical_token_ids(tokenizer, entity_list: list[str]) -> set[int]:
    """
    Pre-compute the set of token IDs that correspond to medical entities.
    We tokenise each entity term (with and without a leading space, since
    subword tokenisers like Qwen's tiktoken-based one often encode
    "gaucher" differently from " gaucher") and collect all resulting token IDs.

    This runs once at startup — O(|entity_list|), cheap.
    """
    token_ids = set()
    for term in entity_list:
        for variant in [term, " " + term, term.capitalize(), " " + term.capitalize()]:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            token_ids.update(ids)
    print(f"[MedUpweight] Medical vocabulary covers {len(token_ids)} unique token IDs "
          f"from {len(entity_list)} entity terms.")
    return token_ids


def build_token_weight_tensor(
    labels: torch.Tensor,
    medical_token_ids: set[int],
    upscale_weight: float,
) -> torch.Tensor:
    """
    For each token in the batch, return upscale_weight if it is a medical
    entity token, else 1.0. Ignored positions (label == -100) get 0.0.

    Args:
        labels:            [B, T]
        medical_token_ids: set of token IDs to upweight
        upscale_weight:    multiplier for medical tokens

    Returns:
        weights: [B, T]
    """
    mask = (labels != -100).float()                     # [B, T]

    # Build a boolean mask for medical tokens
    # We convert the set to a 1-D lookup tensor for vectorised indexing
    B, T = labels.shape
    flat_labels = labels.clamp(min=0).view(-1)          # [B*T], -100 clamped to 0

    # Vectorised lookup: build a dense bool tensor over vocab
    # This is fast because vocab size is ~150k for Qwen and fits in memory easily
    vocab_size = max(medical_token_ids) + 1 if medical_token_ids else 1
    is_medical_vocab = torch.zeros(vocab_size, dtype=torch.bool, device=labels.device)
    idx_tensor = torch.tensor(list(medical_token_ids), dtype=torch.long, device=labels.device)
    is_medical_vocab[idx_tensor] = True

    # Clamp flat_labels that exceed vocab_size (edge case with special tokens)
    safe_labels = flat_labels.clamp(max=vocab_size - 1)
    is_medical_flat = is_medical_vocab[safe_labels]     # [B*T]
    is_medical = is_medical_flat.view(B, T).float()     # [B, T]

    # Build weight tensor: medical tokens get upscale_weight, others get 1.0
    weights = torch.ones_like(mask)
    weights = weights + is_medical * (upscale_weight - 1.0)   # 1.0 + (w-1) for medical
    weights = weights * mask                                   # zero out ignored positions

    return weights                                             # [B, T]


def medical_upweight_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    medical_token_ids: set[int],
    upscale_weight: float = 1.5,
) -> torch.Tensor:
    """
    Standard cross-entropy loss with medical entity tokens upweighted.

    L = sum_{b,t} [ w_bt * CE(logits_bt, y_bt) ] / sum(mask)

    where w_bt = upscale_weight if token is a medical entity, else 1.0.

    The weight tensor is detached (pure scalar multiplier, no gradient through it).
    """
    mask = (labels != -100).float()

    with torch.no_grad():
        token_w = build_token_weight_tensor(labels, medical_token_ids, upscale_weight)

    B, T, V = logits.shape
    ce = F.cross_entropy(
        logits.view(B * T, V),
        labels.clamp(min=0).view(B * T),
        reduction="none",
    ).view(B, T)

    ce = ce * mask
    weighted_ce = ce * token_w
    loss = weighted_ce.sum() / mask.sum().clamp(min=1)
    return loss


# ─────────────────────────────────────────────
#  Medical Upweight Trainer
# ─────────────────────────────────────────────

class MedicalUpweightTrainer(SFTTrainer):
    def __init__(self, *args, medical_token_ids: set, upscale_weight: float = 1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.medical_token_ids = medical_token_ids
        self.upscale_weight = upscale_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss = medical_upweight_loss(
            logits,
            labels,
            self.medical_token_ids,
            self.upscale_weight,
        )
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

# Build medical token ID set once after tokeniser is ready
medical_token_ids = build_medical_token_ids(tokenizer, MEDICAL_ENTITIES)

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

trainer = MedicalUpweightTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    medical_token_ids=medical_token_ids,
    upscale_weight=MEDICAL_TOKEN_UPSCALE_WEIGHT,
    args=SFTConfig(
        output_dir="./checkpoints_medupweight",
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
        logging_dir="./runs_medupweight",
    ),
)

trainer.train(resume_from_checkpoint=False)

model.save_pretrained("finetuned_1.5_medupweight")
tokenizer.save_pretrained("finetuned_1.5_medupweight")

gc.collect()
torch.cuda.empty_cache()
