#!/usr/bin/env python3
"""
Convert Qwen2.5-1.5B-Instruct to GGUF BF16 using llama.cpp's convert script.

Usage:
    uv run convert_qwen_gguf.py
    # or: python convert_qwen_gguf.py  (if pip-based venv)
"""

import os
import sys
import subprocess
import shutil

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
HF_SAVE_DIR = "./qwen2.5-1.5b-instruct-hf"
OUTPUT_DIR = "./qwen2.5-1.5b-instruct-gguf"
LLAMA_CPP_DIR = "./llama.cpp"
OUTTYPE = "bf16"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run(cmd, **kwargs):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def install(*packages):
    """Install packages using uv if available, otherwise pip."""
    uv = shutil.which("uv")
    if uv:
        run([uv, "pip", "install", "-q", *packages])
    else:
        run([sys.executable, "-m", "pip", "install", "-q", *packages])


def check_dep(pkg):
    try:
        __import__(pkg)
    except ImportError:
        sys.exit(
            f"❌  Missing: '{pkg}'.\n"
            f"    Run:  uv pip install transformers accelerate torch"
        )


# ---------------------------------------------------------------------------
# 1. Verify core deps
# ---------------------------------------------------------------------------
for pkg in ("transformers", "accelerate", "torch"):
    check_dep(pkg)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ---------------------------------------------------------------------------
# 2. Download & save HF weights
# ---------------------------------------------------------------------------
if os.path.isdir(HF_SAVE_DIR) and any(
    f.endswith(".safetensors") for f in os.listdir(HF_SAVE_DIR)
):
    print(f"\n✔  HF weights already at '{HF_SAVE_DIR}', skipping download.")
else:
    print(f"\n{'=' * 60}")
    print(f"  Downloading {MODEL_ID} → {HF_SAVE_DIR}")
    print(f"{'=' * 60}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(HF_SAVE_DIR)
    model.save_pretrained(HF_SAVE_DIR, safe_serialization=True)
    print(f"\n✔  Saved HF weights to '{HF_SAVE_DIR}'")
    del model

# ---------------------------------------------------------------------------
# 3. Clone llama.cpp if needed
# ---------------------------------------------------------------------------
if not os.path.isdir(LLAMA_CPP_DIR):
    print(f"\n{'=' * 60}")
    print("  Cloning llama.cpp …")
    print(f"{'=' * 60}")
    run(
        [
            "git",
            "clone",
            "--depth=1",
            "https://github.com/ggerganov/llama.cpp",
            LLAMA_CPP_DIR,
        ]
    )
else:
    print(f"\n✔  llama.cpp already cloned at '{LLAMA_CPP_DIR}'")

# ---------------------------------------------------------------------------
# 4. Install only what convert_hf_to_gguf.py needs.
#    - Skips llama.cpp/requirements.txt (pins transformers==5.5.1 which
#      breaks uv's index resolution against the PyTorch index).
#    - Uses `uv pip install` so pip doesn't need to be present in the venv.
# ---------------------------------------------------------------------------
print("\n  Installing convert-script dependencies …")
install("gguf", "sentencepiece", "protobuf")

# ---------------------------------------------------------------------------
# 5. Convert to GGUF
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

convert_script = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")
if not os.path.exists(convert_script):
    sys.exit(f"❌  Could not find '{convert_script}'. Check your llama.cpp clone.")

output_file = os.path.join(OUTPUT_DIR, f"qwen2.5-1.5b-instruct-{OUTTYPE.upper()}.gguf")

print(f"\n{'=' * 60}")
print(f"  Converting HF weights → GGUF ({OUTTYPE.upper()})")
print(f"  Output: {output_file}")
print(f"{'=' * 60}")

run(
    [
        sys.executable,
        convert_script,
        HF_SAVE_DIR,
        "--outtype",
        OUTTYPE,
        "--outfile",
        output_file,
    ]
)

# ---------------------------------------------------------------------------
# 6. Report
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("  ✅  Conversion complete!")
print(f"{'=' * 60}")

if os.path.exists(output_file):
    size_gb = os.path.getsize(output_file) / (1024**3)
    print(f"\n  GGUF file: {output_file}  ({size_gb:.2f} GB)")
else:
    print("\n  Files in output dir:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"    • {f}")

print(f"""
  Load with llama.cpp:
    ./llama-cli -m {output_file} -p "Hello" -n 128

  Load with Ollama — create a Modelfile:
    FROM {output_file}
""")
