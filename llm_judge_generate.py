#!/usr/bin/env python3
"""
Judge for medical model outputs.
Reads evaluation_results_1727.csv (columns: question, model_output, actual_answer),
sends each row to a local LLM judge (llama.cpp), and appends the judge's output
as new columns. Saves incrementally after every batch.
Uses concurrent requests for efficiency.
"""

import asyncio
import json
import time
import pandas as pd
from openai import AsyncOpenAI
from pathlib import Path

# ------------------------------------------------------------
# CONFIGURATION – ADAPT TO YOUR SETUP
# ------------------------------------------------------------
CSV_FILE = "evaluation_results_1.5B_overfitted.csv"  # Input CSV
OUTPUT_FILE = "llm_judge_eval_1.5B_overfitted.csv"  # Output CSV with all fields + judge columns
BATCH_SIZE = 4  # Number of concurrent requests per batch
BASE_URL = "http://localhost:8080/v1"
API_KEY = "sk-no-key-required"
MODEL = "gpt-3.5-turbo"  # Ignored by llama.cpp
MAX_TOKENS = 1024  # Brief responses need far fewer tokens
TEMPERATURE = 0.0  # Deterministic for reproducibility

# ------------------------------------------------------------
# The full judge prompt with escaped braces (double {{ }} for JSON schema)
# ------------------------------------------------------------
JUDGE_PROMPT_TEMPLATE = """You are a medical fact-checking judge evaluating a language model's answer about a rare disease against a ground truth definition.

Output ONLY a single JSON object — no text before or after it. Keep every string field to 5 words or fewer. Do not write full sentences anywhere in the JSON.

JSON schema:
{{
  "is_correct": true | false,
  "criticality": "severe" | "high" | "moderate" | "low" | "none",
  "confidence": 0.0-1.0,
  "explanation": "≤5 words",
  "confused_with": ["Disease Names"],
  "errors": [
    {{"id": 1, "error_type": "snake_case_label", "details": "≤5 words"}}
  ]
}}

criticality should only be severe when the model gets the name of the disease wrong and hallucinates the entire disease and every detail. Otherwise categorize in high moderate or low. Prevalence numbers, UMLS, treatment are low priority, errors in these should be low criticality.

Error types (use any that fit, add your own):
- wrong_category, wrong_inheritance, wrong_age_of_onset, wrong_gene_locus
- fabricated_prevalence, fabricated_symptom, fabricated_diagnostic_test, hallucinated_treatment
- confused_with_another_disease, feature_blend, name_confusion, wrong_prognosis, wrong_organ_system, other

Use as little words as possible when describing errors.

Rules:
- List each distinct error as a separate object with a unique numeric id starting at 1.
- Put any real disease whose features appear in the model output in the "confused_with" array.
- Ignore repeated/looping text in the model output — evaluate unique content only.
- If is_correct is true, errors must be [].

Disease: {disease_name}
Model output: {model_output}
Ground truth: {ground_truth}
"""


def build_judge_prompt(disease_name, model_output, ground_truth):
    """Insert the three fields into the judge prompt."""
    return JUDGE_PROMPT_TEMPLATE.format(
        disease_name=disease_name, model_output=model_output, ground_truth=ground_truth
    )


# ------------------------------------------------------------
# Load CSV and prepare columns
# ------------------------------------------------------------
def load_dataframe():
    """Load input CSV if output doesn't exist yet, else resume from output CSV."""
    output_path = Path(OUTPUT_FILE)
    if output_path.exists():
        print(f"📂 Resuming from existing output file: {OUTPUT_FILE}")
        df = pd.read_csv(OUTPUT_FILE)
    else:
        print(f"📂 Starting fresh from input file: {CSV_FILE}")
        df = pd.read_csv(CSV_FILE)

    # Ensure all judge output columns exist
    required_cols = [
        "judge_raw",  # raw response string from judge (always saved)
        "judge_parse_status",  # 'ok', 'parse_error', 'request_error'
        "is_correct",
        "criticality",
        "confidence",
        "explanation",
        "confused_with",  # JSON array as string
        "errors",  # JSON array of dicts as string
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""
    return df


def get_unprocessed_indices(df):
    """Return indices where judge_raw is empty/NaN (not yet attempted)."""
    return df.index[df["judge_raw"].isna() | (df["judge_raw"] == "")].tolist()


# ------------------------------------------------------------
# Try to extract JSON from potentially messy LLM output
# ------------------------------------------------------------
def try_parse_judge_json(raw_text):
    """
    Attempt to extract and parse the JSON object from raw LLM output.
    Returns (parsed_dict, error_message). On failure, parsed_dict is None
    and error_message describes what went wrong — but raw_text is always saved.
    """
    if not raw_text:
        return None, "Empty response"

    # Find the outermost { ... } to strip any preamble/postamble
    start_brace = raw_text.find("{")
    end_brace = raw_text.rfind("}")

    if start_brace == -1 or end_brace == -1:
        return None, "No JSON object found in response"

    json_str = raw_text[start_brace : end_brace + 1]

    try:
        data = json.loads(json_str)
        return data, None
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {e}"


# ------------------------------------------------------------
# Async request
# ------------------------------------------------------------
async def single_judge_request(client, prompt, idx):
    start = time.perf_counter()
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        elapsed = time.perf_counter() - start
        raw_output = response.choices[0].message.content
        token_usage = response.usage.completion_tokens
        print(f"  ✅ [{idx}] {token_usage} tokens in {elapsed:.2f}s")
        return raw_output, token_usage, elapsed, None
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"  ❌ [{idx}] Request ERROR: {e}")
        return None, 0, elapsed, str(e)


# ------------------------------------------------------------
# Process a batch and update DataFrame
# ------------------------------------------------------------
async def process_batch(client, batch_indices, df):
    """Send judge requests for batch_indices, update df in place."""
    tasks = []
    for idx in batch_indices:
        row = df.loc[idx]
        prompt = build_judge_prompt(
            disease_name=row["question"],
            model_output=row["model_output"],
            ground_truth=row["actual_answer"],
        )
        tasks.append(single_judge_request(client, prompt, idx))

    results = await asyncio.gather(*tasks)

    successes = 0
    total_tokens = 0

    for idx, (raw_output, tokens, _, request_error) in zip(batch_indices, results):
        if request_error is not None:
            # Network/API failure — store error marker, leave other cols blank
            df.at[idx, "judge_raw"] = f"[REQUEST_ERROR: {request_error}]"
            df.at[idx, "judge_parse_status"] = "request_error"
            print(f"  ⚠️  [{idx}] Marked as request_error, will retry on next run.")
            continue

        # Always save the raw response regardless of JSON validity
        df.at[idx, "judge_raw"] = raw_output.strip()

        parsed_data, parse_error = try_parse_judge_json(raw_output)

        if parsed_data is not None:
            # Happy path: JSON parsed successfully
            df.at[idx, "judge_parse_status"] = "ok"
            df.at[idx, "is_correct"] = str(parsed_data.get("is_correct", ""))
            df.at[idx, "criticality"] = str(parsed_data.get("criticality", ""))
            df.at[idx, "confidence"] = str(parsed_data.get("confidence", ""))
            df.at[idx, "explanation"] = str(parsed_data.get("explanation", ""))
            df.at[idx, "confused_with"] = json.dumps(
                parsed_data.get("confused_with", [])
            )
            df.at[idx, "errors"] = json.dumps(parsed_data.get("errors", []))
            print(
                f"  ✔  [{idx}] Parsed OK — is_correct={parsed_data.get('is_correct')}, criticality={parsed_data.get('criticality')}"
            )
        else:
            # Parse failure — raw response is already saved above, structured cols stay blank
            df.at[idx, "judge_parse_status"] = "parse_error"
            # Leave is_correct, criticality, etc. empty so they're easy to filter
            print(
                f"  ⚠️  [{idx}] Parse error ({parse_error}). Raw response saved to judge_raw."
            )

        successes += 1
        total_tokens += tokens

    print(
        f"  Batch completed: {successes}/{len(batch_indices)} requests successful, {total_tokens} tokens"
    )
    return total_tokens


# ------------------------------------------------------------
# Main incremental processing loop
# ------------------------------------------------------------
async def main():
    df = load_dataframe()
    all_indices = get_unprocessed_indices(df)
    total_rows = len(all_indices)

    if total_rows == 0:
        print("✅ All rows already evaluated. Nothing to do.")
        # Still write output file in case it didn't exist yet
        df.to_csv(OUTPUT_FILE, index=False)
        return

    print(f"📊 Loaded {len(df)} total rows, {total_rows} need judge evaluation.")
    print(f"💾 Results will be saved to: {OUTPUT_FILE}\n")

    output_path = Path(OUTPUT_FILE)
    backup_path = output_path.with_suffix(".backup.csv")

    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

    overall_start = time.perf_counter()
    total_tokens_generated = 0

    for batch_num, start_idx in enumerate(range(0, total_rows, BATCH_SIZE), 1):
        batch_indices = all_indices[start_idx : start_idx + BATCH_SIZE]
        print(f"\n🚀 Batch {batch_num}: processing rows {batch_indices}")

        batch_tokens = await process_batch(client, batch_indices, df)
        total_tokens_generated += batch_tokens

        # Incremental save after every batch
        df.to_csv(output_path, index=False)
        df.to_csv(backup_path, index=False)
        print(f"💾 Saved after batch {batch_num} -> {output_path}")

    overall_elapsed = time.perf_counter() - overall_start

    # Summary stats
    parse_ok = (df["judge_parse_status"] == "ok").sum()
    parse_err = (df["judge_parse_status"] == "parse_error").sum()
    req_err = (df["judge_parse_status"] == "request_error").sum()

    print("\n" + "=" * 60)
    print(f"🎉 All done! Processed {total_rows} rows.")
    print(f"   ✔  Parsed OK:      {parse_ok}")
    print(f"   ⚠️  Parse errors:   {parse_err}  (raw response saved in judge_raw)")
    print(f"   ❌ Request errors:  {req_err}  (re-run script to retry)")
    print(f"⏱️  Total time: {overall_elapsed:.2f}s")
    print(f"🔢 Total tokens: {total_tokens_generated}")
    print(
        f"⚡ Throughput: {total_tokens_generated / max(overall_elapsed, 1):.1f} tokens/sec"
    )
    print(f"💾 Final CSV: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
