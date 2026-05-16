import os
import pandas as pd
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

max_seq_length = 4096
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="finetuned_1.5_full_128",
    max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit,
    fast_inference=False,
)
tokenizer = get_chat_template(tokenizer, chat_template="qwen2.5")
FastLanguageModel.for_inference(model)
eval_df = pd.read_csv("what_is_1727.csv")

checkpoint_path = "eval_checkpoint_1.5B.csv"
if os.path.exists(checkpoint_path):
    done_df = pd.read_csv(checkpoint_path)
    results = done_df.to_dict("records")
    start_idx = len(results)
    print(f"Resuming from row {start_idx}")
else:
    results = []
    start_idx = 0

for i, row in eval_df.iloc[start_idx:].iterrows():
    question = row["question"]
    actual_answer = row["answer"]

    messages = [{"role": "user", "content": question}]
    prompt_with_template = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_with_template, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        use_cache=True,
        temperature=0.0,
        do_sample=False,
        repetition_penalty=1.2,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "assistant\n" in generated_text:
        model_output = generated_text.split("assistant\n", 1)[1]
    else:
        model_output = generated_text

    results.append({
        "question": question,
        "model_output": model_output,
        "actual_answer": actual_answer,
    })

    if (i + 1) % 50 == 0:
        pd.DataFrame(results).to_csv(checkpoint_path, index=False)
        print(f"[{i + 1}/{len(eval_df)}] Checkpoint saved — {question[:50]}...")
    elif (i + 1) % 5 == 0:
        print(f"[{i + 1}/{len(eval_df)}] {question[:50]}...")

result_df = pd.DataFrame(results)
result_df.to_csv("evaluation_results_1.5B_overfitted.csv", index=False)
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
print(f"\nSaved {len(result_df)} rows to evaluation_results_1.5B_overfitted.csv")
