import os

os.environ["VLLM_USE_V1"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

max_seq_length = 4096
load_in_4bit = False

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./pure_sft_small",
    # model_name="./finetuned_0.5b_bft",
    # max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit,
    fast_inference=False,
)
tokenizer = get_chat_template(tokenizer, chat_template="qwen2.5")

messages = []

print(
    "\nInteractive mode started. Type 'quit' or 'exit' to stop, 'clear' to reset conversation."
)
print("Supports: /help, /messages, /stats\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ("quit", "exit"):
        print("Goodbye!")
        break

    if user_input.lower() == "clear":
        messages = []
        print("Conversation cleared.\n")
        continue

    if user_input.lower() == "/help":
        print("Commands:")
        print("  (text)              - Send a message to the model")
        print("  /quit, /exit        - Exit the session")
        print("  /clear              - Clear conversation history")
        print("  /messages           - Show current conversation messages")
        print("  /stats              - Show token count info")
        print("  /history <n>        - Show last n messages")
        print()
        continue

    if user_input.lower() == "/stats":
        print(f"Messages in context: {len(messages)}")
        if messages:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt")
            print(f"Input tokens: {inputs.input_ids.shape[1]}")
        print()
        continue

    if user_input.lower() == "/messages":
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = (
                msg["content"][:100] + "..."
                if len(msg["content"]) > 100
                else msg["content"]
            )
            print(f"  [{i}] {role}: {content}")
        print()
        continue

    if user_input.lower().startswith("/history"):
        try:
            n = int(user_input.split()[1])
            history = messages[-n:] if len(messages) > n else messages
        except (IndexError, ValueError):
            history = messages
        for i, msg in enumerate(history):
            role = msg["role"]
            content = (
                msg["content"][:150] + "..."
                if len(msg["content"]) > 150
                else msg["content"]
            )
            print(f"  [{i}] {role}: {content}")
        print()
        continue

    if not user_input:
        continue

    messages.append({"role": "user", "content": user_input})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print("\nModel: ", end="", flush=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        use_cache=True,
        temperature=0.0,
        do_sample=False,
        repetition_penalty=1.1,
    )

    # After model.generate:
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]   # new tokens only
    assistant_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    messages.append({"role": "assistant", "content": assistant_response})
    print(assistant_response)
    print()
