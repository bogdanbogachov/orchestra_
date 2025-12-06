from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def infer(input_prompt, max_new_tokens=1, temperature=0.6, top_p=0.9):
    model_path = "downloaded_models/downloaded_3_2_1b"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    messages = [{"role": "user", "content": input_prompt}]
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    prompt = "What is machine learning? Answer in 1 word."
    result = infer(prompt)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {result}")
