from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Model path
MODEL_PATH = "downloaded_models/downloaded_3_2_1b"

# Load model and tokenizer
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    device_map="auto"
)

def infer(prompt, max_new_tokens=1, temperature=0.6, top_p=0.9):
    """Minimalist inference function."""
    # Format prompt for instruct model
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
    
    # Decode
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

# Example usage
if __name__ == "__main__":
    prompt = "What is machine learning? Answer in 1 word."
    result = infer(prompt)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {result}")
