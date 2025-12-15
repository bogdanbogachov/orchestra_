from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch


def classify(input_text, labels=None, adapter_path=None):
    model_path = "downloaded_models/downloaded_3_2_1b"

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=5,
        dtype=torch.float32,
        device_map="auto"
    )

    if adapter_path:
        print(f"Loading LoRA adapters from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("✓ LoRA adapters loaded")

    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)

    model_output_logits = outputs.logits  # [B, num_labels]

    probs = torch.nn.functional.softmax(model_output_logits, dim=-1)  # [B, num_labels]

    output = {
        'logits': model_output_logits,
        'probs': probs,
        'loss': outputs.loss
    }

    return output

if __name__ == "__main__":
    text = "What is machine learning my dear smart amazing friend lol?"
    
    result = classify(text, adapter_path="path")
    
    print(f"\nText: {text}")
    print(f"Predicted class: {result['probs'].argmax(dim=-1).item()}")
    print(f"Class probabilities: {result['probs'].tolist()}")
