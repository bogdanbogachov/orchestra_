from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def classify(input_text, labels=None):
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

    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)

    model_output_logits = outputs.logits  # [B, num_labels]

    probs = torch.nn.functional.softmax(model_output_logits, dim=-1)  # [B, num_labels]

    result = {
        'logits': model_output_logits,
        'probs': probs,
        'loss': outputs.loss
    }

    return result

if __name__ == "__main__":
    text = "What is machine learning?"
    result = classify(text)
    
    print(f"\nText: {text}")
    print(f"Default logits: {result['logits']}")
    print(f"Default probabilities: {result['probs']}")
    print(f"Default predicted class: {result['probs'].argmax(dim=-1).item()}")
