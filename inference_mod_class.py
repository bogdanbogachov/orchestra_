from transformers import AutoTokenizer, AutoModel
from custom_llama_classification import LlamaClassificationHead
import torch


def classify(input_text, labels=None):
    model_path = "downloaded_models/downloaded_3_2_1b"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModel.from_pretrained(
        model_path,
        dtype=torch.float32,
        device_map="auto"
    )

    classifier = LlamaClassificationHead(
        config=base_model.config,
        num_labels=5,
        pooling_strategy="mean"  # Options: "mean", "max", "last", "attention"
    ).to(base_model.device)

    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(base_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = base_model(**inputs)
        hidden_states = outputs.last_hidden_state  # [B, T, H]

    output = classifier(
        hidden_states=hidden_states,
        attention_mask=inputs.get("attention_mask"),
        labels=labels
    )
    
    return output

if __name__ == "__main__":
    text = "What is machine learning?"
    result = classify(text)
    
    print(f"\nText: {text}")
    print(f"Logits: {result['logits']}")
    print(f"Probabilities: {result['probs']}")
    print(f"Predicted class: {result['probs'].argmax(dim=-1).item()}")
