from transformers import AutoTokenizer, AutoModel
from custom_llama_classification import LlamaClassificationHead
from peft import PeftModel
import torch
import os


def classify(input_text, labels=None, adapter_path=None, pooling_strategy="mean", use_fft=True):
    model_path = "downloaded_models/downloaded_3_2_1b"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModel.from_pretrained(
        model_path,
        dtype=torch.float32,
        device_map="auto"
    )

    if adapter_path:
        base_model = PeftModel.from_pretrained(base_model, adapter_path)
        print("✓ LoRA adapters loaded")

        classifier_path = os.path.join(adapter_path, "classifier.pt")
        if os.path.exists(classifier_path):
            classifier = torch.load(classifier_path, map_location=base_model.device)
            print("✓ Fine-tuned classifier head loaded from adapter")
        else:
            classifier = LlamaClassificationHead(
                config=base_model.config,
                num_labels=50,
                pooling_strategy=pooling_strategy,
                use_fft=use_fft
            ).to(base_model.device)
    else:
        classifier = LlamaClassificationHead(
            config=base_model.config,
            num_labels=50,
            pooling_strategy=pooling_strategy,
            use_fft=use_fft
        ).to(base_model.device)
        print("✓ Randomly initialized classifier loaded")

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
    text = "What is machine learning my dear smart amazing friend lol?"
    
    result = classify(
        text,
        adapter_path="path",
        pooling_strategy="mean",
        use_fft=True
    )
    
    print(f"\nText: {text}")
    print(f"Predicted class: {result['probs'].argmax(dim=-1).item()}")
    print(f"Class probabilities: {result['probs'].tolist()}")
