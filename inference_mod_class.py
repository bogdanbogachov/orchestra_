from transformers import AutoTokenizer, AutoModel
from transformers.models.llama.modeling_llama import LlamaModel, LlamaClassificationHead
import torch

# Model path
MODEL_PATH = "downloaded_models/downloaded_3_2_1b"

# Load model and tokenizer
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Set pad token (Llama tokenizers typically don't have one by default)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModel.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,
    device_map="auto"
)

# Create classification head
num_labels = 5  # Adjust based on your task
pooling_strategy = "mean"  # Options: "mean", "max", "last", "attention"
classifier = LlamaClassificationHead(
    config=base_model.config,
    num_labels=num_labels,
    pooling_strategy=pooling_strategy
).to(base_model.device)

def classify(text, labels=None):
    """
    Classify text using the Llama model with custom classification head.
    
    Args:
        text: Input text string
        labels: Optional tensor of shape [batch_size] for loss computation
    
    Returns:
        Dictionary with 'logits', 'probs', and optionally 'loss'
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(base_model.device) for k, v in inputs.items()}
    
    # Get hidden states from base model
    with torch.no_grad():
        outputs = base_model(**inputs)
        hidden_states = outputs.last_hidden_state  # [B, T, H]
    
    # Print information about pooling strategy
    seq_len = hidden_states.shape[1]
    num_tokens = seq_len
    if inputs.get("attention_mask") is not None:
        num_valid_tokens = inputs["attention_mask"].sum().item()
    else:
        num_valid_tokens = num_tokens
    
    print(f"\n{'='*60}")
    print(f"[CUSTOM CLASSIFIER] Pooling Strategy: {pooling_strategy}")
    print(f"[CUSTOM CLASSIFIER] Total tokens in sequence: {num_tokens}")
    print(f"[CUSTOM CLASSIFIER] Valid (non-padding) tokens: {num_valid_tokens}")
    print(f"[CUSTOM CLASSIFIER] ALL {num_valid_tokens} tokens will be pooled together using {pooling_strategy} pooling")
    print(f"{'='*60}")
    
    # Classify
    result = classifier(
        hidden_states=hidden_states,
        attention_mask=inputs.get("attention_mask"),
        labels=labels
    )
    
    return result

# Example usage
if __name__ == "__main__":
    text = "What is machine learning?"
    result = classify(text)
    
    print(f"\nText: {text}")
    print(f"Logits: {result['logits']}")
    print(f"Probabilities: {result['probs']}")
    print(f"Predicted class: {result['probs'].argmax(dim=-1).item()}")
