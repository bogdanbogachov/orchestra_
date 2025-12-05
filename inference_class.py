from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Model path
MODEL_PATH = "downloaded_models/downloaded_3_2_1b"

# Load model and tokenizer
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Set pad token (Llama tokenizers typically don't have one by default)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Number of labels for classification
num_labels = 5  # Adjust based on your task

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=num_labels,
    dtype=torch.float32,
    device_map="auto"
)


def classify(text, labels=None):
    """
    Classify text using the Llama model with default classification head.

    Args:
        text: Input text string
        labels: Optional tensor of shape [batch_size] for loss computation

    Returns:
        Dictionary with 'logits', 'probs', and optionally 'loss'
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Get hidden states to prove which token is used
    with torch.no_grad():
        # Get base model outputs to inspect hidden states
        base_outputs = model.model(**inputs)
        hidden_states = base_outputs.last_hidden_state  # [B, T, H]

        # Compute logits for ALL tokens (this is what the model does internally)
        all_token_logits = model.score(hidden_states)  # [B, T, num_labels]

        # Determine which token will be used (last non-padding token)
        seq_len = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]
        if model.config.pad_token_id is None:
            last_token_idx = seq_len - 1
        elif inputs.get("input_ids") is not None:
            # Find last non-padding token (same logic as GenericForSequenceClassification)
            non_pad_mask = (inputs["input_ids"] != model.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(seq_len, device=hidden_states.device, dtype=torch.int32)
            last_token_idx = (token_indices * non_pad_mask).argmax(-1).item()
        else:
            last_token_idx = seq_len - 1

        # Manually extract logits from the last token only
        manual_last_token_logits = all_token_logits[
            torch.arange(batch_size, device=all_token_logits.device), last_token_idx]  # [B, num_labels]

        num_tokens = seq_len
        if inputs.get("attention_mask") is not None:
            num_valid_tokens = inputs["attention_mask"].sum().item()
        else:
            num_valid_tokens = num_tokens

    # Print information about pooling strategy
    print(f"\n{'=' * 60}")
    print(f"[DEFAULT CLASSIFIER] Pooling Strategy: LAST TOKEN ONLY")
    print(f"[DEFAULT CLASSIFIER] Total tokens in sequence: {num_tokens}")
    print(f"[DEFAULT CLASSIFIER] Valid (non-padding) tokens: {num_valid_tokens}")
    print(f"[DEFAULT CLASSIFIER] Computed logits for ALL {num_tokens} tokens: shape {all_token_logits.shape}")
    print(f"[DEFAULT CLASSIFIER] Extracting logits from token at index {last_token_idx} (last non-padding token)")
    print(f"[DEFAULT CLASSIFIER] Tokens 0 to {last_token_idx - 1} are IGNORED for classification")
    print(f"{'=' * 60}")

    # Classify using the model (which internally does the same thing)
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)

    # Extract logits from model output
    model_output_logits = outputs.logits  # [B, num_labels]

    # PROOF: Compare manual extraction with model output
    print(f"\n{'=' * 60}")
    print(f"[PROOF] Manual last token logits: {manual_last_token_logits}")
    print(f"[PROOF] Model output logits:      {model_output_logits}")
    print(f"[PROOF] Are they identical? {torch.allclose(manual_last_token_logits, model_output_logits, atol=1e-6)}")
    print(f"[PROOF] Max difference: {torch.abs(manual_last_token_logits - model_output_logits).max().item():.2e}")
    print(f"{'=' * 60}")

    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(model_output_logits, dim=-1)  # [B, num_labels]

    result = {
        'logits': model_output_logits,
        'probs': probs,
        'loss': outputs.loss
    }

    return result

# Example usage
if __name__ == "__main__":
    text = "What is machine learning?"
    result = classify(text)
    
    print(f"\nText: {text}")
    print(f"Default logits: {result['logits']}")
    print(f"Default probabilities: {result['probs']}")
    print(f"Default predicted class: {result['probs'].argmax(dim=-1).item()}")
