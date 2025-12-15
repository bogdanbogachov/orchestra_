import json
from sklearn.model_selection import train_test_split
from collections import Counter


def print_statistics(data, name):
    """Print statistics about the dataset."""
    labels = [item["label"] for item in data]
    label_counts = Counter(labels)
    
    print(f"\n{name} Statistics:")
    print(f"  Total examples: {len(data)}")
    print(f"  Label distribution:")
    for label in sorted(label_counts.keys()):
        print(f"    Label {label}: {label_counts[label]} examples")


def main():
    input_file = "training_data.json"
    train_output = "train_data.json"
    test_output = "test_data.json"
    
    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print_statistics(data, "Original")
    
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    
    # Split data with stratification
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    
    # Reconstruct JSON format
    train_data = [{"text": text, "label": label} for text, label in zip(train_texts, train_labels)]
    test_data = [{"text": text, "label": label} for text, label in zip(test_texts, test_labels)]
    
    print_statistics(train_data, "Training")
    print_statistics(test_data, "Test")
    
    # Save files
    with open(train_output, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(test_output, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"\n✓ Saved {len(train_data)} training examples to {train_output}")
    print(f"✓ Saved {len(test_data)} test examples to {test_output}")


if __name__ == "__main__":
    main()
