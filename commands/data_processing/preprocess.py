import json
from sklearn.model_selection import train_test_split
from collections import Counter
from config import CONFIG
from logger_config import logger

def print_statistics(data, name):
    labels = [item["label"] for item in data]
    label_counts = Counter(labels)
    
    logger.info(f"\n{name} Statistics:")
    logger.info(f"  Total examples: {len(data)}")
    logger.info(f"  Label distribution:")
    for label in sorted(label_counts.keys()):
        logger.info(f"    Label {label}: {label_counts[label]} examples")


def run_preprocess():
    paths_config = CONFIG['paths']
    data_config = CONFIG['data_processing']
    
    input_file = paths_config['data']['training']
    train_output = paths_config['data']['train']
    test_output = paths_config['data']['test']
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print_statistics(data, "Original")
    
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels,
        test_size=data_config['test_size'],
        random_state=data_config['random_state'],
        stratify=labels if data_config['stratify'] else None
    )
    
    train_data = [{"text": text, "label": label} for text, label in zip(train_texts, train_labels)]
    test_data = [{"text": text, "label": label} for text, label in zip(test_texts, test_labels)]
    
    print_statistics(train_data, "Training")
    print_statistics(test_data, "Test")
    
    with open(train_output, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(test_output, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    logger.info(f"\n✓ Saved {len(train_data)} training examples to {train_output}")
    logger.info(f"✓ Saved {len(test_data)} test examples to {test_output}")
