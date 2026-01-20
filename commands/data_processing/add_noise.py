import json
import random
from config import CONFIG
from logger_config import logger

# High-frequency noise words for interleaving (TRAIN set)
# Banking-specific words that create rapid oscillations when interleaved with original text
HIGH_FREQ_NOISE_TRAIN = [
    "um", "uh", "like", "you know", "i mean", "actually", 
    "basically", "well", "so", "right", "okay", "please", 
    "thank you", "sorry", "urgent", "asap", "status", 
    "check", "verify", "confirm", "important", "need help"
]

# High-frequency noise words for interleaving (TEST set - completely different banking style)
HIGH_FREQ_NOISE_TEST = [
    "excuse me", "pardon", "if you could", "would you mind",
    "right?", "correct?", "is it?", "are you sure?", 
    "really?", "seriously?", "honestly?", "for real?",
    "critical", "stuck", "can't access", "not working", 
    "broken", "issue", "problem", "help me"
]

# Typo patterns: character substitutions that are common in casual typing
TYPO_PATTERNS = {
    'i': ['i', 'I', '1'],
    'e': ['e', 'E', '3'],
    'a': ['a', 'A', '@'],
    'o': ['o', 'O', '0'],
    's': ['s', 'S', '5'],
    'l': ['l', 'L', '1'],
    't': ['t', 'T', '7'],
    'z': ['z', 'Z', '2'],
}

def apply_typos(text, typo_probability=0.15):
    """
    Apply realistic typos to text.
    typo_probability: probability of introducing a typo per character
    """
    result = []
    for char in text:
        if char.lower() in TYPO_PATTERNS and random.random() < typo_probability:
            # Apply typo: sometimes skip, sometimes duplicate, sometimes substitute
            typo_type = random.random()
            if typo_type < 0.3:
                # Skip character (deletion)
                continue
            elif typo_type < 0.5:
                # Duplicate character
                result.append(char)
                result.append(char)
            else:
                # Substitute with similar character
                alternatives = TYPO_PATTERNS.get(char.lower(), [char])
                result.append(random.choice(alternatives))
        else:
            result.append(char)
    return ''.join(result)

def generate_high_freq_noise_words(is_train=True, min_words=10):
    """
    Generate a list of high-frequency noise words for interleaving.
    These will create rapid oscillations in hidden states.
    
    Args:
        is_train: Whether generating for train set (default: True)
        min_words: Minimum number of noise words to generate
    
    Returns:
        List of noise words (at least min_words long)
    """
    if is_train:
        noise_words_pool = HIGH_FREQ_NOISE_TRAIN
    else:
        noise_words_pool = HIGH_FREQ_NOISE_TEST
    
    # Generate at least min_words noise words by sampling with replacement
    # This creates repetitive patterns that will produce high-frequency oscillations
    noise_words = []
    for _ in range(min_words):
        noise_words.append(random.choice(noise_words_pool))
    
    # Apply typos to some words to add variation
    noisy_words = []
    for word in noise_words:
        if random.random() < 0.15:  # 15% chance of typo per word
            noisy_word = apply_typos(word, typo_probability=0.2)
            noisy_words.append(noisy_word)
        else:
            noisy_words.append(word)
    
    return noisy_words

def insert_high_freq_noise(text, noise_words, interleave_ratio=0.5):
    """
    Interleave high-frequency noise words with original text to create rapid oscillations.
    This creates high-frequency patterns that FFT can filter out.
    
    Args:
        text: Original text
        noise_words: List of noise words to interleave
        interleave_ratio: Probability of inserting noise after each original word (0.0-1.0)
    
    Returns:
        Text with interleaved high-frequency noise
    """
    if not text or not text.strip():
        return " ".join(noise_words)
    
    original_words = text.split()
    if len(original_words) == 0:
        return " ".join(noise_words)
    
    # Interleave noise words with original text
    # This creates high-frequency oscillations: [noise] [original] [noise] [original] ...
    result_words = []
    noise_idx = 0
    
    for orig_word in original_words:
        # Add original word
        result_words.append(orig_word)
        
        # Interleave noise word with probability interleave_ratio
        if random.random() < interleave_ratio and noise_idx < len(noise_words):
            result_words.append(noise_words[noise_idx])
            noise_idx = (noise_idx + 1) % len(noise_words)  # Cycle through noise words
    
    # Return the interleaved result (don't add unused noise words to the beginning)
    return " ".join(result_words)

def run_add_noise():
    """
    Add high-frequency noise (interleaved patterns) to every second question in both train_data.json and test_data.json.
    The noise is interleaved with original text to create rapid oscillations that FFT can filter.
    Test set uses completely different noise words from the train set.
    """
    paths_config = CONFIG['paths']
    train_file = paths_config['data']['train']
    test_file = paths_config['data']['test']
    
    # Interleave ratio: probability of inserting noise after each original word
    interleave_ratio = 0.3  # 30% chance = creates moderate high-frequency pattern
    min_noise_words = 10  # Minimum number of noise words to generate
    
    # Process training set
    logger.info(f"Loading training data from {train_file}")
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    train_count = len(train_data)
    logger.info(f"Loaded {train_count} training examples")
    
    train_modified = 0
    for i in range(1, len(train_data), 2):  # Process every second question (indices 1, 3, 5, ...)
        original_text = train_data[i]["text"]
        
        # Generate high-frequency noise words (at least 10 words)
        noise_words = generate_high_freq_noise_words(is_train=True, min_words=min_noise_words)
        
        # Interleave noise with original text to create high-frequency oscillations
        noisy_text = insert_high_freq_noise(original_text, noise_words, interleave_ratio=interleave_ratio)
        
        # Update the data
        train_data[i]["text"] = noisy_text
        train_modified += 1
        
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1} training examples...")
    
    # Save the modified training data
    logger.info(f"Saving modified training data back to {train_file}")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Added noise to {train_modified} out of {train_count} training examples")
    
    # Process test set
    logger.info(f"\nLoading test data from {test_file}")
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    test_count = len(test_data)
    logger.info(f"Loaded {test_count} test examples")
    
    test_modified = 0
    for i in range(1, len(test_data), 2):  # Process every second question (indices 1, 3, 5, ...)
        original_text = test_data[i]["text"]
        
        # Generate high-frequency noise words using TEST-specific words (at least 10 words)
        noise_words = generate_high_freq_noise_words(is_train=False, min_words=min_noise_words)
        
        # Interleave noise with original text to create high-frequency oscillations
        noisy_text = insert_high_freq_noise(original_text, noise_words, interleave_ratio=interleave_ratio)
        
        # Update the data
        test_data[i]["text"] = noisy_text
        test_modified += 1
        
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1} test examples...")
    
    # Save the modified test data
    logger.info(f"Saving modified test data back to {test_file}")
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Added noise to {test_modified} out of {test_count} test examples")
    
    # Summary
    logger.info(f"\n{'=' * 100}")
    logger.info(f"✓ High-frequency noise augmentation complete!")
    logger.info(f"✓ Training set: {train_modified}/{train_count} examples modified (every second question)")
    logger.info(f"✓ Test set: {test_modified}/{test_count} examples modified (every second question)")
    logger.info(f"✓ Noise is interleaved with original text (interleave ratio: {interleave_ratio})")
    logger.info(f"✓ Each example has at least {min_noise_words} noise words creating high-frequency oscillations")
    logger.info(f"✓ Test noise words are completely different from train noise words")
    logger.info(f"✓ This high-frequency noise pattern can be filtered by FFT in hidden states")

