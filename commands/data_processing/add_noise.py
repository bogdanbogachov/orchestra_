import json
import random
from config import CONFIG
from logger_config import logger

# Predefined casual language phrases (at least 5 words each) with typo patterns
CASUAL_PHRASES = [
    "um like you know what i mean",
    "so yeah i guess that works",
    "well actually i think maybe",
    "oh wait let me check that",
    "hmm i dunno about that one",
    "like seriously what the heck",
    "okay so basically what happened",
    "right so i was thinking",
    "yeah so like i mean",
    "well you see the thing is",
    "i mean like honestly though",
    "so basically what i need",
    "um yeah so about that",
    "like i dont really know",
    "well i guess we could try",
    "so like what do you think",
    "yeah i mean that makes sense",
    "okay so here is the deal",
    "right so let me explain this",
    "well actually i was wondering"
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

def generate_noise_phrase():
    """
    Generate a casual language phrase with typos (at least 5 words).
    Returns a string with at least 5 words.
    """
    # Select a random phrase
    phrase = random.choice(CASUAL_PHRASES)
    
    # Ensure it has at least 5 words
    words = phrase.split()
    if len(words) < 5:
        # Add more casual words if needed
        extra_words = ["like", "um", "yeah", "so", "well", "actually", "i mean"]
        max_iterations = 10  # Safety limit to prevent infinite loops
        iterations = 0
        while len(words) < 5 and iterations < max_iterations:
            words.append(random.choice(extra_words))
            iterations += 1
        phrase = " ".join(words)
    
    # Apply typos
    noisy_phrase = apply_typos(phrase, typo_probability=0.12)
    
    return noisy_phrase

def insert_noise_into_text(text, noise_phrase):
    """
    Insert noise phrase into the text at a random position.
    """
    words = text.split()
    if len(words) == 0:
        return noise_phrase
    
    # Insert at a random position (not at the very beginning or end)
    insert_position = random.randint(1, len(words))
    words.insert(insert_position, noise_phrase)
    
    return " ".join(words)

def run_add_noise():
    """
    Add noise (casual language with typos) to every second question in train_data.json.
    The noise is at least 5 words long.
    """
    paths_config = CONFIG['paths']
    train_file = paths_config['data']['train']
    
    logger.info(f"Loading training data from {train_file}")
    with open(train_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_count = len(data)
    logger.info(f"Loaded {original_count} training examples")
    
    # Process every second question (indices 1, 3, 5, ...)
    modified_count = 0
    for i in range(1, len(data), 2):  # Start at 1, step by 2
        original_text = data[i]["text"]
        
        # Generate noise phrase (at least 5 words with typos)
        noise_phrase = generate_noise_phrase()
        
        # Insert noise into the text
        noisy_text = insert_noise_into_text(original_text, noise_phrase)
        
        # Update the data
        data[i]["text"] = noisy_text
        modified_count += 1
        
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1} examples...")
    
    # Save the modified data
    logger.info(f"Saving modified data back to {train_file}")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Added noise to {modified_count} out of {original_count} training examples")
    logger.info(f"✓ Modified every second question (indices 1, 3, 5, ...)")
    logger.info(f"✓ Each noise phrase contains at least 5 words with casual typos")

