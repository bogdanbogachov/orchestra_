import json
import random
from config import CONFIG
from logger_config import logger

# Predefined casual language phrases for TRAIN set (at least 10 words each) with typo patterns
CASUAL_PHRASES_TRAIN = [
    "um like you know what i mean so yeah basically",
    "so yeah i guess that works well actually you see",
    "well actually i think maybe we should consider this",
    "oh wait let me check that real quick before we proceed",
    "hmm i dunno about that one to be completely honest with you",
    "like seriously what the heck is going on here right now",
    "okay so basically what happened was that we tried something new",
    "right so i was thinking maybe we could try a different approach",
    "yeah so like i mean honestly it makes sense when you think about it",
    "well you see the thing is that we need to figure this out together",
    "i mean like honestly though i dont really know what to tell you",
    "so basically what i need is for you to understand the situation better",
    "um yeah so about that i was wondering if we could talk about it",
    "like i dont really know how to explain this but here goes nothing",
    "well i guess we could try something different if that works for you",
    "so like what do you think about this whole situation we are facing",
    "yeah i mean that makes sense when you consider all the factors involved",
    "okay so here is the deal we need to make a decision pretty soon",
    "right so let me explain this to you in a way that makes sense",
    "well actually i was wondering if maybe we should reconsider our options here"
]

# Predefined casual language phrases for TEST set (completely different from train, at least 10 words each)
CASUAL_PHRASES_TEST = [
    "dude that sounds pretty cool to me man i gotta say",
    "bro honestly i have no clue what you are talking about here",
    "man this is getting really complicated if you ask me personally",
    "yo check it out i just realized something important about this whole thing",
    "hey listen up because i need to tell you something really interesting",
    "dang that is wild i never would have thought about it that way",
    "wow that is actually pretty amazing when you stop and think about it",
    "man oh man this is turning into quite the situation we got here",
    "hey you know what i just remembered something that might be relevant",
    "dude seriously this is way more complex than i originally thought it was",
    "bro i gotta be straight with you this whole thing is confusing me",
    "man i wish i could help you out more but i am just not sure",
    "yo real talk this is one of those things that takes time to understand",
    "hey look i am trying my best here but this is pretty difficult",
    "dang it seems like we are going in circles with this conversation here",
    "wow i cannot believe how much we have covered in such a short time",
    "man this conversation is really making me think about things differently now",
    "hey you seem pretty knowledgeable about this stuff which is really helpful",
    "dude i appreciate you taking the time to explain all of this to me",
    "bro honestly i think we are making good progress here so keep going"
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

def _generate_phrases_from_templates(templates, is_train=True):
    """
    Generate a large list of unique phrases from templates.
    This ensures we have enough variety so each phrase repeats <= 25 times.
    """
    phrases = []
    
    # Base templates for train set
    if is_train:
        starters = ["um", "so", "well", "oh", "hmm", "like", "okay", "right", "yeah", "i mean"]
        connectors = ["like", "you know", "i mean", "honestly", "actually", "basically", "so yeah", "well actually"]
        middles = ["what i mean", "what happened", "the thing is", "i was thinking", "i guess", "i dunno", "let me check"]
        endings = ["basically", "you see", "consider this", "we proceed", "honest with you", "right now", "something new", 
                  "different approach", "think about it", "together", "tell you", "situation better", "talk about it", 
                  "here goes nothing", "works for you", "we are facing", "factors involved", "pretty soon", "makes sense",
                  "reconsider our options", "figure this out", "understand better", "try something", "check that"]
    else:
        # Base templates for test set
        starters = ["dude", "bro", "man", "yo", "hey", "dang", "wow"]
        connectors = ["that sounds", "honestly", "this is", "i just", "i need", "that is", "i gotta", "real talk"]
        middles = ["pretty cool", "no clue", "really complicated", "realized something", "tell you something", 
                  "wild i never", "pretty amazing", "turning into", "remembered something", "way more complex"]
        endings = ["i gotta say", "talking about here", "ask me personally", "about this whole thing", "really interesting",
                  "thought about it that way", "stop and think about it", "situation we got here", "might be relevant",
                  "originally thought it was", "whole thing is confusing me", "just not sure", "takes time to understand",
                  "pretty difficult", "going in circles", "such a short time", "think about things differently now",
                  "really helpful", "explain all of this to me", "making good progress"]
    
    # Generate combinations
    for starter in starters:
        for connector in connectors:
            for middle in middles:
                for ending in endings:
                    phrase = f"{starter} {connector} {middle} {ending}"
                    words = phrase.split()
                    if len(words) >= 10:
                        phrases.append(phrase)
                    # Add variations with extra words
                    if len(words) < 10:
                        extra = random.choice(["to me", "man", "you know", "i think", "maybe", "well", "actually"])
                        phrase_extended = f"{phrase} {extra}"
                        if len(phrase_extended.split()) >= 10:
                            phrases.append(phrase_extended)
    
    # Also add the original predefined phrases
    if is_train:
        phrases.extend(CASUAL_PHRASES_TRAIN)
    else:
        phrases.extend(CASUAL_PHRASES_TEST)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_phrases = []
    for phrase in phrases:
        if phrase not in seen:
            seen.add(phrase)
            unique_phrases.append(phrase)
    
    return unique_phrases

def generate_noise_phrase(phrases_list=None, is_train=True):
    """
    Generate a casual language phrase with typos (at least 10 words).
    Returns a string with at least 10 words.
    
    Args:
        phrases_list: List of phrases to choose from (if None, generates dynamically)
        is_train: Whether generating for train set (default: True)
    """
    # Generate phrases dynamically if not provided or if list is too small
    if phrases_list is None or len(phrases_list) < 200:
        # Generate expanded list of phrases
        phrases_list = _generate_phrases_from_templates([], is_train=is_train)
    
    # Select a random phrase
    phrase = random.choice(phrases_list)
    
    # Ensure it has at least 10 words
    words = phrase.split()
    if len(words) < 10:
        # Add more casual words if needed
        extra_words = ["like", "um", "yeah", "so", "well", "actually", "i mean", "you know", "right", "okay"]
        max_iterations = 20  # Safety limit to prevent infinite loops
        iterations = 0
        while len(words) < 10 and iterations < max_iterations:
            words.append(random.choice(extra_words))
            iterations += 1
        phrase = " ".join(words)
    
    # Apply typos
    noisy_phrase = apply_typos(phrase, typo_probability=0.12)
    
    return noisy_phrase

def insert_noise_into_text(text, noise_phrase):
    """
    Insert noise phrase at the beginning of the text.
    """
    if not text or not text.strip():
        return noise_phrase
    
    # Insert noise at the beginning
    return f"{noise_phrase} {text}"

def run_add_noise():
    """
    Add noise (casual language with typos) to every second question in both train_data.json and test_data.json.
    The noise is at least 10 words long.
    Test set uses completely different noise phrases from the train set.
    """
    paths_config = CONFIG['paths']
    train_file = paths_config['data']['train']
    test_file = paths_config['data']['test']
    
    # Process training set
    logger.info(f"Loading training data from {train_file}")
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    train_count = len(train_data)
    logger.info(f"Loaded {train_count} training examples")
    
    train_modified = 0
    for i in range(1, len(train_data), 2):  # Process every second question (indices 1, 3, 5, ...)
        original_text = train_data[i]["text"]
        
        # Generate noise phrase using TRAIN-specific phrases (at least 10 words with typos)
        noise_phrase = generate_noise_phrase(phrases_list=CASUAL_PHRASES_TRAIN, is_train=True)
        
        # Insert noise into the text
        noisy_text = insert_noise_into_text(original_text, noise_phrase)
        
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
        
        # Generate noise phrase using TEST-specific phrases (at least 10 words with typos)
        noise_phrase = generate_noise_phrase(phrases_list=CASUAL_PHRASES_TEST, is_train=False)
        
        # Insert noise into the text
        noisy_text = insert_noise_into_text(original_text, noise_phrase)
        
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
    logger.info(f"✓ Noise augmentation complete!")
    logger.info(f"✓ Training set: {train_modified}/{train_count} examples modified (every second question)")
    logger.info(f"✓ Test set: {test_modified}/{test_count} examples modified (every second question)")
    logger.info(f"✓ Each noise phrase contains at least 10 words with casual typos")
    logger.info(f"✓ Test noise phrases are completely different from train noise phrases")

