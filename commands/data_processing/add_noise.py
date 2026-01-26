import json
import random
import re
from config import CONFIG
from logger_config import logger


# Train noise: casual + chatty
NOISE_TEMPLATES_TRAIN = {
    "openers": [
        "Hi", "Hey", "Hello", "Quick question", "Sorry to bother you",
        "Not sure if this is the right place, but", "Just checking"
    ],
    "closers": [
        "thanks", "thank you", "appreciate it", "please help", "can you check?"
    ],
    "justifications": [
        "because I need to sort this out today",
        "since my payment is due soon",
        "as I'm trying to reconcile my statements",
        "because I'm seeing something weird in the app",
        "since I don't want to get charged extra"
    ],
    "context": [
        "I'm using the mobile app",
        "I'm on the website",
        "I tried again just now",
        "it started happening this morning",
        "I spoke to support earlier",
        "I travel a lot, not sure if that matters",
        "I'm abroad at the moment"
    ],
    "self_corrections": [
        "— I mean", "— sorry, I meant", "(sorry, I mean)", "(I mean)"
    ],
    "followups": [
        "can you confirm the status?",
        "what should I do next?",
        "is there anything I need to verify?",
        "can you walk me through it?"
    ],
    "soft_urgency": [
        "it's kind of urgent", "ASAP if possible", "today if you can"
    ],
}

# Test noise: different style (more formal / cautious / interrogative)
NOISE_TEMPLATES_TEST = {
    "openers": [
        "Good afternoon", "Hello there", "Could you please advise",
        "I would like to ask", "I’m hoping you can clarify"
    ],
    "closers": [
        "many thanks", "kind regards", "thank you in advance", "please advise"
    ],
    "justifications": [
        "as this affects my account balance",
        "because I need this for my records",
        "as I’m trying to ensure everything is correct",
        "since I’m concerned about a potential charge",
        "because this appears inconsistent"
    ],
    "context": [
        "I have attempted this multiple times",
        "I’m currently unable to access the feature",
        "I’m receiving an error message",
        "this occurs on both Wi-Fi and mobile data",
        "I have updated the app recently",
        "I’m using a different device than usual"
    ],
    "self_corrections": [
        "— rather", "— to clarify", "(to clarify)", "(rather)"
    ],
    "followups": [
        "could you confirm what is happening?",
        "what is the expected timeline?",
        "what are the next steps?",
        "is there a formal way to resolve this?"
    ],
    "soft_urgency": [
        "this is time-sensitive", "at your earliest convenience", "as soon as possible"
    ],
}

# Typos: keep them mild and mostly on *noise*, not on the original user text
TYPO_PATTERNS = {
    "i": ["i", "I", "1"],
    "e": ["e", "E", "3"],
    "a": ["a", "A", "@"],
    "o": ["o", "O", "0"],
    "s": ["s", "S", "5"],
    "l": ["l", "L", "1"],
    "t": ["t", "T", "7"],
}

def apply_typos(text: str, typo_probability: float = 0.05) -> str:
    """
    Mild, realistic typos.
    Apply ONLY to noise fragments for realism without corrupting intent.
    """
    out = []
    for ch in text:
        cl = ch.lower()
        if cl in TYPO_PATTERNS and random.random() < typo_probability:
            r = random.random()
            if r < 0.15:
                # deletion
                continue
            elif r < 0.30:
                # duplication
                out.append(ch)
                out.append(ch)
            else:
                out.append(random.choice(TYPO_PATTERNS[cl]))
        else:
            out.append(ch)
    return "".join(out)

# --- helpers ----------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
def normalize_spaces(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", s).strip()

def pick(pool, k=1):
    if k == 1:
        return random.choice(pool)
    return random.sample(pool, k=min(k, len(pool)))

def safe_append_punct(s: str, punct: str) -> str:
    s = s.strip()
    if not s:
        return s
    if s.endswith((".", "?", "!", ",")):
        return s
    return s + punct

def choose_templates(is_train: bool):
    return NOISE_TEMPLATES_TRAIN if is_train else NOISE_TEMPLATES_TEST

def should_keep_numbers_intact(text: str) -> bool:
    # Banking queries often include amounts/IDs; we avoid touching the original anyway.
    return bool(re.search(r"\d", text))

# --- noise strategies -------------------------------------------------

def add_prefix(original: str, T: dict) -> str:
    opener = pick(T["openers"])
    # Some openers are phrases that already contain commas/clauses
    prefix = opener
    if not prefix.endswith((",", "—")):
        prefix = prefix + ","
    return f"{prefix} {original}"

def add_suffix(original: str, T: dict) -> str:
    tail = pick(T["followups"]) if random.random() < 0.65 else pick(T["closers"])
    tail = safe_append_punct(tail, ".")
    return f"{safe_append_punct(original, '.')} {tail}"

def add_context_clause(original: str, T: dict) -> str:
    ctx = pick(T["context"])
    # attach as a short clause; keep it natural
    if random.random() < 0.5:
        # prefix context
        return f"{ctx} — {original}"
    else:
        # suffix context
        ctx = safe_append_punct(ctx, ".")
        return f"{safe_append_punct(original, '.')} {ctx}"

def add_justification(original: str, T: dict) -> str:
    j = pick(T["justifications"])
    # Put justification at the end; doesn't change label but adds realism
    return f"{safe_append_punct(original, '.')} {j}."

def add_self_correction(original: str, T: dict) -> str:
    """
    Insert a self-correction once (not word-by-word).
    Example: "How do I reset my password — I mean, my online banking PIN?"
    """
    words = original.split()
    if len(words) < 6:
        return original  # too short; skip

    insert_at = random.randint(2, min(6, len(words)-2))
    marker = pick(T["self_corrections"])

    # Create a tiny correction phrase without changing core content:
    # duplicate the next 2-3 words as "clarification"
    span_len = random.randint(2, 3)
    span = " ".join(words[insert_at: insert_at + span_len])

    # Build: "... <marker> <span> ..."
    new_words = words[:insert_at] + [marker] + [span] + words[insert_at:]
    return normalize_spaces(" ".join(new_words))

def add_soft_urgency(original: str, T: dict) -> str:
    u = pick(T["soft_urgency"])
    # keep it lightweight and not label-shifting
    if random.random() < 0.5:
        return f"{u} — {original}"
    return f"{safe_append_punct(original, '.')} ({u})."

def add_parenthetical(original: str, T: dict) -> str:
    """
    Add a parenthetical that feels like a real user aside.
    """
    aside = pick(T["context"]) if random.random() < 0.6 else pick(T["justifications"])
    return f"{original} ({aside})."

STRATEGIES = [
    add_prefix,
    add_suffix,
    add_context_clause,
    add_justification,
    add_self_correction,
    add_soft_urgency,
    add_parenthetical,
]

def add_realistic_noise(text: str, is_train: bool, max_extra_ratio: float = 0.45) -> str:
    """
    Add 1–2 realistic noise fragments using a randomly chosen strategy.
    Caps growth so we don't drown the original intent.
    """
    if not text or not text.strip():
        return text

    T = choose_templates(is_train)

    original = normalize_spaces(text)
    orig_len = len(original.split())

    # pick 1 or 2 strategies
    num_ops = 1 if random.random() < 0.70 else 2
    ops = random.sample(STRATEGIES, k=num_ops)

    noisy = original
    for op in ops:
        noisy = normalize_spaces(op(noisy, T))

    # cap length growth
    noisy_len = len(noisy.split())
    if noisy_len > int(orig_len * (1 + max_extra_ratio)):
        # If too long, fall back to a lighter transformation (prefix or suffix only)
        noisy = normalize_spaces(add_prefix(original, T) if random.random() < 0.5 else add_suffix(original, T))

    # Mild typos on *added* noise only is hard to isolate cleanly without diffing.
    # Pragmatic approach: apply typos rarely to the full string, but at very low rate.
    if random.random() < (0.20 if is_train else 0.10):
        noisy = apply_typos(noisy, typo_probability=0.02)

    return noisy

# ---------------------------------------------------------------------
# Main runner (same behavior: every second example)
# ---------------------------------------------------------------------

def run_add_noise():
    """
    Add realistic, meaningful noise to every second question in train/test.
    Train and test use different noise styles (distribution shift).
    """
    paths_config = CONFIG["paths"]
    train_file = paths_config["data"]["train"]
    test_file = paths_config["data"]["test"]

    logger.info(f"Loading training data from {train_file}")
    with open(train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    train_count = len(train_data)
    logger.info(f"Loaded {train_count} training examples")

    train_modified = 0
    for i in range(1, len(train_data), 2):
        original_text = train_data[i]["text"]
        train_data[i]["text"] = add_realistic_noise(original_text, is_train=True)
        train_modified += 1

        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1} training examples...")

    logger.info(f"Saving modified training data back to {train_file}")
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Added realistic noise to {train_modified} / {train_count} training examples")

    logger.info(f"\nLoading test data from {test_file}")
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_count = len(test_data)
    logger.info(f"Loaded {test_count} test examples")

    test_modified = 0
    for i in range(1, len(test_data), 2):
        original_text = test_data[i]["text"]
        test_data[i]["text"] = add_realistic_noise(original_text, is_train=False)
        test_modified += 1

        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1} test examples...")

    logger.info(f"Saving modified test data back to {test_file}")
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Added realistic noise to {test_modified} / {test_count} test examples")

    logger.info(f"\n{'=' * 100}")
    logger.info("✓ Realistic noise augmentation complete!")
    logger.info(f"✓ Training set: {train_modified}/{train_count} modified (every second question)")
    logger.info(f"✓ Test set: {test_modified}/{test_count} modified (every second question)")
    logger.info("✓ Noise is meaningful (context/justification/self-correction), not word-interleaving")
    logger.info("✓ Test noise style differs from train noise style (distribution shift)")
