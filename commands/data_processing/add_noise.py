import json
import random
import re
from math import ceil
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
    "meta": [
        "I'm sorry if I'm missing something obvious",
        "I might be misunderstanding the flow",
        "I just want to make sure I'm doing this correctly",
        "if there's a standard procedure, I'm happy to follow it",
        "I'm trying not to make any mistakes here",
    ],
    "diagnostics": [
        "I already tried logging out and back in",
        "I restarted the app and retried",
        "I cleared cache and tried again",
        "I attempted it from another browser",
        "I double-checked the details before submitting",
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
    "meta": [
        "I would appreciate confirmation of the correct process",
        "I would like to ensure I am following the proper steps",
        "I am seeking clarification to avoid any incorrect action",
        "I would appreciate any guidance you can provide",
        "I would prefer to resolve this in the standard way",
    ],
    "diagnostics": [
        "I have attempted the action multiple times",
        "I have reinstalled the application and retried",
        "I have tried both desktop and mobile",
        "I have verified the information prior to submission",
        "I have confirmed that my connection is stable",
    ],
}

TYPO_PATTERNS = {
    "i": ["i", "I", "1"],
    "e": ["e", "E", "3"],
    "a": ["a", "A", "@"],
    "o": ["o", "O", "0"],
    "s": ["s", "S", "5"],
    "l": ["l", "L", "1"],
    "t": ["t", "T", "7"],
}

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_spaces(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", s).strip()


def safe_append_punct(s: str, punct: str) -> str:
    s = s.strip()
    if not s:
        return s
    if s.endswith((".", "?", "!", ",")):
        return s
    return s + punct


def choose_templates(is_train: bool):
    return NOISE_TEMPLATES_TRAIN if is_train else NOISE_TEMPLATES_TEST


def apply_typos(text: str, typo_probability: float = 0.015) -> str:
    out = []
    for ch in text:
        cl = ch.lower()
        if cl in TYPO_PATTERNS and random.random() < typo_probability:
            r = random.random()
            if r < 0.15:
                continue
            elif r < 0.30:
                out.append(ch)
                out.append(ch)
            else:
                out.append(random.choice(TYPO_PATTERNS[cl]))
        else:
            out.append(ch)
    return "".join(out)


def _rand_sentence(T: dict) -> str:
    """
    A natural 'ticket-like' sentence that adds bulk without changing intent.
    """
    parts = []
    if random.random() < 0.35:
        parts.append(random.choice(T["openers"]))
    if random.random() < 0.85:
        parts.append(random.choice(T["context"]))
    if random.random() < 0.70:
        parts.append(random.choice(T["diagnostics"]))
    if random.random() < 0.70:
        parts.append(random.choice(T["meta"]))
    if random.random() < 0.70:
        parts.append(random.choice(T["justifications"]))
    if random.random() < 0.60:
        parts.append(random.choice(T["soft_urgency"]))
    if random.random() < 0.70:
        parts.append(random.choice(T["followups"]))
    else:
        parts.append(random.choice(T["closers"]))
    # Turn fragments into 1–2 sentences
    s = " ".join(parts)
    s = normalize_spaces(s)
    s = safe_append_punct(s, ".")
    if random.random() < 0.45:
        s2 = normalize_spaces(random.choice(T["meta"]) + ". " + random.choice(T["followups"]))
        s2 = safe_append_punct(s2, ".")
        return f"{s} {s2}"
    return s


def add_realistic_noise(
    text: str,
    is_train: bool,
    target_noise_fraction: float = 0.80,  # <-- what you asked for
    min_orig_tokens_for_strict: int = 6,
    max_total_multiplier: float = 6.0,     # safety
) -> str:
    """
    Enforces: noise_tokens / total_tokens ~= target_noise_fraction
      total_tokens ~= orig_tokens / (1 - target_noise_fraction)

    We do this by appending natural ticket-like sentences until we hit target length.
    """
    if not text or not text.strip():
        return text

    T = choose_templates(is_train)
    original = normalize_spaces(text)
    orig_tokens = original.split()
    orig_len = len(orig_tokens)

    # For very short queries, strict 80% looks absurd; still heavy, but a bit lower.
    if orig_len < min_orig_tokens_for_strict:
        target_noise_fraction = min(target_noise_fraction, 0.65)

    # Desired total length so that original is ~ (1 - f) of final
    desired_total_len = ceil(orig_len / max(1e-6, (1.0 - target_noise_fraction)))
    hard_cap_len = int(orig_len * max_total_multiplier)

    # Start with the original preserved as-is, then add noise after it.
    noisy = safe_append_punct(original, ".")
    while len(noisy.split()) < desired_total_len and len(noisy.split()) < hard_cap_len:
        noisy = normalize_spaces(f"{noisy} {_rand_sentence(T)}")

    # Very light typos rarely (keeps it readable)
    if random.random() < (0.12 if is_train else 0.06):
        noisy = apply_typos(noisy, typo_probability=0.012)

    return noisy


def run_add_noise():
    """
    Add heavy realistic noise to every question in train/test.
    """
    paths_config = CONFIG["paths"]
    train_file = paths_config["data"]["train"]
    test_file = paths_config["data"]["test"]

    logger.info(f"Loading training data from {train_file}")
    with open(train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    logger.info(f"Loaded {len(train_data)} training examples")

    for i in range(len(train_data)):
        train_data[i]["text"] = add_realistic_noise(
            train_data[i]["text"],
            is_train=True,
            target_noise_fraction=0.80,
        )
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1} training examples...")

    logger.info(f"Saving modified training data back to {train_file}")
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\nLoading test data from {test_file}")
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    logger.info(f"Loaded {len(test_data)} test examples")

    for i in range(len(test_data)):
        test_data[i]["text"] = add_realistic_noise(
            test_data[i]["text"],
            is_train=False,
            target_noise_fraction=0.80,
        )
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1} test examples...")

    logger.info(f"Saving modified test data back to {test_file}")
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'=' * 100}")
    logger.info("✓ Heavy realistic noise augmentation complete!")
    logger.info("✓ Noise occupies ~80% of final text (by token fraction).")
    logger.info("✓ Original intent is preserved as the first sentence.")
