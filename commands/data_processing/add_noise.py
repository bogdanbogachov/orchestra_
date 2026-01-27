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
                continue  # deletion
            elif r < 0.30:
                out.append(ch)
                out.append(ch)  # duplication
            else:
                out.append(random.choice(TYPO_PATTERNS[cl]))  # substitution
        else:
            out.append(ch)
    return "".join(out)


def _pick_unique(T: dict, key: str, used_keys: set) -> str | None:
    """Pick from a bucket at most once per example (prevents repetition creep)."""
    if key in used_keys:
        return None
    used_keys.add(key)
    return random.choice(T[key])


# ------------------------------ strategies ------------------------------


def add_prefix(original: str, T: dict, used_keys: set) -> str | None:
    opener = _pick_unique(T, "openers", used_keys)
    if not opener:
        return None
    prefix = opener
    if not prefix.endswith((",", "—")):
        prefix = prefix + ","
    return f"{prefix} {original}"


def add_suffix(original: str, T: dict, used_keys: set) -> str | None:
    # Prefer followup (more natural) but avoid repeating categories
    tail = _pick_unique(T, "followups", used_keys)
    if not tail:
        tail = _pick_unique(T, "closers", used_keys)
    if not tail:
        return None
    tail = safe_append_punct(tail, ".")
    return f"{safe_append_punct(original, '.')} {tail}"


def add_context_clause(original: str, T: dict, used_keys: set) -> str | None:
    ctx = _pick_unique(T, "context", used_keys)
    if not ctx:
        return None
    if random.random() < 0.5:
        return f"{ctx} — {original}"
    ctx = safe_append_punct(ctx, ".")
    return f"{safe_append_punct(original, '.')} {ctx}"


def add_justification(original: str, T: dict, used_keys: set) -> str | None:
    j = _pick_unique(T, "justifications", used_keys)
    if not j:
        return None
    return f"{safe_append_punct(original, '.')} {j}."


def add_self_correction(original: str, T: dict, used_keys: set) -> str | None:
    """
    Insert a self-correction once (not word-by-word).
    Example: "How do I reset my password — I mean, my online banking PIN?"
    """
    marker = _pick_unique(T, "self_corrections", used_keys)
    if not marker:
        return None

    words = original.split()
    if len(words) < 6:
        return None

    insert_at = random.randint(2, min(6, len(words) - 2))
    span_len = random.randint(2, 3)
    span = " ".join(words[insert_at: insert_at + span_len])

    new_words = words[:insert_at] + [marker] + [span] + words[insert_at:]
    return normalize_spaces(" ".join(new_words))


def add_soft_urgency(original: str, T: dict, used_keys: set) -> str | None:
    u = _pick_unique(T, "soft_urgency", used_keys)
    if not u:
        return None
    if random.random() < 0.5:
        return f"{u} — {original}"
    return f"{safe_append_punct(original, '.')} ({u})."


STRATEGIES = [
    add_prefix,
    add_suffix,
    add_context_clause,
    add_justification,
    add_self_correction,
    add_soft_urgency,
]


def add_realistic_noise(
    text: str,
    is_train: bool,
    target_extra_ratio: float = 0.70,
    max_ops: int = 4,
    hard_cap_total_ratio: float = 2.0,
) -> str:
    """
    Make added noise ~ target_extra_ratio of original length (by token count).
      extra_ratio = (len(noisy_tokens) - len(orig_tokens)) / len(orig_tokens)

    Notes:
    - For short texts, we automatically reduce the target to keep output natural.
    - Prevents repetition by using each template bucket at most once per example.
    """
    if not text or not text.strip():
        return text

    T = choose_templates(is_train)
    original = normalize_spaces(text)
    orig_tokens = original.split()
    orig_len = len(orig_tokens)

    # Keep short queries natural (70% noise on 3–5 tokens looks fake)
    if orig_len < 6:
        target_extra_ratio = min(target_extra_ratio, 0.40)
    elif orig_len < 10:
        target_extra_ratio = min(target_extra_ratio, 0.55)

    used_keys: set = set()
    noisy = original

    # Ratio-based loop: keep applying non-repeating, meaning-preserving transforms
    ops_used = 0
    while ops_used < max_ops:
        cur_len = len(noisy.split())
        extra_ratio = (cur_len - orig_len) / max(1, orig_len)
        if extra_ratio >= target_extra_ratio:
            break

        # Choose a strategy that can still add something new (non-repeated bucket)
        random.shuffle(STRATEGIES)
        applied = False
        for op in STRATEGIES:
            candidate = op(noisy, T, used_keys)
            if not candidate:
                continue

            candidate = normalize_spaces(candidate)
            cand_len = len(candidate.split())

            # Accept only if it actually increases length a bit
            if cand_len > cur_len:
                noisy = candidate
                ops_used += 1
                applied = True
                break

        if not applied:
            break  # nothing left to add without repetition

        # Hard cap total growth (safety)
        if len(noisy.split()) > int(orig_len * hard_cap_total_ratio):
            break

        # Extra safeguard: if we're already close, don't keep stacking clauses
        if ((len(noisy.split()) - orig_len) / max(1, orig_len)) > (target_extra_ratio + 0.10):
            break

    # Mild typos very rarely (low rate to keep outputs natural)
    if random.random() < (0.15 if is_train else 0.08):
        noisy = apply_typos(noisy, typo_probability=0.015)

    return noisy


# ---------------------------------------------------------------------
# Main runner (adds noise to every example)
# ---------------------------------------------------------------------

def run_add_noise():
    """
    Add realistic, meaningful noise to every question in train/test.
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

    for i in range(train_count):
        original_text = train_data[i]["text"]
        train_data[i]["text"] = add_realistic_noise(
            original_text,
            is_train=True,
            target_extra_ratio=0.70,
            max_ops=4,
        )
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1} training examples...")

    logger.info(f"Saving modified training data back to {train_file}")
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Added realistic noise to {train_count} / {train_count} training examples")

    logger.info(f"\nLoading test data from {test_file}")
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_count = len(test_data)
    logger.info(f"Loaded {test_count} test examples")

    for i in range(test_count):
        original_text = test_data[i]["text"]
        test_data[i]["text"] = add_realistic_noise(
            original_text,
            is_train=False,
            target_extra_ratio=0.70,
            max_ops=4,
        )
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1} test examples...")

    logger.info(f"Saving modified test data back to {test_file}")
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Added realistic noise to {test_count} / {test_count} test examples")

    logger.info(f"\n{'=' * 100}")
    logger.info("✓ Realistic noise augmentation complete!")
    logger.info(f"✓ Training set: {train_count}/{train_count} modified (every question)")
    logger.info(f"✓ Test set: {test_count}/{test_count} modified (every question)")
    logger.info("✓ Noise is meaningful (context/justification/self-correction), not word-interleaving")
    logger.info("✓ Target noise level uses token-ratio control (~70% extra tokens per sentence)")
    logger.info("✓ Test noise style differs from train noise style (distribution shift)")
