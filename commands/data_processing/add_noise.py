import json
import random
import re
from typing import Dict, List, Tuple
from config import CONFIG
from logger_config import logger

# ----------------------------
# Noise templates
# ----------------------------

# Banking77-ish (domain flavored)
NOISE_TEMPLATES_BANKING_TRAIN = {
    "openers": [
        "Hi", "Hey", "Hello", "Quick question", "Sorry to bother you",
        "Not sure if this is the right place, but", "Just checking"
    ],
    "closers": ["thanks", "thank you", "appreciate it", "please help", "can you check?"],
    "justifications": [
        "because I need to sort this out today",
        "since my payment is due soon",
        "as I'm trying to reconcile my statements",
        "because I'm seeing something weird in the app",
        "since I don't want to get charged extra",
    ],
    "context": [
        "I'm using the mobile app",
        "I'm on the website",
        "I tried again just now",
        "it started happening this morning",
        "I spoke to support earlier",
        "I'm abroad at the moment",
    ],
    "self_corrections": ["— I mean", "— sorry, I meant", "(sorry, I mean)", "(I mean)"],
    "followups": [
        "can you confirm the status?",
        "what should I do next?",
        "is there anything I need to verify?",
        "can you walk me through it?",
    ],
    "soft_urgency": ["it's kind of urgent", "ASAP if possible", "today if you can"],
}

NOISE_TEMPLATES_BANKING_TEST = {
    "openers": [
        "Good afternoon", "Hello there", "Could you please advise",
        "I would like to ask", "I'm hoping you can clarify"
    ],
    "closers": ["many thanks", "kind regards", "thank you in advance", "please advise"],
    "justifications": [
        "as this affects my account balance",
        "because I need this for my records",
        "as I'm trying to ensure everything is correct",
        "since I'm concerned about a potential charge",
        "because this appears inconsistent",
    ],
    "context": [
        "I have attempted this multiple times",
        "I'm currently unable to access the feature",
        "I'm receiving an error message",
        "this occurs on both Wi-Fi and mobile data",
        "I have updated the app recently",
        "I'm using a different device than usual",
    ],
    "self_corrections": ["— rather", "— to clarify", "(to clarify)", "(rather)"],
    "followups": [
        "could you confirm what is happening?",
        "what is the expected timeline?",
        "what are the next steps?",
        "is there a formal way to resolve this?",
    ],
    "soft_urgency": ["this is time-sensitive", "at your earliest convenience", "as soon as possible"],
}

# CLINC150 (domain-neutral; avoid "account/app/support/interface" bias)
NOISE_TEMPLATES_CLINC_TRAIN = {
    "openers": [
        "Hey", "Hi", "Hello", "Quick question", "I was wondering",
        "Can you help me", "One sec",
    ],
    "closers": ["thanks", "thank you", "appreciate it", "that helps"],
    "justifications": [
        "because I'm trying to figure this out",
        "since I'm not sure how to proceed",
        "as I want to make sure I'm doing it right",
        "because I'm a bit confused",
        "since I keep getting stuck on it",
    ],
    "context": [
        "I'm on my phone",
        "I'm on my laptop",
        "I just tried it again",
        "this came up earlier today",
        "I'm in a rush right now",
        "I'm at home at the moment",
    ],
    "self_corrections": ["— wait, I mean", "— actually", "(actually)", "(I mean)"],
    "followups": ["can you help?", "what do I do next?", "is that right?", "can you explain?"],
    "soft_urgency": ["if you can", "when you have a chance", "soon if possible"],
}

NOISE_TEMPLATES_CLINC_TEST = {
    "openers": [
        "Hello", "Hi there", "Could you help", "I would like to ask",
        "Please clarify", "I need to know",
    ],
    "closers": ["thank you", "thanks in advance", "much appreciated"],
    "justifications": [
        "as I need clarification",
        "because I want to understand",
        "since I'm trying to finish this",
        "as this matters to me",
        "because I'm double-checking",
    ],
    "context": [
        "I tried this a couple times",
        "this happened just now",
        "I'm asking for future reference",
        "I'm checking to be sure",
        "I'm doing this step-by-step",
    ],
    "self_corrections": ["— to clarify", "— I should say", "(to clarify)", "(I should say)"],
    "followups": ["what are the next steps?", "can you provide details?", "how should I proceed?", "what should I know?"],
    "soft_urgency": ["when convenient", "at your earliest opportunity", "as soon as you can"],
}


def choose_templates(is_train: bool, noise_type: str) -> Dict[str, List[str]]:
    nt = (noise_type or "").lower()
    if nt == "clinc150":
        return NOISE_TEMPLATES_CLINC_TRAIN if is_train else NOISE_TEMPLATES_CLINC_TEST
    # default
    return NOISE_TEMPLATES_BANKING_TRAIN if is_train else NOISE_TEMPLATES_BANKING_TEST


# ----------------------------
# Mild typos (noise only)
# ----------------------------

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


def pick(pool: List[str]) -> str:
    return random.choice(pool)


def safe_append_punct(s: str, punct: str) -> str:
    s = s.strip()
    if not s:
        return s
    if s.endswith((".", "?", "!", ",")):
        return s
    return s + punct


def apply_typos_to_noise(text: str, typo_probability: float) -> str:
    out = []
    for ch in text:
        cl = ch.lower()
        if cl in TYPO_PATTERNS and random.random() < typo_probability:
            r = random.random()
            if r < 0.15:
                continue  # deletion
            if r < 0.30:
                out.append(ch)  # duplication
                out.append(ch)
            else:
                out.append(random.choice(TYPO_PATTERNS[cl]))
        else:
            out.append(ch)
    return "".join(out)


# ----------------------------
# Noise fragment builders (return strings that get concatenated)
# ----------------------------

def frag_prefix(T: Dict[str, List[str]]) -> str:
    opener = pick(T["openers"]).strip()
    if not opener.endswith((",", "—")):
        opener += ","
    return opener


def frag_suffix(T: Dict[str, List[str]]) -> str:
    tail = pick(T["followups"]) if random.random() < 0.65 else pick(T["closers"])
    return safe_append_punct(tail, ".")


def frag_context(T: Dict[str, List[str]]) -> str:
    ctx = pick(T["context"]).strip()
    return safe_append_punct(ctx, ".")


def frag_justification(T: Dict[str, List[str]]) -> str:
    j = pick(T["justifications"]).strip()
    return safe_append_punct(j, ".")


def frag_soft_urgency(T: Dict[str, List[str]]) -> str:
    u = pick(T["soft_urgency"]).strip()
    # keep it short; parenthetical is less label-shifting
    return f"({u})." if not u.endswith((".", "?", "!", ")")) else u


def frag_parenthetical(T: Dict[str, List[str]]) -> str:
    aside = pick(T["context"]) if random.random() < 0.6 else pick(T["justifications"])
    aside = normalize_spaces(aside)
    return f"({aside})."


def inject_self_correction(original: str, T: Dict[str, List[str]]) -> str:
    """
    Insert a correction marker WITHOUT duplicating original words.
    This keeps meaning stable across diverse CLINC intents.
    """
    words = original.split()
    if len(words) < 6:
        return original
    insert_at = random.randint(2, min(7, len(words) - 2))
    marker = pick(T["self_corrections"])
    new_words = words[:insert_at] + [marker] + words[insert_at:]
    return normalize_spaces(" ".join(new_words))


FRAG_BUILDERS = [frag_prefix, frag_suffix, frag_context, frag_justification, frag_soft_urgency, frag_parenthetical]


def add_realistic_noise(
    text: str,
    is_train: bool,
    noise_type: str = "banking77",
    max_extra_ratio: float = 0.45,
    typo_chance: float = 0.20,
    typo_probability: float = 0.02,
) -> str:
    """
    Adds noise to BOTH train and test; test noise differs via templates.
    Typos are applied ONLY to injected fragments.
    """
    if not text or not text.strip():
        return text

    T = choose_templates(is_train=is_train, noise_type=noise_type)
    original = normalize_spaces(text)
    orig_len = len(original.split())

    # Decide ops: 1–2 fragments + optional self-correction insertion
    num_frags = 1 if random.random() < 0.70 else 2
    builders = random.sample(FRAG_BUILDERS, k=num_frags)

    # Build fragments (noise-only strings)
    fragments = [normalize_spaces(b(T)) for b in builders if b(T).strip()]

    # Optionally apply mild typos to noise fragments only
    if fragments and random.random() < (typo_chance if is_train else (typo_chance * 0.5)):
        fragments = [apply_typos_to_noise(f, typo_probability=typo_probability) for f in fragments]

    # Assemble: pick a structure that doesn’t rewrite original content much
    noisy = original

    # Optional self-correction insertion (rare; avoid overdoing)
    if random.random() < (0.18 if is_train else 0.12):
        noisy = inject_self_correction(noisy, T)

    # Place fragments around the original
    # Heuristic: if we have a prefix-like fragment (endswith comma/—), put it first
    prefixish = [f for f in fragments if f.endswith(",") or f.endswith("—")]
    rest = [f for f in fragments if f not in prefixish]

    if prefixish:
        noisy = f"{prefixish[0]} {noisy}"
        rest = rest + prefixish[1:]

    # Put remaining fragments as suffixes (safer)
    for f in rest:
        noisy = f"{safe_append_punct(noisy, '.')} {f}"

    noisy = normalize_spaces(noisy)

    # Cap length growth; if too long, keep only one light fragment
    noisy_len = len(noisy.split())
    if orig_len > 0 and noisy_len > int(orig_len * (1 + max_extra_ratio)):
        light = frag_prefix(T) if random.random() < 0.5 else frag_suffix(T)
        if random.random() < (typo_chance if is_train else (typo_chance * 0.5)):
            light = apply_typos_to_noise(light, typo_probability=typo_probability)
        if light.endswith(",") or light.endswith("—"):
            noisy = normalize_spaces(f"{light} {original}")
        else:
            noisy = normalize_spaces(f"{safe_append_punct(original, '.')} {light}")

    return noisy


# ----------------------------
# Main runner
# ----------------------------

def run_add_noise(noise_type: str):
    """
    Adds noise to BOTH train and test; test uses different templates (distribution shift).
    """
    paths_config = CONFIG["paths"]
    train_file = paths_config["data"]["train"]
    test_file = paths_config["data"]["test"]

    logger.info(f"Using noise type: {noise_type}")

    # Train
    logger.info(f"Loading training data from {train_file}")
    with open(train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    for i, ex in enumerate(train_data, start=1):
        ex["text"] = add_realistic_noise(ex["text"], is_train=True, noise_type=noise_type)
        if i % 1000 == 0:
            logger.info(f"Processed {i} training examples...")

    logger.info(f"Saving modified training data back to {train_file}")
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Added noise to training set ({len(train_data)} examples)")

    # Test
    logger.info(f"Loading test data from {test_file}")
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    for i, ex in enumerate(test_data, start=1):
        ex["text"] = add_realistic_noise(ex["text"], is_train=False, noise_type=noise_type)
        if i % 1000 == 0:
            logger.info(f"Processed {i} test examples...")

    logger.info(f"Saving modified test data back to {test_file}")
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Added noise to test set ({len(test_data)} examples)")

    logger.info(f"\n{'=' * 80}")
    logger.info("✓ Noise augmentation complete")
    logger.info(f"✓ Noise type: {noise_type}")
    logger.info("✓ Train/test noise differ via separate template pools")
    logger.info("✓ Typos applied to injected noise fragments only")
