"""Build the rapid-pilot datasets from Stack Exchange dumps.

Source: official 2024-04-01 Stack Exchange dump on archive.org (CC BY-SA; predates the 2024 LLM
click-through). Each record id is "<site>:<post id>", i.e. https://<site>.stackexchange.com/q/<post id>,
which keeps per-question attribution recoverable.

Equivalence groups are connected components of moderator-marked duplicate links (LinkTypeId=3),
2..20 members. Groups are split fit/val/test 40/20/40 (GROUP_SPLIT) and never shared across
splits. Classifier training data are "singles": questions with no duplicate link at all, so the
classifier never sees a duplicate-linked question.

Usage:
    python -I rapid_pilot/build_data.py --dumps <dir with <site>/x/Posts.xml> --out rapid_pilot/data
"""
import argparse
import html
import json
import os
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

WATERMARK_MIN_ID = 10**9
MAX_GROUP_SIZE = 20
GROUP_SPLIT = {"fit": 0.4, "val": 0.2, "test": 0.4}
MAX_CHARS = 4000

SITE_TASK_SITES = ["electronics", "unix", "dba", "gis", "diy", "mechanics", "networkengineering"]

# Coarse product/tool families; first matching family wins. Review before use.
FAMILIES = {
    "gis": [
        ("qgis", r"^(qgis.*|pyqgis.*)$"),
        ("esri", r"^(arcgis.*|arcpy|arcmap|arcobjects|model-builder|esri.*)$"),
        ("postgis", r"^postgis.*$"),
        ("gee", r"^google-earth-engine.*$"),
        ("gdal", r"^(gdal|ogr.*|ogr2ogr)$"),
        ("web", r"^(leaflet.*|openlayers.*|geoserver|mapbox.*|cesium)$"),
        ("r", r"^(r|sf|raster-r|terra)$"),
    ],
    "dba": [
        ("sql-server", r"^(sql-server.*|t-sql|ssis|ssrs|ssms)$"),
        ("mysql", r"^(mysql.*|innodb|mariadb.*|phpmyadmin)$"),
        ("postgresql", r"^(postgresql.*|plpgsql)$"),
        ("oracle", r"^(oracle.*|plsql)$"),
        ("mongodb", r"^mongodb.*$"),
        ("sqlite", r"^sqlite.*$"),
        ("db2", r"^db2.*$"),
    ],
}

# Singles per label: classifier train / val / test.
SINGLES_CAP = {"train": 3000, "val": 200, "test": 500}
MIN_SINGLES_PER_LABEL = 1000

_DUP_BLOCKQUOTE_RE = re.compile(r"<blockquote>.*?</blockquote>", re.S | re.I)
_QUESTION_LINK_RE = re.compile(r"<a [^>]*href=\"[^\"]*/(?:questions|q)/\d+[^\"]*\"[^>]*>.*?</a>", re.S | re.I)
_QUESTION_URL_RE = re.compile(r"https?://\S*/(?:questions|q)/\d+\S*", re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_DUP_PHRASE_RE = re.compile(r"possible duplicate:?|\[duplicate\]", re.I)
_WS_RE = re.compile(r"\s+")

# Credentials users pasted into posts (GitHub push protection rejects them). Replaced, not dropped.
_SECRET_RES = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)", re.S),
    re.compile(r"\b[pst]k\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),  # Mapbox
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}"),  # Google API key
    re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}"),  # Slack
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"),  # GitHub
    re.compile(r"\b(?:sk|rk)_live_[0-9A-Za-z]{20,}"),  # Stripe
    re.compile(r"AccountKey=[A-Za-z0-9+/=]{40,}"),  # Azure storage
]


def redact_secrets(text: str) -> str:
    for pattern in _SECRET_RES:
        text = pattern.sub("[REDACTED_TOKEN]", text)
    return text


def clean_text(text: str) -> str:
    """Strip HTML and every pointer to another question.

    Old closed duplicates carry a "Possible Duplicate: <title of the target>" blockquote in the body,
    which would leak the group identity into the input.
    """
    text = _DUP_BLOCKQUOTE_RE.sub(lambda m: " " if re.search(r"duplicate", m.group(0), re.I) else m.group(0), text)
    text = _QUESTION_LINK_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _QUESTION_URL_RE.sub(" ", text)
    text = _DUP_PHRASE_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def parse_tags(raw: str):
    if not raw:
        return []
    if raw.startswith("|"):
        return [t for t in raw.split("|") if t]
    return re.findall(r"<([^>]+)>", raw)


def parse_site(site_dir: str):
    questions = {}
    for _, el in ET.iterparse(os.path.join(site_dir, "Posts.xml")):
        if el.tag == "row" and el.get("PostTypeId") == "1":
            pid = int(el.get("Id"))
            if pid < WATERMARK_MIN_ID:
                title = clean_text(el.get("Title") or "")
                body = clean_text(el.get("Body") or "")
                questions[pid] = {
                    "text": redact_secrets(f"{title}\n{body}"[:MAX_CHARS]),
                    "tags": parse_tags(el.get("Tags") or ""),
                }
        el.clear()

    edges = set()
    for _, el in ET.iterparse(os.path.join(site_dir, "PostLinks.xml")):
        if el.tag == "row" and el.get("LinkTypeId") == "3":
            a, b = int(el.get("PostId")), int(el.get("RelatedPostId"))
            if a in questions and b in questions and a != b:
                edges.add((min(a, b), max(a, b)))
        el.clear()
    return questions, edges


def connected_groups(edges):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    comps = defaultdict(list)
    for x in list(parent):
        comps[find(x)].append(x)
    return [sorted(g) for g in comps.values()]


def family_of(tags, site):
    for name, pattern in FAMILIES[site]:
        if any(re.match(pattern, t) for t in tags):
            return name
    return None


def split_groups(groups, rng):
    groups = sorted(groups, key=lambda g: g[0][0])
    rng.shuffle(groups)
    n_fit = int(len(groups) * GROUP_SPLIT["fit"])
    n_val = int(len(groups) * GROUP_SPLIT["val"])
    return {
        "fit_groups": groups[:n_fit],
        "val_groups": groups[n_fit:n_fit + n_val],
        "test_groups": groups[n_fit + n_val:],
    }


def build_task(name, site_data, labeler, rng):
    """labeler(site, pid) -> label name or None."""
    groups, singles_by_label, dropped_groups = [], defaultdict(list), Counter()
    for site, (questions, edges) in site_data.items():
        linked = {p for e in edges for p in e}
        for members in connected_groups(edges):
            if len(members) > MAX_GROUP_SIZE:
                dropped_groups["too_large"] += 1
                continue
            labels = {labeler(site, p) for p in members}
            if None in labels or len(labels) != 1:
                dropped_groups["unlabelled_or_mixed_label"] += 1
                continue
            label = labels.pop()
            groups.append([(site, p, label) for p in members])
        for pid in sorted(questions):
            if pid not in linked:
                label = labeler(site, pid)
                if label is not None:
                    singles_by_label[label].append((site, pid, label))

    label_names = sorted(l for l, s in singles_by_label.items() if len(s) >= MIN_SINGLES_PER_LABEL)
    kept = set(label_names)
    n_groups_before = len(groups)
    groups = [g for g in groups if g[0][2] in kept]
    dropped_groups["rare_label"] += n_groups_before - len(groups)

    splits = split_groups(groups, rng)
    singles = {"train": [], "val_singles": [], "test_singles": []}
    for label in label_names:
        pool = singles_by_label[label][:]
        rng.shuffle(pool)
        n_tr, n_va, n_te = SINGLES_CAP["train"], SINGLES_CAP["val"], SINGLES_CAP["test"]
        n_te = min(n_te, len(pool) // 4)
        n_va = min(n_va, len(pool) // 8)
        singles["test_singles"] += pool[:n_te]
        singles["val_singles"] += pool[n_te:n_te + n_va]
        singles["train"] += pool[n_te + n_va:n_te + n_va + n_tr]

    label_index = {l: i for i, l in enumerate(label_names)}

    def record(site, pid, label, group=None):
        rec = {"id": f"{site}:{pid}", "text": site_data[site][0][pid]["text"], "label": label_index[label]}
        if group is not None:
            rec["group"] = group
        return rec

    files = {}
    for split_name, items in singles.items():
        rng.shuffle(items)
        files[split_name] = [record(*it) for it in items]
    for split_name, split in splits.items():
        files[split_name] = [
            record(site, pid, label, group=f"{members[0][0]}:{members[0][1]}")
            for members in split
            for site, pid, label in members
        ]

    meta = {
        "task": name,
        "source": "Stack Exchange data dump, archive.org snapshot 2024-04-01 (CC BY-SA; see ContentLicense per post)",
        "label_names": label_names,
        "counts": {k: len(v) for k, v in files.items()},
        "groups": {k: len(v) for k, v in splits.items()},
        "dropped_groups": dict(dropped_groups),
        "max_group_size": MAX_GROUP_SIZE,
        "singles_cap": SINGLES_CAP,
    }
    return files, meta


def write_task(out_dir, files, meta):
    os.makedirs(out_dir, exist_ok=True)
    for split_name, records in files.items():
        with open(os.path.join(out_dir, f"{split_name}.jsonl"), "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dumps", required=True, help="Directory containing <site>/x/{Posts,PostLinks}.xml")
    parser.add_argument("--out", default="rapid_pilot/data")
    parser.add_argument("--tasks", nargs="+", default=["site", "gis", "dba"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    needed = set()
    if "site" in args.tasks:
        needed |= set(SITE_TASK_SITES)
    needed |= {t for t in args.tasks if t in FAMILIES}

    site_data = {}
    for site in sorted(needed):
        print(f"Parsing {site} ...", flush=True)
        site_data[site] = parse_site(os.path.join(args.dumps, site, "x"))

    for task in args.tasks:
        rng = random.Random(args.seed)
        if task == "site":
            data = {s: site_data[s] for s in SITE_TASK_SITES}
            files, meta = build_task(task, data, lambda site, pid: site, rng)
        else:
            questions = site_data[task][0]
            files, meta = build_task(
                task, {task: site_data[task]}, lambda site, pid: family_of(questions[pid]["tags"], site), rng
            )
        write_task(os.path.join(args.out, task), files, meta)


if __name__ == "__main__":
    main()
