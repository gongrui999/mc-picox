"""Shared data contracts, bounded tokenization, and experiment provenance."""

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

DEFAULT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
PICO_CLASSES = ("PARTICIPANTS", "INTERVENTIONS", "OUTCOMES")
PICO_BITS = {"PARTICIPANTS": 4, "INTERVENTIONS": 2, "OUTCOMES": 1}


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def checkpoint_fingerprint(path):
    path = Path(path)
    names = sorted({p for pattern in ("config.json", "model*.safetensors", "model.pt",
                                     "model_config.pt", "pytorch_model*.bin",
                                     "*.safetensors.index.json", "pytorch_model*.index.json",
                                     "tokenizer*.json", "special_tokens_map.json",
                                     "added_tokens.json", "vocab.*", "merges.txt")
                    for p in path.glob(pattern) if p.is_file()})
    if not names:
        raise ValueError(f"No local checkpoint files in {path}")
    checksums = {p.name: file_sha256(p) for p in names}
    return hashlib.sha256(json.dumps(checksums, sort_keys=True).encode()).hexdigest()


def sentence_id(row, index):
    return str(row.get("sentence_id", f"{row.get('pmid', 'record')}:{index}"))


def gold_spans(row):
    """Canonical (type,start,length) set; explicit spans preserve same-type boundaries."""
    if "gold_spans" in row:
        return {(s["type"], int(s["start"]), int(s["end"]) - int(s["start"]))
                for s in row["gold_spans"]}
    labels = row.get("original_labels", row.get("labels"))
    if labels is None:
        raise ValueError("Missing gold_spans or token labels")
    spans = set()
    for label, bit in PICO_BITS.items():
        i = 0
        while i < len(labels):
            if labels[i] <= 0 or not labels[i] & bit:
                i += 1
                continue
            start = i
            while i < len(labels) and labels[i] > 0 and labels[i] & bit:
                i += 1
            spans.add((label, start, i - start))
    return spans


def boundary_labels(row):
    labels = [0] * len(row["tokens"])
    for _, start, length in gold_spans(row):
        end = start + length - 1
        if not 0 <= start <= end < len(labels):
            raise ValueError("Gold span outside token sequence")
        for i in range(start, end + 1):
            if labels[i] == 0:
                labels[i] = 4
        labels[start] = (labels[start] if labels[start] != 4 else 0) | 1
        labels[end] = (labels[end] if labels[end] != 4 else 0) | 2
    return labels


def max_sequence_length(tokenizer, config=None, requested=None):
    encoder = getattr(config, "encoder_config", None)
    model_limit = encoder.get("max_position_embeddings") if encoder else getattr(
        config, "max_position_embeddings", None
    )
    limits = [int(n) for n in (model_limit, getattr(tokenizer, "model_max_length", None))
              if n is not None and 4 <= n < 1_000_000]
    limit = min(limits) if limits else 512
    if requested is not None:
        if not 4 <= requested <= limit:
            raise ValueError(f"max_length must be between 4 and model limit {limit}")
        limit = requested
    return limit


def encode_windows(tokenizer, tokens, max_length, stride=64, tensors=None):
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer with word_ids is required")
    content_length = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if content_length <= 0 or stride < 0:
        raise ValueError("Invalid max_length or stride")
    overlap = min(stride, max(0, content_length // 2))
    return tokenizer(
        tokens, is_split_into_words=True, truncation=True, max_length=max_length,
        stride=overlap, return_overflowing_tokens=True,
        padding="max_length" if tensors else False, return_tensors=tensors,
    )


def experiment_manifest(arguments, files=(), **extra):
    try:
        root = Path(__file__).resolve().parent
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        diff = subprocess.check_output(
            ["git", "diff", "HEAD", "--", "*.py", "*.txt", "*.md"], cwd=root,
            stderr=subprocess.DEVNULL,
        )
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, diff, dirty = None, b"", None
    versions = {}
    for name in ("torch", "transformers", "datasets", "accelerate", "numpy",
                 "scikit-learn", "openpyxl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "schema_version": 1, "git_commit": commit, "working_tree_dirty": dirty,
        "working_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "arguments": vars(arguments) if hasattr(arguments, "__dict__") else arguments,
        "input_sha256": {str(p): file_sha256(p) for p in files},
        "versions": versions, **extra,
    }
