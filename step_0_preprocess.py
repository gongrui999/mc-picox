"""
Step 0: Preprocess EBM-NLP BioC XML data into JSON Lines format.

Reads the EBM-NLP BioC XML file, splits into train/validation/test sets,
and generates:
  1. Sequence labeling JSONL (boundary detection training data)
  2. Span classification JSONL (span classifier training data)

Usage:
    python step_0_preprocess.py \
        --input_file data/bioc/input/ebm_nlp_2_00_ssplit.xml \
        --raw_dir data/raw/ebm_nlp_2_00 \
        --output_path data/bioc/json
"""

import argparse
import glob
import json
import os
import random
import warnings
from collections import defaultdict
from enum import Enum
from typing import Any, List

from bioc import biocxml
from pipeline_utils import experiment_manifest, write_json


def split_pmids(all_pmid, test_pmid, val_ratio=0.05, seed=0):
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be strictly between 0 and 1")
    test = set(test_pmid) & set(all_pmid)
    available = sorted(set(all_pmid) - test)
    if len(available) < 2:
        raise ValueError("At least two non-test documents are needed")
    random.Random(seed).shuffle(available)
    nval = min(len(available) - 1, max(1, int(round(val_ratio * len(available)))))
    return set(available[nval:]), set(available[:nval]), test


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DatasetSplit(Enum):
    train = 0
    validation = 1
    test = 2


class PicoType(Enum):
    PARTICIPANTS = 4
    INTERVENTIONS = 2
    OUTCOMES = 1


# ---------------------------------------------------------------------------
# Span helper
# ---------------------------------------------------------------------------

class Span:
    def __init__(self, start, length):
        self.start = start
        self.length = length
        self.end = start + length

    def __repr__(self):
        return f"Span(start:{self.start}, length:{self.length})"

    def __eq__(self, other):
        return (self.start, int(self.length)) == (other.start, int(other.length))

    def __hash__(self):
        return hash((self.start, int(self.length)))


# ---------------------------------------------------------------------------
# PMID loading
# ---------------------------------------------------------------------------

def load_pmid(folder: str) -> set:
    return set(
        f.rstrip(".AGGREGATED.ann")
        for f in os.listdir(folder)
        if f.endswith(".AGGREGATED.ann")
    )


# ---------------------------------------------------------------------------
# Label / token extraction from BioC
# ---------------------------------------------------------------------------

def extract_pio_labels(sentence: Any, pico_type: PicoType) -> List[Any]:
    """Extract PIO BIO labels from a BioC sentence."""
    label_name = f"starting_spans/{pico_type.name.lower()}"
    if not sentence.annotations:
        return None
    if label_name not in sentence.annotations[0].infons:
        return [None for _ in sentence.annotations]

    raw_labels = [a.infons[label_name] for a in sentence.annotations]
    b_tag = f"B-{pico_type.name}"
    i_tag = f"I-{pico_type.name}"

    labels = []
    for i, raw_label in enumerate(raw_labels):
        if raw_label == "0":
            labels.append("O")
        elif i == 0:
            labels.append(b_tag)
        else:
            labels.append(i_tag if raw_labels[i - 1] == "1" else b_tag)
    return labels


def extract_tokens(sentence: Any) -> List[str]:
    return [a.text for a in sentence.annotations]


def check_overlapping(sentence: Any) -> bool:
    """Check whether PIO labels overlap on any token."""
    def _raw(sent, pt):
        ln = f"starting_spans/{pt.name.lower()}"
        if not sent.annotations or ln not in sent.annotations[0].infons:
            return [0 for _ in sent.annotations]
        return [int(a.infons[ln]) for a in sent.annotations]

    p = _raw(sentence, PicoType.PARTICIPANTS)
    i = _raw(sentence, PicoType.INTERVENTIONS)
    o = _raw(sentence, PicoType.OUTCOMES)
    return max(a + b + c for a, b, c in zip(p, i, o)) > 1


def extract_pico_spans(sentence: Any, pico_type: PicoType) -> List[Span]:
    label_name = f"starting_spans/{pico_type.name.lower()}"
    if not sentence.annotations:
        return []
    if label_name not in sentence.annotations[0].infons:
        return []

    labels = [int(a.infons[label_name]) for a in sentence.annotations]
    span_start = [0] * len(labels)
    span_length = [0.0] * len(labels)

    start = 0
    for i, label in enumerate(labels):
        if label > 0:
            if i == 0 or labels[i - 1] <= 0:
                span_start[i] = 1
                start = i
            span_length[start] += 1

    return [Span(start=i, length=span_length[i]) for i in range(len(span_start)) if span_start[i]]


# ---------------------------------------------------------------------------
# Synthesize negative spans (cross-boundary)
# ---------------------------------------------------------------------------

def synthesize_span_util(span_a: Span, span_b: Span) -> List[Span]:
    if span_a.start > span_b.start:
        span_a, span_b = span_b, span_a
    spans = []
    if span_b.start >= span_a.end:
        spans.append(Span(span_a.start, span_b.end - span_a.start))
    else:
        if span_a.start != span_b.start and span_a.start != span_b.end:
            spans.append(Span(span_b.start, span_a.end - span_b.start))
            spans.append(Span(span_a.start, span_b.end - span_a.start))
    return spans


def synthesize_spans(list_a: List[Span], list_b: List[Span]) -> List[Span]:
    spans = []
    for a in list_a:
        for b in list_b:
            spans += synthesize_span_util(a, b)
    return spans


# ---------------------------------------------------------------------------
# Dataset creation – sequence labeling
# ---------------------------------------------------------------------------

def create_dataset(
    collection,
    dataset_split: DatasetSplit,
    output_path: str,
    train_pmid: set,
    validation_pmid: set,
    test_pmid: set,
    include_overlapping_spans: bool = True,
    only_overlapping_spans: bool = False,
):
    target_map = {
        DatasetSplit.train: train_pmid,
        DatasetSplit.validation: validation_pmid,
        DatasetSplit.test: test_pmid,
    }
    target_pmid_set = target_map[dataset_split]

    output_file = os.path.join(output_path, f"{dataset_split.name}.json")
    os.makedirs(output_path, exist_ok=True)

    overlap_count, total_count = 0, 0

    with open(output_file, "w+", encoding="utf-8") as fout:
        for document in collection.documents:
            pmid = document.id
            if pmid not in target_pmid_set:
                continue
            for passage_index, passage in enumerate(document.passages):
                for sentence_index, sentence in enumerate(passage.sentences):
                    tokens = extract_tokens(sentence)
                    participants = extract_pio_labels(sentence, PicoType.PARTICIPANTS)
                    interventions = extract_pio_labels(sentence, PicoType.INTERVENTIONS)
                    outcomes = extract_pio_labels(sentence, PicoType.OUTCOMES)

                    if not participants or not interventions or not outcomes:
                        warnings.warn(f"Empty annotations in abstract {pmid}.")
                        continue

                    total_count += 1
                    if check_overlapping(sentence):
                        overlap_count += 1
                        if not include_overlapping_spans:
                            continue
                    elif only_overlapping_spans:
                        continue

                    labels = []
                    for p, i, o in zip(participants, interventions, outcomes):
                        label = 0
                        if p not in ("O", None):
                            label |= PicoType.PARTICIPANTS.value
                        if i not in ("O", None):
                            label |= PicoType.INTERVENTIONS.value
                        if o not in ("O", None):
                            label |= PicoType.OUTCOMES.value
                        labels.append(label)

                    data = {"pmid": pmid,
                            "sentence_id": f"{pmid}:{passage_index}:{sentence_index}",
                            "tokens": tokens, "labels": labels}
                    fout.write(f"{json.dumps(data)}\n")

    print(f"{dataset_split.name}: {overlap_count}/{total_count} sentences have overlapping spans.")


# ---------------------------------------------------------------------------
# Dataset creation – span classification
# ---------------------------------------------------------------------------

def span_classification_examples(tokens, pico_spans, pmid, sample_limit=1):
    """Merge identical gold boundaries into one truly multilabel training example."""
    positives = defaultdict(set)
    for pico_type, spans in pico_spans.items():
        for span in spans:
            positives[(int(span.start), int(span.end))].add(pico_type.name)
    negatives = set()
    kinds = list(PicoType)
    for i, a in enumerate(kinds):
        for b in kinds[i + 1:]:
            for span in synthesize_spans(pico_spans[a], pico_spans[b]):
                bounds = (int(span.start), int(span.end))
                if bounds not in positives:
                    negatives.add(bounds)
    candidates = sorted(negatives)
    random.shuffle(candidates)
    selected = {**positives, **{bounds: set() for bounds in candidates[:sample_limit]}}
    examples = []
    for (start, end), labels in sorted(selected.items()):
        if not 0 <= start < end <= len(tokens):
            raise ValueError("Span classification example outside sentence")
        examples.append({
            "pmid": pmid, "tokens": tokens[start:end],
            **{kind.name: kind.name in labels for kind in PicoType},
        })
    return examples


def create_span_clf_dataset(
    collection, dataset_split, output_path, train_pmid, validation_pmid, test_pmid,
    sample_limit=1,
):
    target_map = {DatasetSplit.train: train_pmid, DatasetSplit.validation: validation_pmid,
                  DatasetSplit.test: test_pmid}
    target_pmid_set = target_map[dataset_split]
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{dataset_split.name}_span_clf.json")
    count = defaultdict(int)
    with open(output_file, "w", encoding="utf-8") as stream:
        for document in collection.documents:
            if document.id not in target_pmid_set:
                continue
            for passage in document.passages:
                for sentence in passage.sentences:
                    pico_spans = {kind: extract_pico_spans(sentence, kind) for kind in PicoType}
                    for example in span_classification_examples(
                        extract_tokens(sentence), pico_spans, document.id, sample_limit
                    ):
                        stream.write(json.dumps(example) + "\n")
                        active = [kind.name for kind in PicoType if example[kind.name]]
                        for label in active or ["SYNTHESIZED"]:
                            count[label] += 1
    print(f"{dataset_split.name}: {dict(count)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 0: Preprocess EBM-NLP BioC XML")
    parser.add_argument("--input_file", type=str, default="data/bioc/input/ebm_nlp_2_00_ssplit.xml",
                        help="Path to BioC XML file")
    parser.add_argument("--raw_dir", type=str, default="data/raw/ebm_nlp_2_00",
                        help="Path to raw EBM-NLP directory (for loading PMIDs)")
    parser.add_argument("--output_path", type=str, default="data/bioc/json",
                        help="Output directory for JSONL files")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Fraction of training data for validation")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--split_manifest", default=None,
                        help="Reuse an existing manifest instead of creating a new partition")
    args = parser.parse_args()

    random.seed(args.seed)

    # ---- Load PMIDs from raw EBM-NLP annotations ----
    ann_base = os.path.join(args.raw_dir, "annotations", "aggregated", "starting_spans")
    participants_train = load_pmid(os.path.join(ann_base, "participants", "train"))
    interventions_train = load_pmid(os.path.join(ann_base, "interventions", "train"))
    outcomes_train = load_pmid(os.path.join(ann_base, "outcomes", "train"))

    participants_test = load_pmid(os.path.join(ann_base, "participants", "test", "gold"))
    interventions_test = load_pmid(os.path.join(ann_base, "interventions", "test", "gold"))
    outcomes_test = load_pmid(os.path.join(ann_base, "outcomes", "test", "gold"))

    print(f"PARTICIPANTS: train {len(participants_train)}, test {len(participants_test)}")
    print(f"INTERVENTIONS: train {len(interventions_train)}, test {len(interventions_test)}")
    print(f"OUTCOMES: train {len(outcomes_train)}, test {len(outcomes_test)}")

    # ---- Load BioC collection ----
    print(f"Loading BioC XML from {args.input_file} ...")
    with open(args.input_file, "r", encoding="UTF-8") as fp:
        collection = biocxml.load(fp)

    # ---- Split ----
    all_pmid = set(doc.id for doc in collection.documents)
    official_test = participants_test & interventions_test & outcomes_test & all_pmid
    if args.split_manifest:
        with open(args.split_manifest, encoding="utf-8") as stream:
            partition = json.load(stream)["splits"]
        train_pmid, validation_pmid, test_pmid = (
            set(partition[s]) for s in ("train", "validation", "test")
        )
        if (train_pmid & validation_pmid or train_pmid & test_pmid
                or validation_pmid & test_pmid
                or train_pmid | validation_pmid | test_pmid != all_pmid
                or test_pmid != official_test):
            raise ValueError("Manifest must partition this corpus and preserve official test PMIDs")
    else:
        train_pmid, validation_pmid, test_pmid = split_pmids(
            all_pmid, official_test, args.val_ratio, args.seed
        )
    write_json(os.path.join(args.output_path, "split_manifest.json"), experiment_manifest(
        args, [args.input_file],
        splits={"train": sorted(train_pmid), "validation": sorted(validation_pmid),
                "test": sorted(test_pmid)},
    ))

    print(f"Total: {len(collection.documents)}, Train: {len(train_pmid)}, "
          f"Val: {len(validation_pmid)}, Test: {len(test_pmid)}")

    # ---- Sequence labeling datasets ----
    print("\n=== Creating sequence labeling datasets ===")
    for split in DatasetSplit:
        create_dataset(collection, split, args.output_path,
                       train_pmid, validation_pmid, test_pmid)

    # No-overlap variant
    no_overlap_path = os.path.join(args.output_path, "no_overlap")
    create_dataset(collection, DatasetSplit.train, no_overlap_path,
                   train_pmid, validation_pmid, test_pmid, include_overlapping_spans=False)
    create_dataset(collection, DatasetSplit.test, no_overlap_path,
                   train_pmid, validation_pmid, test_pmid, include_overlapping_spans=False)

    # Overlap-only variant
    overlap_only_path = os.path.join(args.output_path, "overlap_only")
    create_dataset(collection, DatasetSplit.test, overlap_only_path,
                   train_pmid, validation_pmid, test_pmid, only_overlapping_spans=True)

    # ---- Span classification datasets ----
    print("\n=== Creating span classification datasets ===")
    for split in DatasetSplit:
        create_span_clf_dataset(collection, split, args.output_path,
                                train_pmid, validation_pmid, test_pmid)

    print("\nDone.")


if __name__ == "__main__":
    main()
