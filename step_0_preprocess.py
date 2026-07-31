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
            for passage in document.passages:
                for sentence in passage.sentences:
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

                    data = {"pmid": pmid, "tokens": tokens, "labels": labels}
                    fout.write(f"{json.dumps(data)}\n")

    print(f"{dataset_split.name}: {overlap_count}/{total_count} sentences have overlapping spans.")


# ---------------------------------------------------------------------------
# Dataset creation – span classification
# ---------------------------------------------------------------------------

def create_span_clf_dataset(
    collection,
    dataset_split: DatasetSplit,
    output_path: str,
    train_pmid: set,
    validation_pmid: set,
    test_pmid: set,
    sample_limit: int = 1,
):
    target_map = {
        DatasetSplit.train: train_pmid,
        DatasetSplit.validation: validation_pmid,
        DatasetSplit.test: test_pmid,
    }
    target_pmid_set = target_map[dataset_split]

    output_file = os.path.join(output_path, f"{dataset_split.name}_span_clf.json")
    os.makedirs(output_path, exist_ok=True)

    count = defaultdict(int)

    with open(output_file, "w+", encoding="utf-8") as fout:
        for document in collection.documents:
            pmid = document.id
            if pmid not in target_pmid_set:
                continue
            for passage in document.passages:
                for sentence in passage.sentences:
                    tokens = extract_tokens(sentence)
                    pico_spans = {}
                    for pico_type in list(PicoType):
                        spans = extract_pico_spans(sentence, pico_type)
                        pico_spans[pico_type] = spans
                        for span in spans:
                            start, end = int(span.start), int(span.start + span.length)
                            data = {
                                "pmid": pmid,
                                "tokens": tokens[start:end],
                                "PARTICIPANTS": False,
                                "INTERVENTIONS": False,
                                "OUTCOMES": False,
                            }
                            data[pico_type.name] = True
                            fout.write(f"{json.dumps(data)}\n")
                            count[pico_type.name] += 1

                    synthesized = (
                        synthesize_spans(pico_spans[PicoType.PARTICIPANTS], pico_spans[PicoType.INTERVENTIONS])
                        + synthesize_spans(pico_spans[PicoType.PARTICIPANTS], pico_spans[PicoType.OUTCOMES])
                        + synthesize_spans(pico_spans[PicoType.INTERVENTIONS], pico_spans[PicoType.OUTCOMES])
                    )
                    synthesized = list(set(synthesized))
                    random.shuffle(synthesized)
                    for span in synthesized[:sample_limit]:
                        if all(
                            span not in pico_spans[pt]
                            for pt in PicoType
                        ):
                            start, end = int(span.start), int(span.start + span.length)
                            data = {
                                "pmid": pmid,
                                "tokens": tokens[start:end],
                                "PARTICIPANTS": False,
                                "INTERVENTIONS": False,
                                "OUTCOMES": False,
                            }
                            fout.write(f"{json.dumps(data)}\n")
                            count["SYNTHESIZED"] += 1

    stats = ", ".join(f"{k}: {count[k]}" for k in count)
    print(f"{dataset_split.name}: {stats}")


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
    test_pmid = participants_test & interventions_test & outcomes_test
    train_pmid_list = list(all_pmid - test_pmid)
    boundary = int((1.0 - args.val_ratio) * len(train_pmid_list))
    train_pmid = set(train_pmid_list[:boundary])
    validation_pmid = set(train_pmid_list[boundary:])

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
