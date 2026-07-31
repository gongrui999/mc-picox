"""
Step 3: Evaluate PICO span extraction results.

Compares predicted PICO spans against gold-standard labels and reports
Precision / Recall / F1 for each PICO type and overall.

Usage:
    python step_3_evaluate.py \
        --input_folder data/bioc/json/step_2_span_clf \
        --pred_file test_pico_spans.json \
        --confidence_threshold 0.5

    # Optionally compare against a baseline:
    python step_3_evaluate.py \
        --input_folder data/bioc/json/step_2_span_clf \
        --pred_file test_pico_spans.json \
        --baseline_file baseline/test_baseline_pred.json
"""

import argparse
import os
from enum import Enum

from datasets import load_dataset
from tqdm.auto import tqdm


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
# Span & metrics helpers
# ---------------------------------------------------------------------------

class Span:
    def __init__(self, start, length):
        self.start = start
        self.length = length
        self.end = self.start + self.length

    def __repr__(self):
        return f"Span(start={self.start}, length={self.length})"

    def __eq__(self, other):
        return (self.start, int(self.length)) == (other.start, int(other.length))


class ConfusionMatrix:
    def __init__(self, tp=0.0, fp=0.0, fn=0.0):
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def __add__(self, other):
        return ConfusionMatrix(
            self.tp + other.tp, self.fp + other.fp, self.fn + other.fn
        )

    def __iadd__(self, other):
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    def compute(self):
        p = self.tp / (self.tp + self.fp) if self.tp else 0
        r = self.tp / (self.tp + self.fn) if self.tp else 0
        f = 2 * p * r / (p + r) if self.tp else 0
        return p, r, f

    def __repr__(self):
        p, r, f = self.compute()
        return (
            f"  TP: {self.tp}  FP: {self.fp}  FN: {self.fn}\n"
            f"  Precision: {p * 100:.2f}%  Recall: {r * 100:.2f}%  F1: {f * 100:.2f}%\n"
        )


def extract_spans_from_labels(label_sequence, pico_type_value):
    labels = [pico_type_value & l if l > 0 else 0 for l in label_sequence]
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


def eval_pred_single_sample(prediction, reference):
    tp, fp, fn = 0.0, 0.0, 0.0
    prediction.sort(key=lambda x: x.start)
    reference.sort(key=lambda x: x.start)
    pi, ri = 0, 0

    while pi < len(prediction) and ri < len(reference):
        sp, sr = prediction[pi], reference[ri]
        if sp == sr:
            pi += 1
            ri += 1
            tp += 1
        elif sp.start < sr.start:
            pi += 1
            fp += 1
        else:
            ri += 1
            fn += 1

    fp += len(prediction) - pi
    fn += len(reference) - ri
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Evaluation routines
# ---------------------------------------------------------------------------

def evaluate_predictions(dataset, confidence_threshold=0.5):
    metrics = {pt: ConfusionMatrix() for pt in PicoType}
    all_metric = ConfusionMatrix()

    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        original_labels = dataset["original_labels"][i]
        result_dict = dataset["pico_elements"][i]

        for pico_type in PicoType:
            records = result_dict.get(pico_type.name)
            prediction = []
            if records:
                prediction = [
                    Span(start=r["span_start"], length=r["span_length"])
                    for r in records
                    if r["confidence"] > confidence_threshold
                ]
            reference = extract_spans_from_labels(original_labels, pico_type.value)
            tp, fp, fn = eval_pred_single_sample(prediction, reference)
            batch = ConfusionMatrix(tp, fp, fn)
            metrics[pico_type] += batch
            all_metric += batch

    return all_metric, metrics


def evaluate_baseline(pred_dataset, val_dataset):
    metrics = {pt: ConfusionMatrix() for pt in PicoType}
    all_metric = ConfusionMatrix()

    for i in tqdm(range(len(val_dataset)), desc="Evaluating baseline"):
        reference_labels = pred_dataset["original_labels"][i]
        prediction_labels = pred_dataset["pico_pred"][i]

        for pico_type in PicoType:
            pred = extract_spans_from_labels(prediction_labels, pico_type.value)
            ref = extract_spans_from_labels(reference_labels, pico_type.value)
            tp, fp, fn = eval_pred_single_sample(pred, ref)
            batch = ConfusionMatrix(tp, fp, fn)
            metrics[pico_type] += batch
            all_metric += batch

    return all_metric, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 3: Evaluate PICO span extraction")
    parser.add_argument("--input_folder", type=str, default="data/bioc/json/step_2_span_clf")
    parser.add_argument("--pred_file", type=str, default="test_pico_spans.json",
                        help="Prediction JSONL filename inside input_folder")
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--baseline_file", type=str, default=None,
                        help="Optional baseline prediction file for comparison")
    args = parser.parse_args()

    pred_path = os.path.join(args.input_folder, args.pred_file)

    # ---- Load predictions ----
    ebm_nlp = load_dataset(
        "json",
        data_files={"test": pred_path},
    )
    val = ebm_nlp["test"]

    print(f"Loaded {len(val)} samples from {pred_path}\n")

    # ---- PICOX evaluation ----
    print("=" * 60)
    print("PICOX Results")
    print("=" * 60)
    all_metric, per_type = evaluate_predictions(val, args.confidence_threshold)

    for pico_type in PicoType:
        print(f"\n{pico_type.name}:")
        print(per_type[pico_type])

    print(f"\nOVERALL:")
    print(all_metric)

    # ---- Baseline evaluation (optional) ----
    if args.baseline_file:
        print("\n" + "=" * 60)
        print("Baseline Results")
        print("=" * 60)

        baseline = load_dataset("json", data_files={"test": args.baseline_file})
        baseline_val = baseline["test"]

        bl_all, bl_per_type = evaluate_baseline(baseline_val, val)

        for pico_type in PicoType:
            print(f"\n{pico_type.name}:")
            print(bl_per_type[pico_type])

        print(f"\nOVERALL:")
        print(bl_all)


if __name__ == "__main__":
    main()
