"""Strict, duplicate-safe span evaluation, including explicit character-level gold spans."""

import argparse
import json
from pathlib import Path

from pipeline_utils import (
    PICO_CLASSES, experiment_manifest, gold_spans, read_jsonl, write_json,
)


def predicted_spans(row, confidence_threshold=0.5):
    return {
        (label, int(span["span_start"]), int(span["span_length"]))
        for label, spans in row.get("pico_elements", {}).items()
        for span in spans if span["confidence"] >= confidence_threshold
    }


def confusion(prediction, reference):
    return {"tp": len(prediction & reference), "fp": len(prediction - reference),
            "fn": len(reference - prediction)}


def scores(counts):
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    p, r = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
    return {**counts, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0}


def evaluate_predictions(rows, confidence_threshold=0.5):
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("Classification threshold must be in [0,1]")
    total = {label: {"tp": 0, "fp": 0, "fn": 0} for label in PICO_CLASSES}
    count = 0
    for row in rows:
        count += 1
        truth, prediction = gold_spans(row), predicted_spans(row, confidence_threshold)
        if any(t not in PICO_CLASSES for t, _, _ in truth | prediction):
            raise ValueError("Unknown PICO category")
        for label in PICO_CLASSES:
            result = confusion({s for s in prediction if s[0] == label},
                               {s for s in truth if s[0] == label})
            for key, value in result.items():
                total[label][key] += value
    if not count:
        raise ValueError("Cannot evaluate an empty prediction file")
    per_type = {label: scores(value) for label, value in total.items()}
    micro = scores({key: sum(v[key] for v in total.values()) for key in ("tp", "fp", "fn")})
    macro = {key: sum(v[key] for v in per_type.values()) / len(PICO_CLASSES)
             for key in ("precision", "recall", "f1")}
    return {"records": count, "micro": micro, "macro": macro, "per_type": per_type}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_folder", default="data/bioc/json/step_2_span_clf")
    parser.add_argument("--pred_file", default="test_pico_spans.json")
    parser.add_argument("--confidence_threshold", type=float, default=None,
                        help="Defaults to the prediction manifest threshold, or 0.5 for legacy files")
    parser.add_argument("--metrics_file", default=None)
    parser.add_argument("--baseline_file", default=None,
                        help="Optional aligned legacy pico_pred or pico_elements baseline")
    args = parser.parse_args()
    source = Path(args.pred_file)
    if not source.is_file():
        source = Path(args.input_folder) / args.pred_file
    if args.confidence_threshold is None:
        metadata_path = Path(str(source) + ".manifest.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        args.confidence_threshold = metadata.get("classification_threshold", 0.5)
    metrics = evaluate_predictions(read_jsonl(source), args.confidence_threshold)
    if args.baseline_file:
        reference = read_jsonl(source)
        baseline = read_jsonl(args.baseline_file)
        if len(reference) != len(baseline):
            raise ValueError("Baseline and reference must have the same number of sentences")
        converted = []
        for ref, pred in zip(reference, baseline):
            if any(key in pred and pred[key] != ref.get(key) for key in ("pmid", "tokens", "sentence_id")):
                raise ValueError("Baseline sentences do not align with references")
            row = dict(ref)
            if "pico_elements" in pred:
                row["pico_elements"] = pred["pico_elements"]
            elif "pico_pred" in pred:
                row["pico_elements"] = {}
                for label, start, length in gold_spans({"labels": pred["pico_pred"]}):
                    row["pico_elements"].setdefault(label, []).append({
                        "span_start": start, "span_length": length, "confidence": 1.0,
                    })
            else:
                raise ValueError("Baseline requires pico_pred or pico_elements")
            converted.append(row)
        metrics["baseline"] = evaluate_predictions(converted, args.confidence_threshold)
    output = args.metrics_file or str(source) + ".metrics.json"
    inputs = [source] + ([args.baseline_file] if args.baseline_file else [])
    write_json(output, {**metrics, "provenance": experiment_manifest(args, inputs)})
    for label, result in [*metrics["per_type"].items(), ("micro", metrics["micro"])]:
        print(f"{label}: TP={result['tp']} FP={result['fp']} FN={result['fn']} "
              f"P={100 * result['precision']:.2f} R={100 * result['recall']:.2f} "
              f"F1={100 * result['f1']:.2f}")
    print(f"macro F1={100 * metrics['macro']['f1']:.2f}; saved {output}")


if __name__ == "__main__":
    main()
