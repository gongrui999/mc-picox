"""Select separate start/end thresholds using validation end-to-end strict micro F1."""

import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pipeline_utils import (
    checkpoint_fingerprint, experiment_manifest, gold_spans, read_jsonl, write_json,
)
from step_1_2_boundary_prediction import decode_word_scores
from step_2_2_span_clf import extract_span_boundaries, filter_scored_candidates, score_candidates
from step_3_evaluate import scores


def search_thresholds(rows, cached_scores, classification_threshold=0.5, nms_iou_threshold=0.0):
    if len(rows) != len(cached_scores) or not rows:
        raise ValueError("Each nonempty validation row requires a candidate score cache")
    grid, best, best_key = [], None, None
    for si in range(5, 31):
        for ei in range(5, 31):
            ts, te = si / 100, ei / 100
            tp = fp = fn = candidates = 0
            for row, cached in zip(rows, cached_scores):
                selected = {(s, e): value for (s, e), value in cached.items()
                            if row["start_confidence"][s] > ts and row["end_confidence"][e] > te}
                candidates += len(selected)
                spans, classes, _ = filter_scored_candidates(
                    selected, classification_threshold, nms_iou_threshold
                )
                predicted = {(label, span.start, span.length) for span, label in zip(spans, classes)}
                truth = gold_spans(row)
                tp += len(predicted & truth)
                fp += len(predicted - truth)
                fn += len(truth - predicted)
            metrics = scores({"tp": tp, "fp": fp, "fn": fn})
            result = {"start_threshold": ts, "end_threshold": te,
                      "candidate_count": candidates, **metrics}
            grid.append(result)
            key = (metrics["f1"], metrics["recall"], -candidates, ts + te, ts, te)
            if best_key is None or key > best_key:
                best_key, best = key, result
    return best, grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation_predictions", required=True)
    parser.add_argument("--span_clf_model_path", required=True)
    parser.add_argument("--boundary_model_path", default=None,
                        help="Override a moved checkpoint path; its checksum must match cached probabilities")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--classification_threshold", type=float, default=0.5)
    parser.add_argument("--nms_iou_threshold", type=float, default=0.0)
    parser.add_argument("--span_max_length", type=int, default=None)
    parser.add_argument("--span_stride", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    source = Path(args.validation_predictions)
    metadata = json.loads(Path(str(source) + ".manifest.json").read_text(encoding="utf-8"))
    rows = read_jsonl(source)
    if metadata.get("split") != "validation" or any(r.get("split") != "validation" for r in rows):
        raise ValueError("Threshold search accepts validation predictions only")
    boundary_path = args.boundary_model_path or metadata["boundary_model_path"]
    if metadata["boundary_checkpoint_sha256"] != checkpoint_fingerprint(boundary_path):
        raise ValueError("Cached probabilities belong to a different checkpoint state")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.span_clf_model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.span_clf_model_path
    ).to(device).eval()
    cached = []
    for row in tqdm(rows, desc="Cache validation candidate classifications"):
        for key in ("start_confidence", "end_confidence"):
            if len(row[key]) != len(row["tokens"]) or any(not 0 <= p <= 1 for p in row[key]):
                raise ValueError("Malformed word-level probability arrays")
        candidates = extract_span_boundaries(decode_word_scores(
            row["start_confidence"], row["end_confidence"], 0.05, 0.05
        ))
        cached.append(score_candidates(
            row["tokens"], candidates, tokenizer, model, device,
            args.span_max_length, args.span_stride,
        ))
    best, grid = search_thresholds(
        rows, cached, args.classification_threshold, args.nms_iou_threshold
    )
    result = {
        **best, "schema_version": 1, "split": "validation",
        "objective": "strict end-to-end micro F1",
        "tie_break": "higher recall, fewer candidates, higher threshold sum, start, end",
        "decode_rule": metadata["decode_rule"], "max_length": metadata["max_length"],
        "stride": metadata["stride"], "boundary_model_path": boundary_path,
        "boundary_checkpoint_sha256": metadata["boundary_checkpoint_sha256"],
        "span_model_path": str(Path(args.span_clf_model_path).resolve()),
        "span_checkpoint_sha256": checkpoint_fingerprint(args.span_clf_model_path),
        "classification_threshold": args.classification_threshold,
        "nms_iou_threshold": args.nms_iou_threshold,
        "span_max_length": args.span_max_length, "span_stride": args.span_stride,
        "grid_size": len(grid), "provenance": experiment_manifest(args, [source]),
    }
    write_json(args.output_file, result)
    write_json(str(args.output_file) + ".grid.json", grid)
    print(f"Selected start={best['start_threshold']:.2f} end={best['end_threshold']:.2f}; "
          f"validation F1={100 * best['f1']:.2f}. Saved {args.output_file}")


if __name__ == "__main__":
    main()
