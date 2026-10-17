"""Audit endpoint distance: min over predictions of max(start error, end error)."""

import argparse

from pipeline_utils import experiment_manifest, gold_spans, read_jsonl, write_json
from step_2_2_span_clf import extract_span_boundaries
from step_3_evaluate import predicted_spans


def boundary_distance_counts(rows, mode="predicted_spans", confidence_threshold=0.5):
    counts = {"exact": 0, "offset_1": 0, "offset_2": 0, "beyond_2_or_missing": 0}
    for row in rows:
        predicted = predicted_spans(row, confidence_threshold)
        for label, start, length in gold_spans(row):
            end = start + length - 1
            if mode == "predicted_spans":
                candidates = [(s, s + n - 1) for t, s, n in predicted if t == label]
            elif mode == "boundary_pairs":
                candidates = extract_span_boundaries(row["boundary_pred"])
            else:
                raise ValueError("Unknown distance audit mode")
            distance = min((max(abs(s - start), abs(e - end)) for s, e in candidates),
                           default=float("inf"))
            key = {0: "exact", 1: "offset_1", 2: "offset_2"}.get(
                distance, "beyond_2_or_missing"
            )
            counts[key] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--mode", choices=["predicted_spans", "boundary_pairs"],
                        default="predicted_spans")
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    args = parser.parse_args()
    counts = boundary_distance_counts(read_jsonl(args.pred_file), args.mode, args.confidence_threshold)
    write_json(args.output_file, {"counts": counts,
                                 "provenance": experiment_manifest(args, [args.pred_file])})
    print(counts)


if __name__ == "__main__":
    main()
