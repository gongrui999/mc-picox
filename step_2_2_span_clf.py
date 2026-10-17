"""Classify boundary candidates and apply deterministic class-wise confidence NMS."""

import argparse
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pipeline_utils import (
    PICO_CLASSES, checkpoint_fingerprint, encode_windows, experiment_manifest, max_sequence_length, read_jsonl,
    sentence_id, write_json, write_jsonl,
)


class DatasetSplit(Enum):
    train = 0
    validation = 1
    test = 2


@dataclass(frozen=True)
class Span:
    start: int
    length: int


def extract_span_boundaries(boundary_pred):
    starts = [i for i, p in enumerate(boundary_pred) if p & 1]
    ends = [i for i, p in enumerate(boundary_pred) if p & 2]
    return [(s, e) for s in starts for e in ends if s <= e]


def text_span_iou(a, b):
    intersection = max(0, min(a.start + a.length, b.start + b.length) - max(a.start, b.start))
    union = a.length + b.length - intersection
    return intersection / union if union else 0.0


def nms_span(spans, span_class, confidence, iou_threshold=0.0):
    if not len(spans) == len(span_class) == len(confidence) or not 0 <= iou_threshold <= 1:
        raise ValueError("Invalid NMS inputs")
    order = sorted(range(len(spans)),
                   key=lambda i: (-confidence[i], spans[i].start, -spans[i].length, span_class[i]))
    keep = []
    for i in order:
        if any(span_class[i] == span_class[j] and (
            spans[i] == spans[j] or text_span_iou(spans[i], spans[j]) > iou_threshold
        ) for j in keep):
            continue
        keep.append(i)
    return ([spans[i] for i in keep], [span_class[i] for i in keep], [confidence[i] for i in keep])


def score_candidates(tokens, candidates, tokenizer, model, device, max_length=None, stride=64):
    """Cache all class scores; overlong spans use a disclosed mean across windows."""
    limit = max_sequence_length(tokenizer, model.config, max_length)
    classes = [model.config.id2label[i] for i in range(model.config.num_labels)]
    if set(classes) != set(PICO_CLASSES):
        raise ValueError("Span checkpoint must define PARTICIPANTS/INTERVENTIONS/OUTCOMES labels")
    scored = {}
    for start, end in sorted(set(candidates)):
        encoded = encode_windows(tokenizer, tokens[start:end + 1], limit, stride, tensors="pt")
        scores = []
        for i in range(len(encoded["input_ids"])):
            inputs = {k: encoded[k][i:i + 1].to(device)
                      for k in tokenizer.model_input_names if k in encoded}
            with torch.no_grad():
                scores.append(model(**inputs).logits[0].float().sigmoid())
        average = torch.stack(scores).mean(0)
        if not torch.isfinite(average).all():
            raise ValueError("Span model emitted non-finite probabilities")
        probability = average.cpu().tolist()
        scored[(start, end)] = dict(zip(classes, probability))
    return scored


def filter_scored_candidates(scored, threshold=0.5, iou_threshold=0.0):
    if not 0 <= threshold <= 1:
        raise ValueError("Classification threshold must be in [0,1]")
    spans, classes, scores = [], [], []
    for (start, end), probabilities in scored.items():
        for label, score in probabilities.items():
            if score >= threshold:
                spans.append(Span(start, end - start + 1))
                classes.append(label)
                scores.append(score)
    return nms_span(spans, classes, scores, iou_threshold)


def extract_pico_elements_util(tokens, boundary_pred, tokenizer, model, device,
                               threshold=0.5, iou_threshold=0.0, max_length=None, stride=64):
    return filter_scored_candidates(score_candidates(
        tokens, extract_span_boundaries(boundary_pred), tokenizer, model, device, max_length, stride
    ), threshold, iou_threshold)


def build_prediction_row(original, spans, classes, confidence, index=0):
    row = dict(original)
    row["sentence_id"] = sentence_id(original, index)
    n = len(row["tokens"])
    for key in ("start_confidence", "end_confidence"):
        if not isinstance(row.get(key), list) or len(row[key]) != n:
            raise ValueError(f"{key} must be a per-word list")
    elements = {}
    for span, label, score in zip(spans, classes, confidence):
        elements.setdefault(label, []).append({
            "span_start": span.start, "span_length": span.length, "confidence": score,
            "start_confidence": row["start_confidence"][span.start],
            "end_confidence": row["end_confidence"][span.start + span.length - 1],
        })
    row["pico_elements"] = elements
    return row


def extract_pico_elements(dataset_dict, dataset_split, output_path, model, tokenizer,
                          device, threshold=0.5, iou_threshold=0.0, max_length=None, stride=64):
    rows = []
    for i, original in enumerate(tqdm(dataset_dict[dataset_split.name], desc="Span classification")):
        spans, labels, confidence = extract_pico_elements_util(
            original["tokens"], original["boundary_pred"], tokenizer, model, device,
            threshold, iou_threshold, max_length, stride,
        )
        rows.append(build_prediction_row(original, spans, labels, confidence, i))
    output = Path(output_path) / f"{dataset_split.name}_pico_spans.json"
    write_jsonl(output, rows)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--span_clf_model_path", required=True)
    parser.add_argument("--input_folder", default="data/bioc/json/step_1_boundary_pred")
    parser.add_argument("--input_file", default=None)
    parser.add_argument("--output_path", default="data/bioc/json/step_2_span_clf")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold_config", default=None)
    parser.add_argument("--nms_iou_threshold", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--base_model", default=None, help="Legacy tokenizer override; prefer checkpoint tokenizer")
    args = parser.parse_args()
    if args.threshold_config:
        calibration = json.loads(Path(args.threshold_config).read_text(encoding="utf-8"))
        if calibration.get("split") != "validation":
            raise ValueError("Threshold config must originate from validation")
        if calibration.get("span_checkpoint_sha256") != checkpoint_fingerprint(args.span_clf_model_path):
            raise ValueError("Span checkpoint changed after calibration")
        args.threshold = calibration["classification_threshold"]
        args.nms_iou_threshold = calibration["nms_iou_threshold"]
        args.max_length = calibration["span_max_length"]
        args.stride = calibration["span_stride"]
        if args.base_model:
            raise ValueError("A calibrated run must use the saved checkpoint tokenizer")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(args.span_clf_model_path).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model or args.span_clf_model_path, use_fast=True)
    source = Path(args.input_file or Path(args.input_folder) / f"{args.split}_boundary_pred.json")
    if args.threshold_config:
        metadata = json.loads(Path(str(source) + ".manifest.json").read_text(encoding="utf-8"))
        for key in ("boundary_checkpoint_sha256", "start_threshold", "end_threshold",
                    "decode_rule", "max_length", "stride"):
            if metadata.get(key) != calibration.get(key):
                raise ValueError(f"Boundary probabilities do not match calibrated setting {key}")
    output = extract_pico_elements(
        {args.split: read_jsonl(source)}, DatasetSplit[args.split], args.output_path,
        model, tokenizer, device, args.threshold, args.nms_iou_threshold,
        args.max_length, args.stride,
    )
    write_json(str(output) + ".manifest.json", experiment_manifest(
        args, [source], span_model_path=str(Path(args.span_clf_model_path).resolve()),
        classification_threshold=args.threshold, nms_iou_threshold=args.nms_iou_threshold,
        long_span_policy="mean of per-window sigmoid scores", classification_comparison=">=",
    ))
    print(f"Saved predictions to {output}")


if __name__ == "__main__":
    main()
