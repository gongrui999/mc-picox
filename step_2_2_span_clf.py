"""
Step 2-2: End-to-end PICO span extraction (boundary + span classification).

Loads boundary predictions from Step 1-2 and a trained span classifier from
Step 2-1. For each sentence, enumerates candidate spans from predicted
boundaries, classifies each candidate, and applies NMS to remove duplicates.

Usage:
    python step_2_2_span_clf.py \
        --base_model microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract \
        --span_clf_model_path pico_span/span_clf/<model_name> \
        --input_folder data/bioc/json/step_1_boundary_pred \
        --output_path data/bioc/json/step_2_span_clf \
        --split test \
        --threshold 0.5
"""

import argparse
import json
import os
from enum import Enum

import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class DatasetSplit(Enum):
    train = 0
    validation = 1
    test = 2


class PicoType(Enum):
    PARTICIPANTS = 4
    INTERVENTIONS = 2
    OUTCOMES = 1


PICO_CLASSES = ["PARTICIPANTS", "INTERVENTIONS", "OUTCOMES"]


# ---------------------------------------------------------------------------
# Span helpers
# ---------------------------------------------------------------------------

class Span:
    def __init__(self, start, length):
        self.start = start
        self.length = length

    def __repr__(self):
        return f"Span(start={self.start}, length={self.length})"

    def __eq__(self, other):
        return (self.start, int(self.length)) == (other.start, int(other.length))


def extract_span_boundaries(boundary_pred):
    starts = [i for i, p in enumerate(boundary_pred) if p & 1]
    ends = [i for i, p in enumerate(boundary_pred) if p & 2]
    return [(s, e) for s in starts for e in ends if s <= e]


def text_span_iou(a: Span, b: Span) -> float:
    a_start, a_end = a.start, a.start + a.length
    b_start, b_end = b.start, b.start + b.length
    u = max(a_end, b_end) - min(a_start, b_start)
    if u == 0.0:
        return 0.0
    i = 0.0
    if a_start <= b_end and b_start <= a_end:
        i = min(a_end, b_end) - max(a_start, b_start)
    return i / u


def nms_span(spans, span_class, confidence, iou_threshold=0.0):
    keep = [True] * len(spans)
    for i in range(len(spans) - 1):
        if not keep[i]:
            continue
        for j in range(i + 1, len(spans)):
            if not keep[j] or span_class[i] != span_class[j]:
                continue
            iou = text_span_iou(spans[i], spans[j])
            if iou > iou_threshold:
                if spans[i].length > spans[j].length:
                    keep[i] = False
                else:
                    keep[j] = False

    return (
        [s for s, k in zip(spans, keep) if k],
        [c for c, k in zip(span_class, keep) if k],
        [c for c, k in zip(confidence, keep) if k],
    )


# ---------------------------------------------------------------------------
# PICO element extraction
# ---------------------------------------------------------------------------

def extract_pico_elements_util(tokens, boundary_pred, tokenizer, model,
                               device, threshold=0.5):
    candidate_spans = extract_span_boundaries(boundary_pred)
    spans, pico_class, confidence = [], [], []

    for start, end in candidate_spans:
        content = tokens[start: end + 1]
        x = tokenizer(content, padding=True, return_tensors="pt", is_split_into_words=True)
        x = {k: v.to(device) for k, v in x.items()}

        with torch.no_grad():
            y = model(**x)

        probability = np.squeeze(
            torch.nn.functional.sigmoid(y.logits).cpu().numpy()
        ).tolist()

        if isinstance(probability, float):
            probability = [probability]

        for i, p in enumerate(probability):
            if p < threshold:
                continue
            pico_class.append(model.config.id2label[i])
            confidence.append(p)
            spans.append(Span(start=start, length=len(content)))

    return nms_span(spans, pico_class, confidence)


def extract_pico_elements(dataset_dict, dataset_split, output_path,
                          model, tokenizer, device, threshold=0.5):
    output_file = os.path.join(output_path, f"{dataset_split.name}_pico_spans.json")
    os.makedirs(output_path, exist_ok=True)

    dataset = dataset_dict[dataset_split.name]

    with open(output_file, "w+", encoding="utf-8") as fout:
        for i in tqdm(range(len(dataset)), desc=f"Span classification ({dataset_split.name})"):
            row = {
                "pmid": dataset["pmid"][i],
                "tokens": dataset["tokens"][i],
                "original_labels": dataset["original_labels"][i],
                "boundary_pred": dataset["boundary_pred"][i],
            }
            start_confidence = dataset["start_confidence"][i]
            end_confidence = dataset["end_confidence"][i]

            spans, pico_cls, conf = extract_pico_elements_util(
                row["tokens"], row["boundary_pred"],
                tokenizer, model, device,
                threshold=threshold,
            )

            row["pico_elements"] = {}
            for s, c, cf in zip(spans, pico_cls, conf):
                if c not in row["pico_elements"]:
                    row["pico_elements"][c] = []
                span_dict = {
                    "span_start": s.start,
                    "span_length": s.length,
                    "confidence": cf,
                }
                row["pico_elements"][c].append(span_dict)
                row["start_confidence"] = start_confidence[s.start]
                row["end_confidence"] = end_confidence[s.start + s.length - 1]

            fout.write(f"{json.dumps(row)}\n")

    print(f"Saved PICO span predictions to {output_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 2-2: End-to-end span extraction")
    parser.add_argument("--base_model", type=str,
                        default="microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract")
    parser.add_argument("--span_clf_model_path", type=str, required=True,
                        help="Path to the trained span classifier checkpoint")
    parser.add_argument("--input_folder", type=str,
                        default="data/bioc/json/step_1_boundary_pred")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Override input JSONL file (default: <split>_boundary_pred.json)")
    parser.add_argument("--output_path", type=str, default="data/bioc/json/step_2_span_clf")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "validation", "test"])
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Span classification probability threshold")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load model ----
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.span_clf_model_path)
    model.to(device)
    model.eval()

    # ---- Load boundary prediction data ----
    input_file = args.input_file or os.path.join(
        args.input_folder, f"{args.split}_boundary_pred.json"
    )
    ebm_nlp = load_dataset(
        "json",
        data_files={
            "train": input_file,
            "validation": input_file,
            "test": input_file,
        },
    )
    print(ebm_nlp)

    split_enum = DatasetSplit[args.split]
    extract_pico_elements(
        ebm_nlp, split_enum, args.output_path, model, tokenizer, device,
        threshold=args.threshold,
    )

    print("Done.")


if __name__ == "__main__":
    main()
