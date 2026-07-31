"""
Step 1-2: Run boundary prediction on the dataset using a trained boundary model.

Supports both baseline (AutoModelForTokenClassification) and multichannel models.
Auto-detects model type from saved config.

Usage:
    python step_1_2_boundary_prediction.py \
        --base_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
        --boundary_model_path pico_span/boundary_models/<model_name> \
        --input_folder data/bioc/json \
        --output_path data/bioc/json/step_1_boundary_pred \
        --split test \
        --threshold 0.25
"""

import argparse
import json
import os
from enum import Enum

import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from model import MultiChannelBoundaryModel


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


class SpanBoundary(Enum):
    outside = 0
    start = 1
    end = 2
    both = 3
    inside = 4


# ---------------------------------------------------------------------------
# Merge predictions back to word level
# ---------------------------------------------------------------------------

def merge_boundary_labels_by_word_id(boundary_pred, word_ids, num_words):
    labels = [0] * num_words
    for pred, word_id in zip(boundary_pred, word_ids):
        if word_id is None:
            continue
        labels[word_id] = labels[word_id] | int(pred)
    return labels


def merge_boundary_probability_by_word_id(y_prob, word_ids, num_words):
    start_prob = [0.0] * num_words
    end_prob = [0.0] * num_words
    for prob, word_id in zip(y_prob, word_ids):
        if word_id is None:
            continue
        start_prob[word_id] = max(
            start_prob[word_id],
            float(prob[SpanBoundary.start.value]),
            float(prob[SpanBoundary.both.value]),
        )
        end_prob[word_id] = max(
            end_prob[word_id],
            float(prob[SpanBoundary.end.value]),
            float(prob[SpanBoundary.both.value]),
        )
    return start_prob, end_prob


# ---------------------------------------------------------------------------
# Threshold-based boundary label extraction
# ---------------------------------------------------------------------------

class BoundaryLabel:
    def __init__(self):
        self.value = 0

    def set_start(self):
        self.value |= SpanBoundary.start.value

    def set_end(self):
        self.value |= SpanBoundary.end.value


def extract_boundary_from_prob_dist(prob_dist, threshold):
    label = BoundaryLabel()
    if prob_dist[SpanBoundary.start.value] > threshold or prob_dist[SpanBoundary.both.value] > threshold:
        label.set_start()
    if prob_dist[SpanBoundary.end.value] > threshold or prob_dist[SpanBoundary.both.value] > threshold:
        label.set_end()
    return label.value


# ---------------------------------------------------------------------------
# Generate boundary predictions
# ---------------------------------------------------------------------------

def generate_boundary_labels(dataset_dict, dataset_split, output_path,
                             model, tokenizer, boundary_threshold=0.5):
    output_file = os.path.join(output_path, f"{dataset_split.name}_boundary_pred.json")
    os.makedirs(output_path, exist_ok=True)

    dataset = dataset_dict[dataset_split.name]
    device = next(model.parameters()).device

    with open(output_file, "w+", encoding="utf-8") as fout:
        for i in tqdm(range(len(dataset)), desc=f"Boundary prediction ({dataset_split.name})"):
            row = {
                "pmid": dataset["pmid"][i],
                "tokens": dataset["tokens"][i],
                "original_labels": dataset["labels"][i],
            }
            x = tokenizer(
                row["tokens"], padding=True, return_tensors="pt", is_split_into_words=True
            )
            x = {k: v.to(device) for k, v in x.items()}

            with torch.no_grad():
                y = model(**x)

            logits = y["logits"] if isinstance(y, dict) else y.logits
            y_prob = np.squeeze(
                torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
            )
            y_pred = [extract_boundary_from_prob_dist(p, boundary_threshold) for p in y_prob]

            word_ids = tokenizer(
                row["tokens"], is_split_into_words=True
            ).word_ids()

            row["boundary_pred"] = merge_boundary_labels_by_word_id(
                y_pred, word_ids, len(row["tokens"])
            )
            start_prob, end_prob = merge_boundary_probability_by_word_id(
                y_prob, word_ids, len(row["tokens"])
            )
            row["start_confidence"] = start_prob
            row["end_confidence"] = end_prob
            fout.write(f"{json.dumps(row)}\n")

    print(f"Saved boundary predictions to {output_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 1-2: Boundary prediction")
    parser.add_argument("--base_model", type=str,
                        default="microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract")
    parser.add_argument("--boundary_model_path", type=str, required=True,
                        help="Path to the trained boundary model checkpoint")
    parser.add_argument("--input_folder", type=str, default="data/bioc/json")
    parser.add_argument("--output_path", type=str, default="data/bioc/json/step_1_boundary_pred")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "validation", "test"])
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Boundary probability threshold")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda / cpu). Auto-detected if not set.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load model ----
    tokenizer = AutoTokenizer.from_pretrained(args.boundary_model_path)

    config_path = os.path.join(args.boundary_model_path, "model_config.pt")
    if os.path.exists(config_path):
        config = torch.load(config_path, map_location="cpu", weights_only=True)
        print(f"Detected model type: {config['model_type']}")
        model = MultiChannelBoundaryModel(
            bert_model_name=config["bert_model_name"],
            num_labels=config["num_labels"],
        )
        state_dict = torch.load(
            os.path.join(args.boundary_model_path, "model.pt"),
            map_location="cpu", weights_only=True,
        )
        model.load_state_dict(state_dict)
    else:
        print("Detected model type: baseline")
        model = AutoModelForTokenClassification.from_pretrained(args.boundary_model_path)

    model.to(device)
    model.eval()

    # ---- Load data ----
    split_file = os.path.join(args.input_folder, f"{args.split}.json")
    ebm_nlp = load_dataset(
        "json",
        data_files={
            "train": split_file,
            "validation": split_file,
            "test": split_file,
        },
    )
    keep = ["pmid", "tokens", "labels"]
    for s in ebm_nlp:
        remove = [f for f in ebm_nlp[s].features if f not in keep]
        if remove:
            ebm_nlp[s] = ebm_nlp[s].remove_columns(remove)

    split_enum = DatasetSplit[args.split]
    generate_boundary_labels(
        ebm_nlp, split_enum, args.output_path, model, tokenizer,
        boundary_threshold=args.threshold,
    )

    print("Done.")


if __name__ == "__main__":
    main()
