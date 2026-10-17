"""Predict complete word-level boundary probabilities using bounded overlapping windows."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from model import MultiChannelBoundaryModel
from pipeline_utils import (
    checkpoint_fingerprint, encode_windows, experiment_manifest, max_sequence_length, read_jsonl,
    sentence_id, write_json, write_jsonl,
)


def load_boundary_model(path, device):
    path = Path(path)
    config_path = path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    if config.get("model_type") == "mc_picox_boundary":
        model = MultiChannelBoundaryModel.from_pretrained(path)
    elif (path / "model_config.pt").exists():
        # Compatibility with the original local multichannel format.
        old = torch.load(path / "model_config.pt", map_location="cpu", weights_only=True)
        model = MultiChannelBoundaryModel(
            bert_model_name=old["bert_model_name"], num_labels=old["num_labels"], loss_type="ce"
        )
        model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
    else:
        model = AutoModelForTokenClassification.from_pretrained(path)
    model.to(device).eval()
    return model, AutoTokenizer.from_pretrained(path, use_fast=True)


def extract_boundary_from_prob_dist(prob, threshold=None, start_threshold=None,
                                    end_threshold=None, decode_rule="sum"):
    threshold = 0.25 if threshold is None else threshold
    ts = threshold if start_threshold is None else start_threshold
    te = threshold if end_threshold is None else end_threshold
    if not (0 <= ts <= 1 and 0 <= te <= 1):
        raise ValueError("Boundary thresholds must be in [0,1]")
    if decode_rule not in ("sum", "legacy_or"):
        raise ValueError("Unknown decode rule")
    qs = prob[1] + prob[3] if decode_rule == "sum" else max(prob[1], prob[3])
    qe = prob[2] + prob[3] if decode_rule == "sum" else max(prob[2], prob[3])
    return int(qs > ts) | (2 * int(qe > te))


def merge_boundary_probability_by_word_id(y_prob, word_ids, num_words, decode_rule="sum"):
    start, end = [0.0] * num_words, [0.0] * num_words
    for prob, word in zip(y_prob, word_ids):
        if word is None:
            continue
        qs = float(prob[1] + prob[3]) if decode_rule == "sum" else float(max(prob[1], prob[3]))
        qe = float(prob[2] + prob[3]) if decode_rule == "sum" else float(max(prob[2], prob[3]))
        start[word], end[word] = max(start[word], qs), max(end[word], qe)
    return start, end


def predict_word_probabilities(tokens, model, tokenizer, max_length=None, stride=64,
                               decode_rule="sum"):
    if not tokens:
        return [], []
    limit = max_sequence_length(tokenizer, model.config, max_length)
    encoded = encode_windows(tokenizer, tokens, limit, stride, tensors="pt")
    start, end = np.zeros(len(tokens)), np.zeros(len(tokens))
    covered = set()
    device = next(model.parameters()).device
    for i in range(len(encoded["input_ids"])):
        word_ids = encoded.word_ids(i)  # Reuse precisely the encoding sent to the model.
        inputs = {k: encoded[k][i:i + 1].to(device)
                  for k in tokenizer.model_input_names if k in encoded}
        with torch.no_grad():
            probs = model(**inputs).logits[0].float().softmax(-1).cpu().numpy()
        if not np.isfinite(probs).all():
            raise ValueError("Boundary model emitted non-finite probabilities")
        a, b = merge_boundary_probability_by_word_id(probs, word_ids, len(tokens), decode_rule)
        start, end = np.maximum(start, a), np.maximum(end, b)
        covered.update(w for w in word_ids if w is not None)
    required = {i for i, token in enumerate(tokens) if not token.isspace()}
    if required - covered:
        raise ValueError("Tokenizer dropped non-whitespace words; normalize unsupported tokens explicitly")
    return start.tolist(), end.tolist()


def decode_word_scores(start, end, ts, te):
    if not 0 <= ts <= 1 or not 0 <= te <= 1 or len(start) != len(end):
        raise ValueError("Invalid boundary scores or thresholds")
    return [int(a > ts) | (2 * int(b > te)) for a, b in zip(start, end)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary_model_path", required=True)
    parser.add_argument("--input_folder", default="data/bioc/json")
    parser.add_argument("--output_path", default="data/bioc/json/step_1_boundary_pred")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--start_threshold", type=float, default=None)
    parser.add_argument("--end_threshold", type=float, default=None)
    parser.add_argument("--threshold_config", default=None)
    parser.add_argument("--decode_rule", choices=["sum", "legacy_or"], default="sum")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--base_model", default=None, help="Deprecated; models/tokenizers load from checkpoint")
    args = parser.parse_args()
    ts = args.threshold if args.start_threshold is None else args.start_threshold
    te = args.threshold if args.end_threshold is None else args.end_threshold
    if args.threshold_config:
        calibration = json.loads(Path(args.threshold_config).read_text(encoding="utf-8"))
        if calibration.get("split") != "validation":
            raise ValueError("Calibration must originate from validation only")
        ts, te = calibration["start_threshold"], calibration["end_threshold"]
        args.decode_rule = calibration["decode_rule"]
        if calibration.get("boundary_checkpoint_sha256") != checkpoint_fingerprint(args.boundary_model_path):
            raise ValueError("Boundary checkpoint changed after calibration")
        if args.max_length is None:
            args.max_length = calibration.get("max_length")
        if args.stride != calibration.get("stride", args.stride):
            raise ValueError("Use the same stride as validation calibration")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_boundary_model(args.boundary_model_path, device)
    source = Path(args.input_folder) / f"{args.split}.json"
    rows = read_jsonl(source)
    predictions = []
    for i, original in enumerate(tqdm(rows, desc="Boundary probabilities")):
        row = dict(original)
        row["original_labels"] = row.pop("labels", row.get("original_labels", None))
        if row["original_labels"] is None:
            raise ValueError("Input requires token labels")
        row["sentence_id"], row["split"] = sentence_id(original, i), args.split
        start, end = predict_word_probabilities(
            row["tokens"], model, tokenizer, args.max_length, args.stride, args.decode_rule
        )
        row.update(start_confidence=start, end_confidence=end,
                   boundary_pred=decode_word_scores(start, end, ts, te))
        predictions.append(row)
    output = Path(args.output_path) / f"{args.split}_boundary_pred.json"
    write_jsonl(output, predictions)
    write_json(str(output) + ".manifest.json", experiment_manifest(
        args, [source], boundary_model_path=str(Path(args.boundary_model_path).resolve()),
        max_length=max_sequence_length(tokenizer, model.config, args.max_length),
        stride=args.stride, decode_rule=args.decode_rule,
        split=args.split, start_threshold=ts, end_threshold=te,
        boundary_checkpoint_sha256=checkpoint_fingerprint(args.boundary_model_path),
        aggregation="sum class evidence then max over subwords and overlapping windows",
    ))
    print(f"Saved {len(predictions)} records to {output}")


if __name__ == "__main__":
    main()
