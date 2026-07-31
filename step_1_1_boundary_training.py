"""
Step 1-1: Train a boundary detector for PICO span extraction.

Supports two modes:
  --model_type baseline   : single-layer AutoModelForTokenClassification (original)
  --model_type multichannel : MultiChannelBoundaryModel with gated fusion (improved)

Usage:
    python step_1_1_boundary_training.py \
        --input_folder data/bioc/json \
        --model_checkpoint microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
        --output_dir pico_span/boundary_models \
        --model_type multichannel \
        --epochs 8 --learning_rate 5e-5
"""

import argparse
import os
from datetime import datetime
from enum import Enum

import evaluate
import numpy as np
import torch
from datasets import ClassLabel, Sequence, load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from model import MultiChannelBoundaryModel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOUNDARY_LABELS = [0, 1, 2, 3, 4]  # OUT, START, END, BOTH, IN


class PicoType(Enum):
    PARTICIPANTS = 4
    INTERVENTIONS = 2
    OUTCOMES = 1


# ---------------------------------------------------------------------------
# Boundary label extraction from PICO bitmask labels
# ---------------------------------------------------------------------------

def start_appears(a: int, b: int) -> bool:
    for p in PicoType:
        if (a & p.value) < (b & p.value):
            return True
    return False


def end_appears(a: int, b: int) -> bool:
    for p in PicoType:
        if (a & p.value) > (b & p.value):
            return True
    return False


def extract_boundary_labels(labels):
    boundary_labels = []
    for i, l in enumerate(labels):
        if l > 0:
            b = 0
            is_start = (i == 0) or start_appears(labels[i - 1], l)
            is_end = (i == len(labels) - 1) or end_appears(l, labels[i + 1])
            if is_start:
                b |= 1
            if is_end:
                b |= 2
            if not is_start and not is_end:
                b = 4
            boundary_labels.append(b)
        else:
            boundary_labels.append(0)
    return boundary_labels


# ---------------------------------------------------------------------------
# Token / label alignment for word-piece tokenizers
# ---------------------------------------------------------------------------

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            new_labels.append(labels[word_id])
    return new_labels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

token_precision_metric = evaluate.load("precision")
token_recall_metric = evaluate.load("recall")
token_f1_metric = evaluate.load("f1")


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    decoded_labels = [[l for l in label if l != -100] for label in labels]
    decoded_predictions = [
        [p for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    flat_labels = [l for dl in decoded_labels for l in dl]
    flat_predictions = [p for dp in decoded_predictions for p in dp]

    token_precision = token_precision_metric.compute(
        predictions=flat_predictions, references=flat_labels, average="macro"
    )
    token_recall = token_recall_metric.compute(
        predictions=flat_predictions, references=flat_labels, average="macro"
    )
    token_f1 = token_f1_metric.compute(
        predictions=flat_predictions, references=flat_labels, average="macro"
    )

    start_tp = start_fp = start_fn = 0
    end_tp = end_fp = end_fn = 0

    for label, pred in zip(flat_labels, flat_predictions):
        if label in (0, 4):
            if pred == 1:
                start_fp += 1
            elif pred == 2:
                end_fp += 1
            elif pred == 3:
                start_fp += 1
                end_fp += 1
        elif label == 1:
            if pred in (0, 4):
                start_fn += 1
            elif pred == 1:
                start_tp += 1
            elif pred == 2:
                start_fn += 1
                end_fp += 1
            elif pred == 3:
                start_tp += 1
                end_fp += 1
        elif label == 2:
            if pred in (0, 4):
                end_fn += 1
            elif pred == 1:
                start_fp += 1
                end_fn += 1
            elif pred == 2:
                end_tp += 1
            elif pred == 3:
                start_fp += 1
                end_tp += 1
        elif label == 3:
            if pred in (0, 4):
                start_fn += 1
                end_fn += 1
            elif pred == 1:
                start_tp += 1
                end_fn += 1  # missed end
            elif pred == 2:
                start_fn += 1  # missed start
                end_tp += 1
            elif pred == 3:
                start_tp += 1
                end_tp += 1

    def safe_div(a, b):
        return a / b if b else 0.0

    sp = safe_div(start_tp, start_tp + start_fp)
    sr = safe_div(start_tp, start_tp + start_fn)
    sf = safe_div(2 * sp * sr, sp + sr) if start_tp else 0.0

    ep = safe_div(end_tp, end_tp + end_fp)
    er = safe_div(end_tp, end_tp + end_fn)
    ef = safe_div(2 * ep * er, ep + er) if end_tp else 0.0

    return {
        "overall_precision": token_precision["precision"],
        "overall_recall": token_recall["recall"],
        "overall_f1": token_f1["f1"],
        "start_precision": sp,
        "start_recall": sr,
        "start_f1": sf,
        "end_precision": ep,
        "end_recall": er,
        "end_f1": ef,
    }


# ---------------------------------------------------------------------------
# Custom Trainer for MultiChannelBoundaryModel
# ---------------------------------------------------------------------------

class MultiChannelTrainer(Trainer):
    """Trainer subclass that handles the dict output from MultiChannelBoundaryModel."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs["loss"]
        # Trainer expects outputs with a .logits attribute for compute_metrics
        logits = outputs["logits"]

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        out.loss = loss
        return (loss, out) if return_outputs else loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 1-1: Boundary detector training")
    parser.add_argument("--input_folder", type=str, default="data/bioc/json")
    parser.add_argument("--model_checkpoint", type=str,
                        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    parser.add_argument("--output_dir", type=str, default="pico_span/boundary_models")
    parser.add_argument("--model_type", type=str, default="multichannel",
                        choices=["baseline", "multichannel"],
                        help="baseline = original single-head, multichannel = gated fusion")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    # ---- Load data ----
    ebm_nlp = load_dataset(
        "json",
        data_files={
            "train": os.path.join(args.input_folder, "train.json"),
            "validation": os.path.join(args.input_folder, "validation.json"),
            "test": os.path.join(args.input_folder, "test.json"),
        },
    )
    keep = ["pmid", "tokens", "labels"]
    for split in ebm_nlp:
        remove = [f for f in ebm_nlp[split].features if f not in keep]
        if remove:
            ebm_nlp[split] = ebm_nlp[split].remove_columns(remove)

    print(ebm_nlp)

    # ---- Tokenizer & model ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    id2label = {i: str(label) for i, label in enumerate(BOUNDARY_LABELS)}
    label2id = {v: k for k, v in id2label.items()}

    if args.model_type == "baseline":
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_checkpoint,
            id2label=id2label,
            label2id=label2id,
        )
    else:
        model = MultiChannelBoundaryModel(
            bert_model_name=args.model_checkpoint,
            num_labels=len(BOUNDARY_LABELS),
        )

    print(f"\nModel type: {args.model_type}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}  Trainable: {trainable_params:,}\n")

    # ---- Tokenize & align ----
    def tokenize_and_align_labels(dataset):
        tokenized_inputs = tokenizer(
            dataset["tokens"], truncation=True, is_split_into_words=True
        )
        new_labels = []
        for i, labels in enumerate(dataset["labels"]):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(
                align_labels_with_tokens(extract_boundary_labels(labels), word_ids)
            )
        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    tokenized_dataset = ebm_nlp.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=["pmid", "tokens"],
    )
    tokenized_dataset = tokenized_dataset.cast_column(
        "labels", Sequence(ClassLabel(names=[str(b) for b in BOUNDARY_LABELS]))
    )
    print(tokenized_dataset)

    # ---- Training ----
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_name = f"boundaries-{args.model_type}-ebm_nlp-{timestamp}"
    run_output_dir = os.path.join(args.output_dir, model_name)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        run_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="start_recall",
        greater_is_better=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        push_to_hub=False,
    )

    TrainerClass = MultiChannelTrainer if args.model_type == "multichannel" else Trainer

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    trainer.train()

    # ---- Save ----
    save_path = os.path.join(args.output_dir, model_name)
    os.makedirs(save_path, exist_ok=True)

    if args.model_type == "baseline":
        model.save_pretrained(save_path)
    else:
        torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
        # Save model config for loading later
        config = {
            "bert_model_name": args.model_checkpoint,
            "num_labels": len(BOUNDARY_LABELS),
            "model_type": args.model_type,
        }
        torch.save(config, os.path.join(save_path, "model_config.pt"))

    tokenizer.save_pretrained(save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
