"""Train a multilabel P/I/O classifier with bounded windows and reproducible settings."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    Trainer, TrainingArguments, set_seed,
)

from pipeline_utils import (
    DEFAULT_MODEL, PICO_CLASSES, encode_windows, experiment_manifest,
    max_sequence_length, read_jsonl, write_json,
)


def tokenize_span_batch(batch, tokenizer, max_length, stride):
    encoded = encode_windows(tokenizer, batch["tokens"], max_length, stride)
    samples = encoded.pop("overflow_to_sample_mapping")
    encoded["labels"] = [[float(batch[label][sample]) for label in PICO_CLASSES]
                         for sample in samples]
    return dict(encoded)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predicted = np.asarray(logits) >= 0.0  # sigmoid(logit) >= .5, without overflow.
    p, r, f, _ = precision_recall_fscore_support(
        labels, predicted, average="macro", zero_division=0
    )
    return {"precision": p, "recall": r, "f1": f}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_folder", default="data/bioc/json")
    parser.add_argument("--model_checkpoint", default=DEFAULT_MODEL)
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--output_dir", default="pico_span/span_clf")
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--val_file", default=None)
    parser.add_argument("--test_file", default=None, help="Legacy option; test data are never used for training")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--use_cpu", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_checkpoint, revision=args.model_revision, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_checkpoint, revision=args.model_revision,
        problem_type="multi_label_classification", num_labels=3,
        id2label=dict(enumerate(PICO_CLASSES)),
        label2id={name: i for i, name in enumerate(PICO_CLASSES)},
    )
    length = max_sequence_length(tokenizer, model.config, args.max_length)
    files = [
        Path(args.train_file or Path(args.input_folder) / "train_span_clf.json"),
        Path(args.val_file or Path(args.input_folder) / "validation_span_clf.json"),
    ]
    data = {}
    for split, path in zip(("train", "validation"), files):
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f"Empty {split} set")
        for row in rows:
            if not row["tokens"] or any(name not in row for name in PICO_CLASSES):
                raise ValueError("Each span must contain tokens and all three binary labels")
        ds = Dataset.from_list(rows)
        data[split] = ds.map(
            lambda batch: tokenize_span_batch(batch, tokenizer, length, args.stride),
            batched=True, remove_columns=ds.column_names,
        )
    output = Path(args.output_dir) / (
        "span-clf-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    write_json(output / "run_manifest.json", experiment_manifest(
        args, files, encoder_config=model.config.to_dict(), max_length=length,
        long_span_policy="replicate span labels per window; average window scores at inference",
    ))
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output), learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size, num_train_epochs=args.epochs,
            weight_decay=args.weight_decay, eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
            seed=args.seed, data_seed=args.seed, max_steps=args.max_steps,
            use_cpu=args.use_cpu, report_to=[], save_total_limit=2,
        ),
        train_dataset=data["train"], eval_dataset=data["validation"],
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    trainer.save_state()
    write_json(output / "validation_metrics.json", trainer.evaluate())
    print(f"Model saved to {output}")


if __name__ == "__main__":
    main()
