"""Train baseline or gated boundary models with offline metrics and HF checkpoints."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
    Trainer, TrainingArguments, set_seed,
)

from model import DEFAULT_ALPHA, MultiChannelBoundaryModel
from pipeline_utils import (
    DEFAULT_MODEL, boundary_labels, encode_windows, experiment_manifest,
    max_sequence_length, read_jsonl, write_json,
)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    valid = labels != -100
    truth, predicted = labels[valid], predictions[valid]
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, average="macro", zero_division=0
    )
    result = {"overall_precision": precision, "overall_recall": recall, "overall_f1": f1}
    combined_tp = combined_gold = 0
    for name, kinds in (("start", (1, 3)), ("end", (2, 3))):
        gold, pred = np.isin(truth, kinds), np.isin(predicted, kinds)
        tp, ng, npred = int((gold & pred).sum()), int(gold.sum()), int(pred.sum())
        p, r = tp / npred if npred else 0.0, tp / ng if ng else 0.0
        result.update({name + "_precision": p, name + "_recall": r,
                       name + "_f1": 2 * p * r / (p + r) if p + r else 0.0})
        combined_tp += tp
        combined_gold += ng
    result["boundary_recall"] = combined_tp / combined_gold if combined_gold else 0.0
    return result


def tokenize_boundary_batch(batch, tokenizer, max_length, stride):
    encoded = encode_windows(tokenizer, batch["tokens"], max_length, stride)
    mapping = encoded.pop("overflow_to_sample_mapping")
    encoded["labels"] = [
        [-100 if word is None else batch["boundary_labels"][sample][word]
         for word in encoded.word_ids(i)]
        for i, sample in enumerate(mapping)
    ]
    return dict(encoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_folder", default="data/bioc/json")
    parser.add_argument("--model_checkpoint", default=DEFAULT_MODEL)
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--output_dir", default="pico_span/boundary_models")
    parser.add_argument("--model_type", choices=["baseline", "multichannel"], default="multichannel")
    parser.add_argument("--loss_type", choices=["ce", "focal"], default=None)
    parser.add_argument("--focal_alpha", type=float, nargs=5, default=list(DEFAULT_ALPHA))
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--epochs", type=float, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--selection_metric",
                        choices=["boundary_recall", "start_recall", "end_recall"],
                        default="boundary_recall")
    args = parser.parse_args()
    args.loss_type = args.loss_type or ("ce" if args.model_type == "baseline" else "focal")
    if args.model_type == "baseline" and args.loss_type != "ce":
        parser.error("The baseline uses CE; focal ablations use --model_type multichannel")
    set_seed(args.seed)  # Seed before randomly initializing the new heads.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_checkpoint, revision=args.model_revision, use_fast=True
    )
    if args.model_type == "baseline":
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_checkpoint, revision=args.model_revision, num_labels=5,
            id2label={i: str(i) for i in range(5)}, label2id={str(i): i for i in range(5)},
        )
    else:
        model = MultiChannelBoundaryModel(
            bert_model_name=args.model_checkpoint, revision=args.model_revision,
            loss_type=args.loss_type, focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
        )
    length = max_sequence_length(tokenizer, model.config, args.max_length)
    files = [Path(args.input_folder) / f"{split}.json" for split in ("train", "validation")]
    data = {}
    for split, path in zip(("train", "validation"), files):
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f"Empty {split} set")
        dataset = Dataset.from_list([
            {"tokens": row["tokens"], "boundary_labels": boundary_labels(row)} for row in rows
        ])
        data[split] = dataset.map(
            lambda batch: tokenize_boundary_batch(batch, tokenizer, length, args.stride),
            batched=True, remove_columns=dataset.column_names,
        )
    name = "boundaries-" + args.model_type + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(args.output_dir) / name
    manifest = experiment_manifest(
        args, files, max_length=length, encoder_config=getattr(
            model.config, "encoder_config", model.config.to_dict()),
        metric_definition="Pooled start/end recall over non-ignored labeled subwords",
    )
    write_json(output / "run_manifest.json", manifest)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output), eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model=args.selection_metric,
            greater_is_better=True, learning_rate=args.learning_rate,
            num_train_epochs=args.epochs, weight_decay=args.weight_decay,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            seed=args.seed, data_seed=args.seed, max_steps=args.max_steps,
            use_cpu=args.use_cpu, report_to=[], save_total_limit=2,
        ),
        train_dataset=data["train"], eval_dataset=data["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics, processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    trainer.save_state()
    write_json(output / "validation_metrics.json", trainer.evaluate())
    print(f"Model saved to {output}")


if __name__ == "__main__":
    main()
