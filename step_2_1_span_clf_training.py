"""
Step 2-1: Train a span classifier for PICO element type classification.

Fine-tunes a PubMedBERT model for multi-label sequence classification
(Participants / Interventions / Outcomes) using sigmoid activation.

Usage:
    python step_2_1_span_clf_training.py \
        --input_folder data/bioc/json \
        --model_checkpoint microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract \
        --output_dir pico_span/span_clf \
        --epochs 3 --learning_rate 2e-5
"""

import argparse
import os
from datetime import datetime
from enum import Enum

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PICO_CLASSES = ["PARTICIPANTS", "INTERVENTIONS", "OUTCOMES"]


class PicoType(Enum):
    PARTICIPANTS = 4
    INTERVENTIONS = 2
    OUTCOMES = 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 2-1: Span classifier training")
    parser.add_argument("--input_folder", type=str, default="data/bioc/json")
    parser.add_argument("--model_checkpoint", type=str,
                        default="microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract")
    parser.add_argument("--output_dir", type=str, default="pico_span/span_clf")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train_file", type=str, default=None,
                        help="Override training JSONL filename (default: train_span_clf.json)")
    parser.add_argument("--val_file", type=str, default=None,
                        help="Override validation JSONL filename")
    parser.add_argument("--test_file", type=str, default=None,
                        help="Override test JSONL filename")
    args = parser.parse_args()

    train_file = args.train_file or os.path.join(args.input_folder, "train_span_clf.json")
    val_file = args.val_file or os.path.join(args.input_folder, "validation_span_clf.json")
    test_file = args.test_file or os.path.join(args.input_folder, "test_span_clf.json")

    # ---- Load data ----
    span_clf = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "validation": val_file,
            "test": test_file,
        },
    )
    print(span_clf)

    # ---- Label mapping ----
    id2label = {i: label for i, label in enumerate(PICO_CLASSES)}
    label2id = {v: k for k, v in id2label.items()}

    # ---- Tokenizer & model ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_checkpoint,
        problem_type="multi_label_classification",
        num_labels=len(PICO_CLASSES),
        id2label=id2label,
        label2id=label2id,
    )

    # ---- Preprocessing ----
    def preprocess_function(dataset):
        encoding = tokenizer(
            dataset["tokens"],
            truncation=True,
            is_split_into_words=True,
        )
        labels_batch = {k: dataset[k] for k in dataset.keys() if k in PICO_CLASSES}
        labels_matrix = np.zeros((len(dataset["tokens"]), len(PICO_CLASSES)))
        for idx, label in enumerate(PICO_CLASSES):
            labels_matrix[:, idx] = labels_batch[label]
        encoding["labels"] = labels_matrix.tolist()
        return encoding

    tokenized_dataset = span_clf.map(
        preprocess_function,
        batched=True,
        remove_columns=span_clf["train"].column_names,
    )
    print(tokenized_dataset)

    # ---- Metrics ----
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        probs = torch.nn.functional.sigmoid(torch.tensor(predictions))
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= 0.5)] = 1

        f1 = f1_score(y_true=labels, y_pred=y_pred, average="macro")
        precision = precision_score(y_true=labels, y_pred=y_pred, average="macro")
        recall = recall_score(y_true=labels, y_pred=y_pred, average="macro")

        return {"precision": precision, "recall": recall, "f1": f1}

    # ---- Training ----
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_name = f"span-clf-PICO_NER-ebm_nlp_bioc-{timestamp}"
    run_output_dir = os.path.join(args.output_dir, model_name)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=run_output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    trainer.train()

    save_path = os.path.join(args.output_dir, model_name)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nModel saved to {save_path}")
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")


if __name__ == "__main__":
    main()
