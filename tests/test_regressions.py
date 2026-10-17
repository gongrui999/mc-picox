"""Offline regression checks for the reviewed failure cases and real HF integration."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import (
    BertConfig, BertForSequenceClassification, BertModel, PreTrainedTokenizerFast,
    Trainer, TrainingArguments, DataCollatorForTokenClassification,
)

from model import MultiChannelBoundaryModel, boundary_loss
from pipeline_utils import (
    boundary_labels, checkpoint_fingerprint, encode_windows, gold_spans, max_sequence_length,
)
from prepare_ccg_pico import build_dataset, grouped_partition, normalized_offsets, template_key, training_row
from pipeline_utils import read_jsonl, write_jsonl
from analyze_boundary_errors import boundary_distance_counts
from run_qwen_baseline import exact_response_spans, demonstrations
from step_0_preprocess import split_pmids, span_classification_examples, PicoType
from step_1_1_boundary_training import compute_metrics, tokenize_boundary_batch
from step_1_2_boundary_prediction import (
    decode_word_scores, extract_boundary_from_prob_dist, predict_word_probabilities,
)
from step_2_2_span_clf import Span, build_prediction_row, nms_span, score_candidates
from step_2_1_span_clf_training import tokenize_span_batch
from step_3_evaluate import evaluate_predictions
from calibrate_thresholds import search_thresholds
from convert_ernie_health import converted_state, paddle_parameter_name

torch.set_num_threads(1)


def tiny_tokenizer():
    vocab = {token: i for i, token in enumerate(
        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "b", "c", "d", "##b"]
    )}
    backend = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    backend.normalizer = normalizers.BertNormalizer(lowercase=True)
    backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]", model_max_length=16,
    )


def tiny_encoder():
    return BertModel(BertConfig(
        vocab_size=10, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=24, max_position_embeddings=16,
    ))


class LossAndModelTests(unittest.TestCase):
    def test_calibration_fingerprint_tracks_tokenizer_changes_and_relocation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first, moved = root / "first", root / "moved"
            first.mkdir()
            moved.mkdir()
            for target in (first, moved):
                (target / "config.json").write_text('{"model_type":"bert"}', encoding="utf-8")
                (target / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
            initial = checkpoint_fingerprint(first)
            self.assertEqual(initial, checkpoint_fingerprint(moved))
            (first / "tokenizer.json").write_text('{"version":"2.0"}', encoding="utf-8")
            self.assertNotEqual(initial, checkpoint_fingerprint(first))

    def test_focal_matches_formula_and_masks_ignored_tokens(self):
        probabilities = torch.tensor([[[.1, .2, .3, .15, .25], [.9, .02, .02, .03, .03]]])
        logits = probabilities.log().requires_grad_()
        labels = torch.tensor([[1, -100]])
        actual = boundary_loss(logits, labels)
        expected = -2.0 * (1 - .2) ** 2 * np.log(.2)
        self.assertAlmostEqual(actual.item(), expected, places=6)
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(logits.grad[0, 1].abs().sum()), 0.0)

    def test_gamma_zero_is_ce_and_all_ignored_is_differentiable(self):
        logits = torch.randn(2, 3, 5, requires_grad=True)
        labels = torch.tensor([[0, 1, -100], [2, 3, 4]])
        self.assertTrue(torch.allclose(
            boundary_loss(logits, labels, alpha=[1] * 5, gamma=0),
            boundary_loss(logits, labels, loss_type="ce"),
        ))
        zero = boundary_loss(logits, torch.full_like(labels, -100))
        zero.backward()
        self.assertEqual(zero.item(), 0.0)

    def test_encoder_not_reinitialized_and_checkpoint_roundtrip(self):
        encoder = tiny_encoder()
        before = encoder.embeddings.word_embeddings.weight.detach().clone()
        model = MultiChannelBoundaryModel(encoder=encoder).eval()
        self.assertTrue(torch.equal(before, model.bert.embeddings.word_embeddings.weight))
        x = torch.tensor([[2, 5, 6, 3]])
        with torch.no_grad():
            reference = model(input_ids=x).logits
        self.assertEqual(set(model(input_ids=x)), {"logits"})
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            loaded = MultiChannelBoundaryModel.from_pretrained(folder).eval()
            with torch.no_grad():
                self.assertTrue(torch.allclose(reference, loaded(input_ids=x).logits))

    def test_real_trainer_trains_evaluates_saves_and_restores(self):
        tokenizer = tiny_tokenizer()
        model = MultiChannelBoundaryModel(encoder=tiny_encoder())
        encoded = tokenizer(["a", "b"], is_split_into_words=True)
        sample = dict(encoded, labels=[-100, 1, 2, -100])
        with tempfile.TemporaryDirectory() as folder:
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=folder, use_cpu=True, report_to=[], disable_tqdm=True,
                    num_train_epochs=1, per_device_train_batch_size=1,
                    per_device_eval_batch_size=1, eval_strategy="epoch", save_strategy="epoch",
                    load_best_model_at_end=True, metric_for_best_model="boundary_recall",
                    greater_is_better=True,
                ),
                train_dataset=[sample], eval_dataset=[sample],
                data_collator=DataCollatorForTokenClassification(tokenizer),
                compute_metrics=compute_metrics,
            )
            trainer.train()
            metrics = trainer.evaluate()
            self.assertIn("eval_boundary_recall", metrics)
            self.assertIsNotNone(trainer.state.best_model_checkpoint)
            result = trainer.predict([sample])
            self.assertEqual(result.predictions.shape[-1], 5)
            self.assertFalse(isinstance(result.predictions, tuple))


class BoundaryAndWindowTests(unittest.TestCase):
    def test_sum_boundary_differs_from_legacy_or(self):
        p = [.60, .14, .08, .16, .02]
        self.assertEqual(extract_boundary_from_prob_dist(p, .25), 1)
        self.assertEqual(extract_boundary_from_prob_dist(p, .25, decode_rule="legacy_or"), 0)
        self.assertEqual(decode_word_scores([.25], [.25], .25, .25), [0])

    def test_unequal_start_end_thresholds(self):
        self.assertEqual(decode_word_scores([.2], [.2], .1, .3), [1])
        self.assertEqual(decode_word_scores([.2], [.2], .3, .1), [2])

    def test_windowed_prediction_covers_tail(self):
        model = MultiChannelBoundaryModel(encoder=tiny_encoder()).eval()
        start, end = predict_word_probabilities(["a"] * 40, model, tiny_tokenizer(), stride=4)
        self.assertEqual(len(start), 40)
        self.assertTrue(all(0 < p <= 1 for p in start + end))

    def test_character_tokens_preserve_spaces_and_original_offsets(self):
        model = MultiChannelBoundaryModel(encoder=tiny_encoder()).eval()
        start, end = predict_word_probabilities(list("a b"), model, tiny_tokenizer())
        self.assertEqual(len(start), 3)
        self.assertEqual(start[1], 0)
        self.assertGreater(start[2], 0)
        self.assertGreater(end[2], 0)

    def test_training_windows_keep_gold_tail(self):
        batch = {"tokens": [["a"] * 40], "boundary_labels": [[0] * 39 + [3]]}
        encoded = tokenize_boundary_batch(batch, tiny_tokenizer(), 16, 4)
        self.assertGreater(len(encoded["input_ids"]), 1)
        self.assertTrue(any(3 in labels for labels in encoded["labels"]))
        self.assertTrue(all(len(ids) <= 16 for ids in encoded["input_ids"]))

    def test_span_training_window_mapping_and_long_span_prediction(self):
        batch = {"tokens": [["a"] * 40], "PARTICIPANTS": [True],
                 "INTERVENTIONS": [False], "OUTCOMES": [True]}
        encoded = tokenize_span_batch(batch, tiny_tokenizer(), 16, 4)
        self.assertGreater(len(encoded["labels"]), 1)
        self.assertTrue(all(label == [1., 0., 1.] for label in encoded["labels"]))
        config = tiny_encoder().config
        config.num_labels = 3
        config.id2label = dict(enumerate(("PARTICIPANTS", "INTERVENTIONS", "OUTCOMES")))
        model = BertForSequenceClassification(config).eval()
        scored = score_candidates(["a"] * 40, [(0, 39)], tiny_tokenizer(), model, "cpu")
        self.assertEqual(set(scored[(0, 39)]), set(config.id2label.values()))

    def test_model_limit_overrides_tokenizer_sentinel(self):
        tokenizer = tiny_tokenizer()
        tokenizer.model_max_length = 10 ** 30
        self.assertEqual(max_sequence_length(tokenizer, tiny_encoder().config), 16)
        with self.assertRaises(ValueError):
            max_sequence_length(tokenizer, tiny_encoder().config, 17)

    def test_combined_recall_counts_start_and_end(self):
        logits = np.zeros((1, 2, 5))
        logits[0, :, 1] = 1
        result = compute_metrics((logits, np.array([[1, 2]])))
        self.assertEqual(result["start_recall"], 1)
        self.assertEqual(result["end_recall"], 0)
        self.assertEqual(result["boundary_recall"], .5)


class NmsAndEvaluationTests(unittest.TestCase):
    def test_boundary_distance_uses_both_endpoints(self):
        rows = [{"tokens": ["a"] * 10, "gold_spans": [
            {"type": "OUTCOMES", "start": 1, "end": 4},
            {"type": "PARTICIPANTS", "start": 1, "end": 4}],
            "pico_elements": {"OUTCOMES": [
                {"span_start": 2, "span_length": 3, "confidence": .8}]}}]
        self.assertEqual(boundary_distance_counts(rows),
                         {"exact": 0, "offset_1": 1, "offset_2": 0, "beyond_2_or_missing": 1})

    def test_nms_keeps_high_confidence_long_span(self):
        result = nms_span([Span(0, 5), Span(1, 2)], ["OUTCOMES"] * 2, [.95, .6])
        self.assertEqual(result[0], [Span(0, 5)])

    def test_discarded_candidate_does_not_suppress_other_spans(self):
        result = nms_span([Span(0, 4), Span(0, 1), Span(3, 5)],
                          ["OUTCOMES"] * 3, [.6, .9, .8])
        self.assertEqual(set(result[0]), {Span(0, 1), Span(3, 5)})

    def test_cross_type_overlap_and_duplicate_removal(self):
        result = nms_span([Span(0, 2)] * 3, ["PARTICIPANTS", "OUTCOMES", "OUTCOMES"],
                          [.8, .7, .9])
        self.assertEqual(len(result[0]), 2)
        self.assertEqual(set(result[1]), {"PARTICIPANTS", "OUTCOMES"})

    def test_confidence_arrays_survive_empty_and_nonempty_predictions(self):
        row = {"tokens": ["a", "b"], "pmid": "1",
               "start_confidence": [.1, .2], "end_confidence": [.3, .4]}
        empty = build_prediction_row(row, [], [], [])
        actual = build_prediction_row(row, [Span(0, 2)], ["OUTCOMES"], [.9])
        self.assertEqual(empty["start_confidence"], [.1, .2])
        self.assertEqual(actual["end_confidence"], [.3, .4])
        self.assertEqual(actual["pico_elements"]["OUTCOMES"][0]["end_confidence"], .4)

    def test_explicit_same_type_spans_and_threshold_equality(self):
        row = {"tokens": ["a", "b", "c"], "labels": [4, 4, 4],
               "gold_spans": [{"type": "PARTICIPANTS", "start": 0, "end": 1},
                              {"type": "PARTICIPANTS", "start": 1, "end": 3}],
               "pico_elements": {"PARTICIPANTS": [
                   {"span_start": 0, "span_length": 1, "confidence": .5},
                   {"span_start": 1, "span_length": 2, "confidence": .9},
                   {"span_start": 1, "span_length": 2, "confidence": .9},
               ]}}
        metrics = evaluate_predictions([row])
        self.assertEqual(metrics["micro"]["tp"], 2)
        self.assertEqual(metrics["micro"]["fp"], 0)
        self.assertEqual(boundary_labels(row), [3, 1, 2])

    def test_full_676_threshold_grid(self):
        rows = [{"tokens": ["a", "b"], "labels": [4, 4],
                 "start_confidence": [.3, .0], "end_confidence": [.0, .2]}]
        cached = [{(0, 1): {"PARTICIPANTS": .9, "INTERVENTIONS": .1, "OUTCOMES": .1}}]
        best, grid = search_thresholds(rows, cached)
        self.assertEqual(len(grid), 676)
        self.assertEqual(best["f1"], 1)
        self.assertLess(best["start_threshold"], .3)
        self.assertLess(best["end_threshold"], .2)


class DataAndBaselineTests(unittest.TestCase):
    def test_ccg_build_emits_explicit_spans_and_rejects_unresolved_offsets(self):
        records = [{"record_id": str(i), "text": chr(65 + i // 2) + str(i),
                    "template_id": str(i // 2), "offset_status": "ready", "unresolved": [],
                    "spans": [{"type": "PARTICIPANTS", "start": 0, "end": 1}]}
                   for i in range(12)]
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "records.jsonl"
            output = Path(folder) / "data"
            write_jsonl(source, records)
            manifest = build_dataset(source, output, 8, 2, 2, seed=4)
            self.assertEqual(len(manifest["splits"]["train"]), 8)
            self.assertEqual(len(read_jsonl(output / "train.json")), 8)
            self.assertIn("gold_spans", read_jsonl(output / "test.json")[0])
            records[0]["unresolved"] = [{"field": "Disease"}]
            write_jsonl(source, records)
            with self.assertRaisesRegex(ValueError, "offsets"):
                build_dataset(source, output, 8, 2, 2, seed=4)
            records[0]["unresolved"] = []
            records[0]["spans"][0]["end"] = 1000
            write_jsonl(source, records)
            with self.assertRaisesRegex(ValueError, "offsets"):
                build_dataset(source, output, 8, 2, 2, seed=4)

    def test_paddle_electra_parameter_mapping_roundtrip(self):
        class Target:
            def state_dict(self):
                return {"encoder.layer.0.attention.self.query.weight": torch.empty(3, 2),
                        "embeddings.LayerNorm.weight": torch.empty(3)}
        target = Target()
        source = {
            "electra.encoder.layers.0.self_attn.q_proj.weight": np.array([[1., 2., 3.], [4., 5., 6.]]),
            "electra.embeddings.layer_norm.weight": np.array([.7, .8, .9]),
        }
        restored = converted_state(source, target)
        self.assertEqual(set(restored), set(target.state_dict()))
        self.assertTrue(torch.equal(restored["encoder.layer.0.attention.self.query.weight"],
                                    torch.tensor([[1., 4.], [2., 5.], [3., 6.]])))
        self.assertTrue(torch.allclose(restored["embeddings.LayerNorm.weight"],
                                       torch.tensor([.7, .8, .9])))
        source.pop(next(iter(source)))
        with self.assertRaises(ValueError):
            converted_state(source, target)

    def test_identical_gold_boundaries_have_joint_labels(self):
        from step_0_preprocess import Span as SourceSpan
        by_type = {PicoType.PARTICIPANTS: [SourceSpan(0, 2)],
                   PicoType.INTERVENTIONS: [], PicoType.OUTCOMES: [SourceSpan(0, 2)]}
        examples = span_classification_examples(["a", "b"], by_type, "1")
        self.assertEqual(len(examples), 1)
        self.assertTrue(examples[0]["PARTICIPANTS"])
        self.assertTrue(examples[0]["OUTCOMES"])

    def test_partition_is_independent_of_python_hash_seed(self):
        code = ("from step_0_preprocess import split_pmids; import json; "
                "print(json.dumps([sorted(s) for s in split_pmids("
                "{str(i) for i in range(50)}, {'0'}, .2, 7)]))")
        outputs = []
        for seed in ("1", "2"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            outputs.append(subprocess.check_output([sys.executable, "-c", code],
                                                   env=env, text=True))
        self.assertEqual(outputs[0], outputs[1])
        train, validation, test = split_pmids(set(map(str, range(50))), {"0"}, .2, 7)
        self.assertFalse(train & validation or train & test or validation & test)

    def test_offset_normalization_maps_to_original_text(self):
        self.assertEqual(normalized_offsets("SBP 140 mmHg", "SBP140mmHg"), [(0, 12)])
        self.assertEqual(normalized_offsets("a a", "a"), [(0, 1), (2, 3)])

    def test_group_split_preserves_templates_and_exact_sizes(self):
        records = [{"record_id": str(i), "text": chr(65 + i // 2) + str(i),
                    "template_id": str(i // 2),
                    "spans": [{"type": "PARTICIPANTS", "start": 0, "end": 1}]}
                   for i in range(12)]
        a = grouped_partition(records, 8, 2, 2, 4)
        b = grouped_partition(list(reversed(records)), 8, 2, 2, 4)
        self.assertEqual({k: len(v) for k, v in a.items()},
                         {"train": 8, "validation": 2, "test": 2})
        self.assertEqual({k: {r["record_id"] for r in v} for k, v in a.items()},
                         {k: {r["record_id"] for r in v} for k, v in b.items()})
        groups = [{template_key(r) for r in rs} for rs in a.values()]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

    def test_qwen_parser_preserves_repeated_exact_mentions(self):
        found, unmatched = exact_response_spans(
            '[{"text":"a","type":"P"},{"text":"missing","type":"O"}]', ["a", "b", "a"]
        )
        self.assertEqual(found, {("PARTICIPANTS", 0, 1), ("PARTICIPANTS", 2, 1)})
        self.assertEqual(len(unmatched), 1)

    def test_three_shot_demonstrations_cannot_use_test_documents(self):
        train = [{"pmid": str(i), "sentence_id": str(i)} for i in range(3)]
        with self.assertRaises(ValueError):
            demonstrations(train, ["0", "1", "2"], [{"pmid": "0"}])


if __name__ == "__main__":
    unittest.main()
