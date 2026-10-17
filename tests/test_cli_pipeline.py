"""Exercise the public pipeline commands on tiny local models, without network/GPU."""

import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from pipeline_utils import read_jsonl, write_jsonl
from test_regressions import tiny_encoder, tiny_tokenizer


def invoke(module, *arguments):
    with patch.object(sys, "argv", [module + ".py", *map(str, arguments)]):
        importlib.import_module(module).main()


class OfflineCliPipelineTests(unittest.TestCase):
    def test_train_calibrate_predict_and_evaluate(self):
        with tempfile.TemporaryDirectory(prefix="picox-cli-") as tmp:
            root = Path(tmp)
            encoder = root / "encoder"
            tiny_encoder().save_pretrained(encoder)
            tiny_tokenizer().save_pretrained(encoder)
            data = root / "data"
            for split in ("train", "validation", "test"):
                write_jsonl(data / f"{split}.json", [
                    {"pmid": split + "1", "sentence_id": split + ":1",
                     "tokens": ["a", "b", "c"], "labels": [4, 4, 0]},
                    {"pmid": split + "2", "sentence_id": split + ":2",
                     "tokens": ["b", "c"], "labels": [2, 1]},
                ])
                write_jsonl(data / f"{split}_span_clf.json", [
                    {"pmid": split + "1", "tokens": ["a", "b"],
                     "PARTICIPANTS": True, "INTERVENTIONS": False, "OUTCOMES": False},
                    {"pmid": split + "2", "tokens": ["b"],
                     "PARTICIPANTS": False, "INTERVENTIONS": True, "OUTCOMES": True},
                ])
            for kind in ("baseline", "multichannel"):
                invoke("step_1_1_boundary_training",
                       "--input_folder", data, "--model_checkpoint", encoder,
                       "--output_dir", root / kind, "--model_type", kind,
                       "--epochs", "1", "--batch_size", "2", "--use_cpu")
                checkpoint = next((root / kind).iterdir())
                self.assertTrue((checkpoint / "run_manifest.json").is_file())
                invoke("step_1_2_boundary_prediction",
                       "--boundary_model_path", checkpoint, "--input_folder", data,
                       "--output_path", root / (kind + "-predictions"),
                       "--split", "validation", "--device", "cpu")
            boundary = next((root / "multichannel").iterdir())
            probability_folder = root / "multichannel-predictions"
            invoke("step_2_1_span_clf_training",
                   "--input_folder", data, "--model_checkpoint", encoder,
                   "--output_dir", root / "span", "--epochs", "1",
                   "--batch_size", "2", "--use_cpu")
            span = next((root / "span").iterdir())
            threshold_file = root / "thresholds.json"
            invoke("calibrate_thresholds",
                   "--validation_predictions", probability_folder / "validation_boundary_pred.json",
                   "--span_clf_model_path", span, "--output_file", threshold_file,
                   "--classification_threshold", "0", "--device", "cpu")
            selected = json.loads(threshold_file.read_text(encoding="utf-8"))
            self.assertEqual(selected["grid_size"], 676)
            self.assertEqual(selected["split"], "validation")
            invoke("step_1_2_boundary_prediction",
                   "--boundary_model_path", boundary, "--input_folder", data,
                   "--output_path", probability_folder, "--split", "test",
                   "--threshold_config", threshold_file, "--device", "cpu")
            invoke("step_2_2_span_clf",
                   "--span_clf_model_path", span, "--input_folder", probability_folder,
                   "--output_path", root / "predictions", "--split", "test",
                   "--threshold_config", threshold_file, "--device", "cpu")
            metrics_file = root / "metrics.json"
            invoke("step_3_evaluate", "--input_folder", root / "predictions",
                   "--metrics_file", metrics_file)
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            self.assertEqual(metrics["records"], 2)
            self.assertEqual(metrics["micro"]["tp"] + metrics["micro"]["fn"], 3)
            self.assertIn("macro", metrics)
            predictions = read_jsonl(root / "predictions/test_pico_spans.json")
            predicted_count = sum(len(spans) for row in predictions
                                  for spans in row["pico_elements"].values())
            self.assertEqual(metrics["micro"]["tp"] + metrics["micro"]["fp"], predicted_count)
            self.assertEqual(metrics["provenance"]["arguments"]["confidence_threshold"], 0)
            # Genuine leakage guard: reject test output even if its filename is supplied.
            with self.assertRaisesRegex(ValueError, "validation"):
                invoke("calibrate_thresholds",
                       "--validation_predictions", probability_folder / "test_boundary_pred.json",
                       "--span_clf_model_path", span, "--output_file", root / "bad.json")


if __name__ == "__main__":
    unittest.main()
