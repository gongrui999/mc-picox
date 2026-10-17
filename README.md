# MC-PICOX

A two-stage span extractor for overlapping Participants, Interventions and Outcomes.
The maintained implementation is the Python pipeline. The original notebooks are
historical reference material.

The boundary model provides point, convolutional-window and CLS channels with
token-wise softmax gating. It supports cross-entropy and class-weighted focal
loss, separate validation-selected start/end thresholds, and a multilabel span
classifier with confidence-based class-wise NMS.

**Implementation status:** the repaired pipeline has offline regression and
end-to-end smoke tests. Full EBM-NLP/CCG-PICO benchmark reruns are still required;
this revision does not claim to have reproduced the paper's reported F1 values.
See [repair status](docs/REVIEW_FIXES.md) and
[experiment protocol](docs/REPRODUCIBILITY.md).

## Installation and checks

The captured environment uses Python 3.13.3, PyTorch 2.10.0 and Transformers 5.3.0.
Both direct dependencies and their captured dependency closure are version-pinned.

For CPU testing:

~~~bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
~~~

For GPU training, install the corresponding PyTorch 2.10.0 CUDA wheel first.
The lock records package versions, not a CUDA driver or hardware specification.
GitHub Actions runs the offline tests on CPU; test models are created locally
and do not download pretrained weights.

## Data and model contracts

Run commands from the repository root. Files named *.json below are JSON Lines,
one sentence/span per line.

Boundary input:

~~~json
{"pmid":"doc-1","sentence_id":"doc-1:0:0","tokens":["Patients","improved"],"labels":[4,1]}
~~~

Token labels use P=4, I=2, O=1 as a bitmask. An optional gold_spans list contains
objects with type, start and end, using token-index half-open intervals [start,end).
Explicit spans take precedence over the bitmask and preserve adjacent or overlapping
same-type gold spans. CCG uses original characters as tokens, preserving whitespace
and character offsets. Whitespace discarded by a tokenizer receives zero boundary
probability and is not mistaken for an uncovered word.

Span-classification input:

~~~json
{"pmid":"doc-1","tokens":["Patients"],"PARTICIPANTS":true,"INTERVENTIONS":false,"OUTCOMES":false}
~~~

Identical gold boundaries are merged into one multilabel example. Crossed
boundaries supply non-entity negatives.

All training defaults use the same public PubMedBERT-base checkpoint. To use a
large model or a Chinese encoder, pass its **actual model ID/revision or local
checkpoint** to both training commands. Encoder size is read from its config
and saved in the run manifest. A base run must not be described as the paper's
large-model experiment.

## EBM-NLP preprocessing

~~~bash
git submodule update --init --recursive
python -m tarfile -e datasets/EBM-NLP/ebm_nlp_2_00.tar.gz data/raw
python convert_ebm_nlp_to_bioc.py --ebm_nlp_dir data/raw/ebm_nlp_2_00 --output_file data/bioc/input/ebm_nlp_2_00_ssplit.xml
python step_0_preprocess.py --input_file data/bioc/input/ebm_nlp_2_00_ssplit.xml --raw_dir data/raw/ebm_nlp_2_00 --output_path data/bioc/json --val_ratio 0.05 --seed 0
~~~

The official gold test PMIDs remain fixed. Non-test PMIDs are sorted and
shuffled using a local seeded RNG. split_manifest.json records all IDs and
source hashes. Reuse that partition with --split_manifest PATH when repeating
an experiment. Newly generated partitions are not assumed to match historical
paper runs.

## Train the two models

Default multichannel training uses focal loss with gamma=2 and weights
(OUT, START, END, BOTH, IN)=(0.15,2,2,6,1):

~~~bash
python step_1_1_boundary_training.py --input_folder data/bioc/json --model_type multichannel --loss_type focal --epochs 8 --learning_rate 5e-5 --batch_size 8 --seed 0
python step_2_1_span_clf_training.py --input_folder data/bioc/json --epochs 3 --learning_rate 2e-5 --batch_size 16 --seed 0
~~~

Use --model_type baseline for the single-head CE baseline.
Use --model_type multichannel --loss_type ce for the MC-only ablation.
Both commands accept --model_checkpoint, --model_revision, --max_length,
--stride and --use_cpu.

Boundary checkpoint selection defaults to boundary_recall: pooled start/end
true positives divided by pooled gold start/end counts over labeled subwords,
using argmax predictions. start_recall and end_recall remain explicit options.
This checkpoint metric is distinct from the word-level, end-to-end validation
objective used to calibrate thresholds.

New multichannel models use the standard HF config.json and model.safetensors
format and reload without downloading their encoder again. The boundary loader
also accepts the original model_config.pt/model.pt format. Tokenizers load from
the saved model directory.

## Calibrate on validation, then evaluate test

Replace BOUNDARY_MODEL_DIR and SPAN_MODEL_DIR with the output directories of
the preceding commands. Use a new output path for each experimental condition.

~~~bash
python step_1_2_boundary_prediction.py --boundary_model_path BOUNDARY_MODEL_DIR --input_folder data/bioc/json --output_path data/bioc/json/boundaries --split validation
python calibrate_thresholds.py --validation_predictions data/bioc/json/boundaries/validation_boundary_pred.json --span_clf_model_path SPAN_MODEL_DIR --output_file data/bioc/json/thresholds.json
python step_1_2_boundary_prediction.py --boundary_model_path BOUNDARY_MODEL_DIR --input_folder data/bioc/json --output_path data/bioc/json/boundaries --split test --threshold_config data/bioc/json/thresholds.json
python step_2_2_span_clf.py --span_clf_model_path SPAN_MODEL_DIR --input_folder data/bioc/json/boundaries --output_path data/bioc/json/predictions --split test --threshold_config data/bioc/json/thresholds.json
python step_3_evaluate.py --input_folder data/bioc/json/predictions --pred_file test_pico_spans.json
~~~

Boundary evidence is q_start=p(START)+p(BOTH) and q_end=p(END)+p(BOTH).
The decision uses q > threshold. Subword/window evidence is combined by max
after summing the relevant classes; probabilities are not added across subwords.

Calibration checks that inputs are validation records, evaluates all 676 pairs
from 0.05 to 0.30 in 0.01 steps, and maximizes strict end-to-end micro F1.
It caches candidate class scores within the run. Ties favor higher recall,
fewer candidates, then higher threshold sum/start/end. The frozen config stores
model checksums, thresholds, tokenization settings and classification/NMS settings.
Test inference checks those model checksums.

Without calibration, use --start_threshold and --end_threshold, or the shared
--threshold convenience option. --decode_rule legacy_or preserves the historical
OR rule for explicitly labeled legacy comparisons; sum is the default.

Class predictions use score >= threshold. NMS operates within each category,
orders by classification confidence and breaks ties by start then longer length.
Exact duplicates are removed. --nms_iou_threshold controls overlap suppression;
1.0 preserves distinct same-type overlaps. Fix this setting across compared
conditions or select it using validation only.

## Long sequences

Training and inference use bounded overlapping tokenizer windows with global
word IDs. The effective maximum comes from the encoder position limit, not an
unbounded tokenizer sentinel. Overlap is clamped below the window content size.
Boundary inference covers the full input, including its tail.

For a span longer than one classifier window, each window receives the span's
multilabel targets during training; inference averages per-window sigmoid scores.
This is an explicit fallback policy, recorded in manifests. It is not evidence
of a measured long-span improvement.

## Reviewed Chinese CCG-PICO annotations

The reviewed structured annotations are in
datasets/CCG-PICO/ccg_pico_recommendations.xlsx. The source workbook is preserved.

~~~bash
python prepare_ccg_pico.py export --output data/ccg/aligned_records.jsonl
~~~

Export maps reviewed fields onto exact character offsets in Final, allowing
whitespace normalization while preserving original coordinates. Repeated or
unmatched fields are listed in the alignment report, not silently dropped or
assigned guessed boundaries. Supply complete-record offsets in an overrides JSON,
then rerun export with --overrides PATH. See the
[CCG format and workflow](datasets/CCG-PICO/README.md).

Once every record has ready offsets:

~~~bash
python prepare_ccg_pico.py build --records data/ccg/aligned_records.jsonl --output data/ccg/json --train_size 541 --validation_size 90 --test_size 90 --seed 0
~~~

The builder preserves normalized template groups and chooses a seeded,
label-balanced split with the requested sizes. It also accepts an existing
verified --split_manifest. A newly generated split is not the missing original
paper partition.

The same train/calibrate/predict/evaluate commands work with --input_folder
data/ccg/json and an actual compatible Chinese checkpoint. For PaddleNLP
ERNIE-Health weights, see [conversion instructions](docs/ERNIE_HEALTH.md).
The converter refuses incomplete mappings or failed tokenizer/encoder parity.

## Three-shot Qwen baseline

The runner uses a user-specified OpenAI-compatible endpoint. It reads the API
key from QWEN_API_KEY; no key belongs in a command, source file or Git commit.
It makes no request during tests or module import.

~~~bash
python run_qwen_baseline.py --train_file data/bioc/json/train.json --evaluation_file data/bioc/json/test.json --example_ids TRAIN_SENTENCE_1 TRAIN_SENTENCE_2 TRAIN_SENTENCE_3 --base_url https://YOUR_PROVIDER/v1 --model Qwen2.5-72B-Instruct --output_file data/qwen/test_predictions.jsonl
python step_3_evaluate.py --pred_file data/qwen/test_predictions.jsonl
~~~

Choose exactly three training sentence IDs. The runner rejects document overlap
between training and evaluation, uses fixed demonstrations and temperature=0,
and saves prompts, raw responses and parsing diagnostics. Repeated returned text
maps to all exact token-aligned occurrences. For CCG use --joiner none.
Provider model revisions and responses must be retained for a reported experiment;
no Qwen benchmark has been run as part of this repair.

## Error analysis

~~~bash
python analyze_missed_spans.py --pred_path data/bioc/json/predictions/test_pico_spans.json --output_path data/analysis/missed.json
python analyze_short_missed.py --missed_path data/analysis/missed.json --train_path data/bioc/json/train.json --pred_path data/bioc/json/predictions/test_pico_spans.json --output_path data/analysis/short.json
python analyze_boundary_errors.py --pred_file data/bioc/json/predictions/test_pico_spans.json --output_file data/analysis/distance.json
python analyze_chinese_dataset.py --input_file datasets/CCG-PICO/ccg_pico_recommendations.xlsx
~~~

Sentence IDs replace the former PMID-plus-length lookup. A prediction checksum
prevents mixing misses with changed predictions. The distance audit explicitly
defines distance as the minimum max(start error,end error); use --mode
boundary_pairs for type-independent candidate-pair auditing.

## Repository artifacts and citation

data/, pico_span/, caches, logs and local credentials are excluded from Git.
The root ZIP/XML/JSON samples are retained for compatibility; they are not
current experimental results. All new runs emit manifests and strict count-based
metrics. Keep full-run artifacts with releases or an experiment archive.

This implementation extends the original PICOX work:
[A span-based model for extracting overlapping PICO entities from randomized
controlled trial publications](https://doi.org/10.1093/jamia/ocae065).
The official corpus is [EBM-NLP](https://github.com/bepnye/EBM-NLP).
