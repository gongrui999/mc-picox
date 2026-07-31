# MC-PICOX

**Multi-Channel PICO Extraction for Overlapping Biomedical Entities**

MC-PICOX is a Python implementation of a two-stage, span-based PICO extraction
pipeline. It extends PICOX with a multi-channel gated boundary detector for
extracting overlapping **Participants (P)**, **Interventions (I)**, and
**Outcomes (O)** from randomized controlled trial text.

The pipeline first localizes type-independent span boundaries and then applies a
multi-label span classifier. Separating localization from entity typing allows
the model to represent overlapping PICO entities.

## Highlights

- End-to-end Python scripts; Jupyter is not required.
- Baseline and multi-channel boundary detectors.
- Point, local-window, and sentence-level feature channels.
- Token-wise softmax gating for adaptive feature fusion.
- Multi-label P/I/O span classification.
- Strict span-level precision, recall, and F1 evaluation.
- Utilities for missed-span and Chinese dataset analysis.

## Architecture

The multi-channel boundary detector uses a pretrained biomedical Transformer
encoder and constructs three views for each token:

1. **Point channel**: the contextual representation of the current token.
2. **Window channel**: local context encoded by a one-dimensional convolution.
3. **Sentence channel**: the `[CLS]` representation broadcast to every token.

A learned token-wise gate combines the three channels without increasing the
final hidden dimension. The fused representation is classified into five
boundary labels: `OUT`, `START`, `END`, `BOTH`, and `IN`.

Predicted start/end positions form candidate spans. A second PubMedBERT model
then performs independent P/I/O classification with sigmoid outputs.

## Repository Structure

```text
.
├── model.py                              # Multi-channel gated boundary model
├── convert_ebm_nlp_to_bioc.py            # Raw EBM-NLP -> BioC XML
├── step_0_preprocess.py                  # BioC XML -> JSONL datasets
├── step_1_1_boundary_training.py         # Train boundary detector
├── step_1_2_boundary_prediction.py       # Predict boundaries
├── step_2_1_span_clf_training.py         # Train P/I/O span classifier
├── step_2_2_span_clf.py                  # End-to-end span extraction
├── step_3_evaluate.py                    # Strict span-level evaluation
├── analyze_missed_spans.py               # Missed-span statistics
├── analyze_short_missed.py               # Short-span error analysis
├── analyze_chinese_dataset.py            # Chinese recommendation analysis
├── requirements.txt
└── README.md
```

The original notebooks are retained as reference material. The `.py` scripts
are recommended for reproducible training and inference.

## Requirements

- Python 3.10 or later
- PyTorch
- Hugging Face Transformers and Datasets
- A CUDA-capable GPU is strongly recommended for PubMedBERT-large

Create an environment:

```bash
conda create -n mc-picox python=3.10 -y
conda activate mc-picox
pip install -r requirements.txt
pip install "accelerate>=1.1.0"
```

For GPU training, install the CUDA-compatible PyTorch build for your system
from the [official PyTorch installation guide](https://pytorch.org/get-started/locally/)
before installing the remaining requirements.

Verify the environment:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Dataset

The official [EBM-NLP repository](https://github.com/bepnye/EBM-NLP) is
included as a Git submodule under `datasets/EBM-NLP`. Clone MC-PICOX with:

```bash
git clone --recurse-submodules https://github.com/gongrui999/mc-picox.git
```

For an existing clone, initialize the dataset submodule with:

```bash
git submodule update --init --recursive
```

Extract `datasets/EBM-NLP/ebm_nlp_2_00.tar.gz` into `data/raw/`. The directory
passed to `--ebm_nlp_dir` must contain the extracted `documents` and
`annotations` folders.

Recommended local layout:

```text
data/
├── raw/
│   └── ebm_nlp_2_00/
└── bioc/
    ├── input/
    └── json/
```

Generated data and model artifacts remain excluded from Git.

## Running the Pipeline

Run all commands from the directory containing the Python scripts.

### 1. Convert EBM-NLP to BioC XML

```bash
python convert_ebm_nlp_to_bioc.py --ebm_nlp_dir data/raw/ebm_nlp_2_00 --output_file data/bioc/input/ebm_nlp_2_00_ssplit.xml
```

### 2. Create Training, Validation, and Test Files

```bash
python step_0_preprocess.py --input_file data/bioc/input/ebm_nlp_2_00_ssplit.xml --raw_dir data/raw/ebm_nlp_2_00 --output_path data/bioc/json --val_ratio 0.05 --seed 0
```

This step creates:

```text
data/bioc/json/
├── train.json
├── validation.json
├── test.json
├── train_span_clf.json
├── validation_span_clf.json
├── test_span_clf.json
├── no_overlap/
└── overlap_only/
```

### 3. Train the Boundary Detector

Multi-channel model:

```bash
python step_1_1_boundary_training.py --input_folder data/bioc/json --model_checkpoint microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract --output_dir pico_span/boundary_models --model_type multichannel --epochs 8 --learning_rate 5e-5 --batch_size 8
```

Original single-head baseline:

```bash
python step_1_1_boundary_training.py --input_folder data/bioc/json --model_checkpoint microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract --output_dir pico_span/boundary_models --model_type baseline --epochs 8 --learning_rate 5e-5 --batch_size 8
```

The trained model is saved under a timestamped directory:

```text
pico_span/boundary_models/boundaries-<model_type>-ebm_nlp-<timestamp>/
```

### 4. Predict Span Boundaries

Replace `<BOUNDARY_MODEL_DIR>` with the directory created in Step 3.

```bash
python step_1_2_boundary_prediction.py --base_model microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract --boundary_model_path <BOUNDARY_MODEL_DIR> --input_folder data/bioc/json --output_path data/bioc/json/step_1_boundary_pred --split test --threshold 0.25
```

Output:

```text
data/bioc/json/step_1_boundary_pred/test_boundary_pred.json
```

### 5. Train the Span Classifier

```bash
python step_2_1_span_clf_training.py --input_folder data/bioc/json --model_checkpoint microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract --output_dir pico_span/span_clf --epochs 3 --learning_rate 2e-5 --batch_size 16
```

The model is saved under:

```text
pico_span/span_clf/span-clf-PICO_NER-ebm_nlp_bioc-<timestamp>/
```

### 6. Extract PICO Spans

Replace `<SPAN_MODEL_DIR>` with the directory created in Step 5.

```bash
python step_2_2_span_clf.py --base_model microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract --span_clf_model_path <SPAN_MODEL_DIR> --input_folder data/bioc/json/step_1_boundary_pred --output_path data/bioc/json/step_2_span_clf --split test --threshold 0.5
```

Output:

```text
data/bioc/json/step_2_span_clf/test_pico_spans.json
```

### 7. Evaluate

```bash
python step_3_evaluate.py --input_folder data/bioc/json/step_2_span_clf --pred_file test_pico_spans.json --confidence_threshold 0.5
```

The evaluator reports TP, FP, FN, precision, recall, and F1 for P, I, O, and
the micro-aggregated overall result. A prediction is correct only when its
entity type, start position, and length exactly match the gold span.

## Output and Model Directories

Training checkpoints and generated predictions can be large. They should
normally be excluded from Git:

```gitignore
data/
pico_span/
*.pt
*.xlsx
__pycache__/
.cache/
```

Do not publish restricted clinical data, private API credentials, or local
model caches.

## Troubleshooting

### CUDA is not detected

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If this prints `False`, reinstall PyTorch using the CUDA build matching your
driver.

### Out-of-memory error

Reduce `--batch_size`, use gradient accumulation, or replace PubMedBERT-large
with a compatible base-size checkpoint. Keep the same checkpoint family for
training and prediction.

### Model loading mismatch

Use the same Transformer checkpoint for boundary training and inference.
Multi-channel checkpoints contain `model.pt` and `model_config.pt`; baseline
models use the Hugging Face `save_pretrained` format.

## Citation

This implementation extends the PICOX framework. If you use it, please cite the
original work:

```bibtex
@article{zhang2024picox,
  title   = {A span-based model for extracting overlapping PICO entities from randomized controlled trial publications},
  author  = {Zhang, G. and Zhou, Y. and Hu, Y. and Xu, H. and Weng, C. and Peng, Y.},
  journal = {Journal of the American Medical Informatics Association},
  volume  = {31},
  number  = {5},
  pages   = {1163--1171},
  year    = {2024},
  doi     = {10.1093/jamia/ocae065}
}
```

Please also cite the MC-PICOX paper after its bibliographic information becomes
available.

## Acknowledgements

MC-PICOX is built on the original PICOX pipeline and the EBM-NLP corpus. We
thank their authors for releasing the code and data resources.
