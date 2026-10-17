# ERNIE-Health integration

The released PaddleNLP ERNIE-Health encoder uses the ELECTRA implementation.
The official [PaddleNLP configuration](https://github.com/PaddlePaddle/PaddleNLP/blob/v2.6.1/paddlenlp/transformers/electra/configuration.py)
and [model implementation](https://github.com/PaddlePaddle/PaddleNLP/blob/v2.6.1/paddlenlp/transformers/electra/modeling.py)
are the references for the adapter.

Do not pass the Paddle model name directly to the PyTorch AutoModel loader.
Use an existing verified HF-compatible checkpoint, or run the converter in a
separate environment containing compatible PaddlePaddle, PaddleNLP, PyTorch,
Transformers and tokenizers packages:

~~~bash
python convert_ernie_health.py --source ernie-health-chinese --output_dir models/ernie-health-hf
~~~

The source may also be a local PaddleNLP encoder directory. The converter loads
the encoder rather than a downstream task head, maps every required ELECTRA
parameter, transposes linear weights, and checks every shape. It preserves the
vocabulary IDs and verifies tokenizer outputs on Chinese/medical examples.
It compares encoder hidden states numerically and refuses to save if parity
exceeds the requested tolerance.

After a successful conversion, point **both** training commands at the saved
directory:

~~~bash
python step_1_1_boundary_training.py --input_folder data/ccg/json --model_checkpoint models/ernie-health-hf --model_type multichannel --loss_type focal --seed 0
python step_2_1_span_clf_training.py --input_folder data/ccg/json --model_checkpoint models/ernie-health-hf --seed 0
~~~

Continue with validation calibration and test evaluation using the trained
checkpoints, as documented in the main README.

The regular regression suite tests mapping, transposition and shape/missing-key
rejection without installing Paddle. A real Paddle/ERNIE-Health conversion,
tokenizer/encoder parity check and full Chinese training were **not** executed
during this repair. Archive the conversion manifest and source package versions
when doing that experiment. The main requirements lock describes the tested
PyTorch pipeline, not an untested Paddle conversion environment.
