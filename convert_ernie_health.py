"""Export a PaddleNLP ERNIE-Health ELECTRA encoder to this PyTorch pipeline.

Run in an environment with PaddleNLP/Paddle installed as well as Transformers.
Every target tensor is required; tokenizer and encoder parity are checked before saving.
"""

import argparse
import re

import numpy as np
import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import ElectraConfig, ElectraModel, PreTrainedTokenizerFast

from pipeline_utils import checkpoint_fingerprint, write_json


def paddle_parameter_name(name):
    if name.startswith("embeddings."):
        return name.replace("embeddings.LayerNorm.", "embeddings.layer_norm."), False
    if name.startswith("embeddings_project."):
        return name, name.endswith(".weight")
    match = re.fullmatch(r"encoder\.layer\.(\d+)\.(.+)", name)
    if not match:
        raise ValueError(f"Unsupported target parameter: {name}")
    index, suffix = match.groups()
    mapping = {
        "attention.self.query": "self_attn.q_proj",
        "attention.self.key": "self_attn.k_proj",
        "attention.self.value": "self_attn.v_proj",
        "attention.output.dense": "self_attn.out_proj",
        "attention.output.LayerNorm": "norm1",
        "intermediate.dense": "linear1",
        "output.dense": "linear2",
        "output.LayerNorm": "norm2",
    }
    for hf_prefix, paddle_prefix in mapping.items():
        if suffix.startswith(hf_prefix + "."):
            tail = suffix[len(hf_prefix):]
            return (f"encoder.layers.{index}.{paddle_prefix}{tail}",
                    tail == ".weight" and "LayerNorm" not in hf_prefix)
    raise ValueError(f"Unsupported target parameter: {name}")


def converted_state(source, target):
    converted = {}
    for name, tensor in target.state_dict().items():
        source_name, transpose = paddle_parameter_name(name)
        aliases = [source_name, "electra." + source_name,
                   "discriminator.electra." + source_name]
        matches = [candidate for candidate in aliases if candidate in source]
        if len(matches) != 1:
            raise ValueError(f"Expected one source tensor for {name}, found {matches}")
        value = source[matches[0]]
        array = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
        array = array.T if transpose else array
        if tuple(array.shape) != tuple(tensor.shape):
            raise ValueError(f"Shape mismatch converting {name}: {array.shape} != {tuple(tensor.shape)}")
        converted[name] = torch.as_tensor(array.copy(), dtype=tensor.dtype)
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="ernie-health-chinese")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tolerance", type=float, default=5e-4)
    args = parser.parse_args()
    # Optional packages are intentionally not imported by the training pipeline.
    try:
        import paddle
        from paddlenlp.transformers import AutoModel as PaddleAutoModel, AutoTokenizer as PaddleAutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Use a separate PaddleNLP environment to run this converter") from exc
    paddle.set_device("cpu")
    source = PaddleAutoModel.from_pretrained(args.source)
    source.eval()
    source_tokenizer = PaddleAutoTokenizer.from_pretrained(args.source)
    config_dict = source.config.to_dict() if hasattr(source.config, "to_dict") else dict(source.config)
    if config_dict.get("model_type") not in (None, "electra"):
        raise ValueError("This adapter requires the ERNIE-Health ELECTRA backbone")
    keys = ("vocab_size", "embedding_size", "hidden_size", "num_hidden_layers",
            "num_attention_heads", "intermediate_size", "hidden_act", "hidden_dropout_prob",
            "attention_probs_dropout_prob", "max_position_embeddings", "type_vocab_size",
            "initializer_range", "pad_token_id", "layer_norm_eps")
    model = ElectraModel(ElectraConfig(**{k: config_dict[k] for k in keys if k in config_dict}))
    model.load_state_dict(converted_state(source.state_dict(), model), strict=True)
    model.eval()
    vocabulary = {source_tokenizer.convert_ids_to_tokens(i): i
                  for i in range(model.config.vocab_size)}
    backend = Tokenizer(models.WordPiece(vocabulary, unk_token=source_tokenizer.unk_token))
    backend.normalizer = normalizers.BertNormalizer(
        lowercase=getattr(source_tokenizer, "do_lower_case", True)
    )
    backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocabulary["[CLS]"]), ("[SEP]", vocabulary["[SEP]"])],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]",
        pad_token="[PAD]", mask_token="[MASK]", model_max_length=model.config.max_position_embeddings,
    )
    maximum_error = 0.0
    for text in ("高血压患者应控制血压。", "SBP≥140 mmHg", "成年人使用药物治疗"):
        source_ids = source_tokenizer(text)["input_ids"]
        if source_ids != tokenizer(text)["input_ids"]:
            raise ValueError("Tokenizer parity failed; check source tokenizer normalization")
        ids = np.array([source_ids], dtype="int64")
        with paddle.no_grad():
            output = source(input_ids=paddle.to_tensor(ids),
                            token_type_ids=paddle.zeros_like(paddle.to_tensor(ids)),
                            attention_mask=paddle.ones_like(paddle.to_tensor(ids)))
        if hasattr(output, "last_hidden_state"):
            output = output.last_hidden_state
        elif isinstance(output, (tuple, list)):
            output = output[0]
        with torch.no_grad():
            expected = model(input_ids=torch.from_numpy(ids),
                             attention_mask=torch.ones_like(torch.from_numpy(ids))).last_hidden_state.numpy()
        error = float(np.max(np.abs(output.numpy() - expected)))
        if not np.isfinite(error) or error > args.tolerance:
            raise ValueError(f"Encoder parity failed: max error {error}")
        maximum_error = max(maximum_error, error)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    write_json(args.output_dir + "/conversion_manifest.json", {
        "source": args.source, "maximum_absolute_error": maximum_error,
        "tolerance": args.tolerance, "checkpoint_sha256": checkpoint_fingerprint(args.output_dir),
        "paddle_version": paddle.__version__, "torch_version": torch.__version__,
    })
    print(f"Saved parity-checked encoder to {args.output_dir}")


if __name__ == "__main__":
    main()
