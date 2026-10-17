"""Hugging Face compatible gated boundary detector and masked focal loss."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

DEFAULT_ALPHA = (0.15, 2.0, 2.0, 6.0, 1.0)  # OUT, START, END, BOTH, IN


def boundary_loss(logits, labels, loss_type="focal", alpha=DEFAULT_ALPHA, gamma=2.0):
    """Mean over labeled subwords; ignored positions do not change the denominator."""
    valid = labels.ne(-100)
    if not valid.any():
        return logits.sum() * 0.0
    selected, targets = logits[valid].float(), labels[valid]
    if loss_type == "ce":
        return F.cross_entropy(selected, targets)
    if loss_type != "focal":
        raise ValueError("loss_type must be ce or focal")
    if gamma < 0 or len(alpha) != logits.shape[-1] or any(a < 0 for a in alpha):
        raise ValueError("Invalid focal gamma or class weights")
    log_pt = F.log_softmax(selected, dim=-1).gather(1, targets[:, None]).squeeze(1)
    weights = selected.new_tensor(alpha)[targets]
    return (-weights * (1.0 - log_pt.exp()).pow(gamma) * log_pt).mean()


class MultiChannelConfig(PretrainedConfig):
    model_type = "mc_picox_boundary"
    keys_to_ignore_at_inference = ["gate_weights"]

    def __init__(self, encoder_config=None, window_kernel=3, boundary_loss_type="focal",
                 focal_alpha=None, focal_gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder_config = encoder_config or {}
        self.window_kernel = window_kernel
        self.boundary_loss_type = boundary_loss_type
        self.focal_alpha = list(DEFAULT_ALPHA if focal_alpha is None else focal_alpha)
        self.focal_gamma = focal_gamma


class MultiChannelBoundaryModel(PreTrainedModel):
    """One encoder, point/window/CLS channels, and token-wise softmax gating."""

    config_class = MultiChannelConfig
    base_model_prefix = "bert"

    def __init__(self, config=None, bert_model_name=None, num_labels=5, window_kernel=3,
                 loss_type="focal", focal_alpha=DEFAULT_ALPHA, focal_gamma=2.0,
                 revision=None, encoder=None):
        if isinstance(config, str):
            bert_model_name, config = config, None
        if config is None:
            if encoder is None:
                if not bert_model_name:
                    raise ValueError("Provide an encoder config or bert_model_name")
                encoder = AutoModel.from_pretrained(bert_model_name, revision=revision)
            config = MultiChannelConfig(
                encoder_config=encoder.config.to_dict(), num_labels=num_labels,
                window_kernel=window_kernel, boundary_loss_type=loss_type,
                focal_alpha=focal_alpha, focal_gamma=focal_gamma,
            )
        if config.window_kernel < 1 or config.window_kernel % 2 != 1:
            raise ValueError("window_kernel must be a positive odd integer")
        if not config.encoder_config:
            raise ValueError("encoder_config is required to reload the model offline")
        super().__init__(config)
        if encoder is None:
            encoder_dict = dict(config.encoder_config)
            encoder_type = encoder_dict.pop("model_type")
            encoder = AutoModel.from_config(AutoConfig.for_model(encoder_type, **encoder_dict))
        self.bert = encoder
        self.num_labels = config.num_labels
        hidden = self.bert.config.hidden_size
        self.window_conv = nn.Conv1d(
            hidden, hidden, config.window_kernel, padding=config.window_kernel // 2
        )
        self.gate_linear = nn.Linear(hidden * 3, 3)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, self.num_labels)
        # HF skips encoder modules already initialized by AutoModel; this also
        # installs the metadata required for safe save/load on Transformers 5.
        self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None,
                output_gate_weights=False):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
        hidden = self.bert(**inputs).last_hidden_state
        masked = hidden if attention_mask is None else hidden * attention_mask.unsqueeze(-1)
        window = self.window_conv(masked.transpose(1, 2)).transpose(1, 2)
        sentence = hidden[:, :1].expand_as(hidden)
        gate = torch.softmax(self.gate_linear(torch.cat((hidden, window, sentence), -1)), -1)
        fused = sum(gate[..., i:i + 1] * channel
                    for i, channel in enumerate((hidden, window, sentence)))
        logits = self.classifier(self.dropout(fused))
        loss = None if labels is None else boundary_loss(
            logits, labels, self.config.boundary_loss_type,
            self.config.focal_alpha, self.config.focal_gamma,
        )
        if output_gate_weights:
            result = {"logits": logits, "gate_weights": gate}
            if loss is not None:
                result["loss"] = loss
            return result
        return TokenClassifierOutput(loss=loss, logits=logits)
