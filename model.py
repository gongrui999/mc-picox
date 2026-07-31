"""
Multi-Channel Gated Fusion Boundary Detection Model for PICO span extraction.

Architecture:
    PubMedBERT Encoder
         │
         H  [batch, seq_len, 768]
         │
    ┌────┼────────────┐
    │    │             │
  Token Window(Conv1d) Sentence([CLS])
  V_point  V_window    V_sentence
    │    │             │
    └────┼────────────┘
         │
    Gated Fusion (softmax-weighted sum)
         │
    H_fusion [batch, seq_len, 768]
         │
    Classifier → logits [batch, seq_len, 5]
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class MultiChannelBoundaryModel(nn.Module):

    def __init__(self, bert_model_name: str, num_labels: int = 5, window_kernel: int = 3):
        super().__init__()
        self.num_labels = num_labels

        # ── Base Encoder ──
        self.bert = AutoModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size  # 768

        # ── Channel 2: Window (Conv1d) ──
        padding = window_kernel // 2  # kernel=3 → padding=1, keeps seq_len unchanged
        self.window_conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=window_kernel,
            padding=padding,
        )

        # ── Gated Fusion ──
        self.gate_linear = nn.Linear(hidden_size * 3, 3)

        # ── Classification Head ──
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # ── Base Encoder ──
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        H = bert_outputs.last_hidden_state  # [batch, seq_len, 768]

        # ── Channel 1: Token (point feature) ──
        V_point = H  # [batch, seq_len, 768]

        # ── Channel 2: Window (local context via Conv1d) ──
        # Conv1d expects [batch, channels, seq_len]
        H_t = H.transpose(1, 2)           # [batch, 768, seq_len]
        V_window = self.window_conv(H_t)   # [batch, 768, seq_len]
        V_window = V_window.transpose(1, 2)  # [batch, seq_len, 768]

        # ── Channel 3: Sentence ([CLS] broadcast) ──
        cls_vec = H[:, 0:1, :]            # [batch, 1, 768]
        V_sentence = cls_vec.expand_as(H)  # [batch, seq_len, 768]

        # ── Gated Fusion ──
        concat = torch.cat([V_point, V_window, V_sentence], dim=-1)  # [batch, seq_len, 768*3]
        G = torch.softmax(self.gate_linear(concat), dim=-1)          # [batch, seq_len, 3]

        g_point = G[:, :, 0].unsqueeze(-1)     # [batch, seq_len, 1]
        g_window = G[:, :, 1].unsqueeze(-1)    # [batch, seq_len, 1]
        g_sentence = G[:, :, 2].unsqueeze(-1)  # [batch, seq_len, 1]

        H_fusion = g_point * V_point + g_window * V_window + g_sentence * V_sentence
        # [batch, seq_len, 768]

        # ── Classification Head ──
        logits = self.classifier(self.dropout(H_fusion))  # [batch, seq_len, 5]

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits, "gate_weights": G}
