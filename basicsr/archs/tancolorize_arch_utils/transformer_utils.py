import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.normalize_before = normalize_before

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        if query_pos is not None:
            tgt = tgt + query_pos
        if self.normalize_before:
            tgt = self.norm(tgt)
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        if not self.normalize_before:
            tgt = self.norm(tgt)
        return tgt


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.normalize_before = normalize_before

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        if query_pos is not None:
            tgt = tgt + query_pos
        if self.normalize_before:
            tgt = self.norm(tgt)
        if pos is not None:
            memory = memory + pos
        tgt2, _ = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        if not self.normalize_before:
            tgt = self.norm(tgt)
        return tgt


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.normalize_before = normalize_before

    def forward(self, tgt):
        if self.normalize_before:
            tgt = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        if not self.normalize_before:
            tgt = self.norm(tgt)
        return tgt


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# Re-export Transformer from transformer.py for convenience
from .transformer import Transformer
