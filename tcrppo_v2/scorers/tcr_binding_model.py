"""Fast CDR3β×Peptide binding classifier — sequence-only, no 3D features.

Trained on the tc-hard dataset (566K samples, 1360 peptides) which uses
hard negatives designed to be challenging for sequence-based models.
This provides a much more robust training signal than ERGO (5K McPAS samples)
or the full tFold pipeline (which requires 8s/sample 3D structure prediction).

Architecture:
  CDR3β sequence ──→ [AA Embed + Positional] → BiLSTM → per-residue [H]
  Peptide sequence ──→ [AA Embed + Positional] → BiLSTM → per-residue [H]
  Cross-attention: CDR3β attends to Peptide
  Pool + MLP → sigmoid → binding probability

Speed: ~0.5ms/sample (1000-16000x faster than tFold)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Standard amino acid vocabulary
AA_VOCAB = {
    "<PAD>": 0, "<UNK>": 1,
    "A": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9,
    "K": 10, "L": 11, "M": 12, "N": 13, "P": 14, "Q": 15, "R": 16,
    "S": 17, "T": 18, "V": 19, "W": 20, "Y": 21,
}
N_TOKENS = len(AA_VOCAB)


def encode_sequence(seq: str, max_len: int) -> torch.Tensor:
    """Encode AA sequence to integer tensor, padded to max_len."""
    ids = [AA_VOCAB.get(c, AA_VOCAB["<UNK>"]) for c in seq[:max_len]]
    ids += [AA_VOCAB["<PAD>"]] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class SequenceEncoder(nn.Module):
    """Shared encoder: AA embedding + positional → BiLSTM → per-residue."""

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(N_TOKENS, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(128, embed_dim)  # max 128 positions
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, n_layers,
            batch_first=True, bidirectional=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode token_ids → (per_residue [B, L, H], mask [B, L])."""
        B, L = token_ids.shape
        mask = token_ids != 0  # [B, L]

        pos_ids = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed(token_ids) + self.pos_embed(pos_ids)  # [B, L, E]
        x = self.dropout(x)

        # Pack for LSTM efficiency
        lengths = mask.sum(dim=1).cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False,
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=L)

        out = self.out_proj(out)  # [B, L, H]
        out = self.norm(out)
        return out, mask


class CrossAttentionBlock(nn.Module):
    """Multi-head cross-attention: query attends to key-value."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, q: torch.Tensor, kv: torch.Tensor,
        q_mask: torch.Tensor, kv_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attend q to kv.  Returns [B, Lq, D]."""
        B, Lq, D = q.shape
        Lk = kv.shape[1]
        H, dh = self.n_heads, self.d_head

        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)

        Q = self.W_q(q_n).view(B, Lq, H, dh).permute(0, 2, 1, 3)
        K = self.W_k(kv_n).view(B, Lk, H, dh).permute(0, 2, 1, 3)
        V = self.W_v(kv_n).view(B, Lk, H, dh).permute(0, 2, 1, 3)

        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(dh)

        # Mask out padding positions in keys
        kv_m = kv_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, Lk]
        attn = attn.masked_fill(~kv_m, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).permute(0, 2, 1, 3).reshape(B, Lq, D)
        out = self.W_o(out)
        out = self.dropout(out)

        # Residual + FFN
        h = q + out
        h = h + self.ffn(h)
        # Zero out padding
        h = h * q_mask.unsqueeze(-1).float()
        return h


class TCRBindingModel(nn.Module):
    """Fast CDR3β × Peptide binding classifier.

    Architecture:
      CDR3β → BiLSTM → cross-attend to peptide → pool → MLP → logit
      Peptide → BiLSTM → (used as key/value for cross-attention)
    """

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        n_lstm_layers: int = 2,
        n_attn_layers: int = 2,
        n_heads: int = 4,
        mlp_hidden: int = 256,
        dropout: float = 0.15,
        max_cdr3_len: int = 30,
        max_pep_len: int = 25,
    ):
        super().__init__()
        self.max_cdr3_len = max_cdr3_len
        self.max_pep_len = max_pep_len

        # Shared encoder for both CDR3β and peptide
        self.encoder = SequenceEncoder(embed_dim, hidden_dim, n_lstm_layers, dropout)

        # Cross-attention layers: CDR3β attends to peptide
        self.cross_attn = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_attn_layers)
        ])

        # Pooling → classifier
        pool_dim = hidden_dim * 2  # mean + max pool
        self.classifier = nn.Sequential(
            nn.LayerNorm(pool_dim),
            nn.Linear(pool_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, 1),
        )

        self._count_params()

    def _count_params(self) -> None:
        self.n_params = sum(p.numel() for p in self.parameters())

    def _masked_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean + max pool over valid positions."""
        mask_f = mask.unsqueeze(-1).float()
        x_masked = x * mask_f
        lengths = mask_f.sum(dim=1).clamp(min=1)
        mean_pool = x_masked.sum(dim=1) / lengths
        x_for_max = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        max_pool = x_for_max.max(dim=1)[0]
        max_pool = max_pool.masked_fill(max_pool == float("-inf"), 0.0)
        return torch.cat([mean_pool, max_pool], dim=-1)

    def forward(
        self,
        cdr3b_ids: torch.Tensor,
        pep_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass. Returns binding logits [B]."""
        # Encode sequences
        cdr3b_feats, cdr3b_mask = self.encoder(cdr3b_ids)  # [B, L_b, H]
        pep_feats, pep_mask = self.encoder(pep_ids)  # [B, L_p, H]

        # Cross-attention: CDR3β attends to peptide
        h = cdr3b_feats
        for layer in self.cross_attn:
            h = layer(h, pep_feats, cdr3b_mask, pep_mask)

        # Pool and classify
        pooled = self._masked_pool(h, cdr3b_mask)  # [B, 2H]
        logits = self.classifier(pooled).squeeze(-1)  # [B]
        return logits

    def predict_proba(
        self,
        cdr3b_ids: torch.Tensor,
        pep_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return binding probability in [0, 1]."""
        return torch.sigmoid(self.forward(cdr3b_ids, pep_ids))


def build_model(config: Optional[Dict] = None) -> TCRBindingModel:
    """Build model with default or custom config."""
    defaults = dict(
        embed_dim=64,
        hidden_dim=128,
        n_lstm_layers=2,
        n_attn_layers=2,
        n_heads=4,
        mlp_hidden=256,
        dropout=0.15,
        max_cdr3_len=30,
        max_pep_len=25,
    )
    if config:
        defaults.update(config)
    return TCRBindingModel(**defaults)
