"""Actor-Critic policy with TCR-peptide cross-attention.

Architecture:
  obs = [TCR_emb(1280) | pMHC_emb(1280) | scalars(2)]
  
  TCR pathway:    TCR_emb → Linear → ReLU
  Peptide pathway: pMHC_emb → Linear → ReLU
  Cross-attention: TCR attends to peptide
  Fusion: [TCR_features | attended_pep | pMHC_features] → backbone
  
This forces the network to explicitly model TCR-peptide interaction.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from tcrppo_v2.utils.constants import (
    NUM_OPS, NUM_AMINO_ACIDS, MAX_TCR_LEN,
    OP_SUB, OP_INS, OP_DEL, OP_STOP,
)


class CrossAttentionFusion(nn.Module):
    """Cross-attention: TCR attends to peptide."""
    
    def __init__(self, tcr_dim: int, pep_dim: int, hidden_dim: int, n_heads: int = 4):
        super().__init__()
        self.tcr_dim = tcr_dim
        self.pep_dim = pep_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        
        # Query from TCR, Key/Value from peptide
        self.q_proj = nn.Linear(tcr_dim, hidden_dim)
        self.k_proj = nn.Linear(pep_dim, hidden_dim)
        self.v_proj = nn.Linear(pep_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, tcr_features: torch.Tensor, pep_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tcr_features: [B, tcr_dim]
            pep_features: [B, pep_dim]
        
        Returns:
            attended: [B, hidden_dim] — TCR features attended by peptide
        """
        B = tcr_features.shape[0]
        
        # Project to Q, K, V
        Q = self.q_proj(tcr_features)  # [B, hidden_dim]
        K = self.k_proj(pep_features)  # [B, hidden_dim]
        V = self.v_proj(pep_features)  # [B, hidden_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(B, self.n_heads, self.head_dim)  # [B, n_heads, head_dim]
        K = K.view(B, self.n_heads, self.head_dim)
        V = V.view(B, self.n_heads, self.head_dim)
        
        # Attention scores (self-attention style, but cross between TCR and peptide)
        # For single-token case, we just do scaled dot-product
        attn_scores = (Q * K).sum(dim=-1, keepdim=True) * self.scale  # [B, n_heads, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # Normalize over heads
        
        # Apply attention
        attended = (attn_weights * V).sum(dim=1)  # [B, head_dim] — wait, this is wrong
        # Actually for single-token K/V, we just weight the value
        attended = (attn_weights * V).view(B, self.hidden_dim)  # [B, hidden_dim]
        
        # Output projection
        out = self.out_proj(attended)
        return out


class ActorCriticCrossAttn(nn.Module):
    """Actor-Critic with TCR-peptide cross-attention fusion."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 512,
        max_tcr_len: int = MAX_TCR_LEN,
        use_cross_attn: bool = True,
        n_attn_heads: int = 4,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.max_tcr_len = max_tcr_len
        self.use_cross_attn = use_cross_attn
        
        # Assume obs = [TCR_emb(1280) | pMHC_emb(1280) | scalars(2)]
        self.tcr_dim = 1280
        self.pep_dim = 1280
        self.scalar_dim = obs_dim - self.tcr_dim - self.pep_dim
        
        if use_cross_attn:
            # Separate pathways
            self.tcr_encoder = nn.Sequential(
                nn.Linear(self.tcr_dim, hidden_dim // 2),
                nn.ReLU(),
            )
            self.pep_encoder = nn.Sequential(
                nn.Linear(self.pep_dim, hidden_dim // 2),
                nn.ReLU(),
            )
            
            # Cross-attention
            self.cross_attn = CrossAttentionFusion(
                tcr_dim=hidden_dim // 2,
                pep_dim=hidden_dim // 2,
                hidden_dim=hidden_dim // 2,
                n_heads=n_attn_heads,
            )
            
            # Fusion backbone
            # Input: [TCR_features(256) | attended_pep(256) | pep_features(256) | scalars]
            fusion_input_dim = hidden_dim // 2 + hidden_dim // 2 + hidden_dim // 2 + self.scalar_dim
            self.backbone = nn.Sequential(
                nn.Linear(fusion_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        else:
            # Fallback: simple concatenation (original architecture)
            self.backbone = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        # Head 1: op_type (4-way)
        self.op_head = nn.Linear(hidden_dim, NUM_OPS)

        # Head 2: position (max_tcr_len-way), conditioned on op embedding
        self.op_embed = nn.Embedding(NUM_OPS, 32)
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_tcr_len),
        )

        # Head 3: token (20-way), conditioned on op + position
        self.pos_embed = nn.Embedding(max_tcr_len, 32)
        self.token_head = nn.Sequential(
            nn.Linear(hidden_dim + 32 + 32, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, NUM_AMINO_ACIDS),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        # Policy heads: smaller init for stable exploration
        nn.init.orthogonal_(self.op_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation through cross-attention or simple backbone."""
        if self.use_cross_attn:
            # Split observation
            tcr_emb = obs[:, :self.tcr_dim]
            pep_emb = obs[:, self.tcr_dim:self.tcr_dim + self.pep_dim]
            scalars = obs[:, self.tcr_dim + self.pep_dim:]
            
            # Encode separately
            tcr_features = self.tcr_encoder(tcr_emb)  # [B, 256]
            pep_features = self.pep_encoder(pep_emb)  # [B, 256]
            
            # Cross-attention: TCR attends to peptide
            attended_pep = self.cross_attn(tcr_features, pep_features)  # [B, 256]
            
            # Fuse
            fused = torch.cat([tcr_features, attended_pep, pep_features, scalars], dim=-1)
            features = self.backbone(fused)
        else:
            features = self.backbone(obs)
        
        return features

    def forward(
        self,
        obs: torch.Tensor,
        action_masks: Optional[Dict[str, torch.Tensor]] = None,
        actions: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs: [B, obs_dim] observation tensor.
            action_masks: Dict with 'op_mask' [B, 4] and 'pos_mask' [B, max_tcr_len] bool.
            actions: If provided, (op, pos, tok) tensors for evaluating log-probs.

        Returns:
            (op, pos, tok, value) if actions=None (sampling mode)
            (log_prob, entropy, value, _) if actions provided (evaluation mode)
        """
        features = self._encode_obs(obs)

        if actions is None:
            return self._sample(features, action_masks)
        else:
            return self._evaluate(features, action_masks, actions)

    def _sample(
        self,
        features: torch.Tensor,
        action_masks: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions autoregressively."""
        B = features.shape[0]
        device = features.device

        # Head 1: op_type
        op_logits = self.op_head(features)
        if action_masks is not None and "op_mask" in action_masks:
            op_logits = op_logits.masked_fill(~action_masks["op_mask"], float("-inf"))
        op_dist = Categorical(logits=op_logits, validate_args=False)
        op = op_dist.sample()

        # Head 2: position (conditioned on op)
        op_emb = self.op_embed(op)
        pos_input = torch.cat([features, op_emb], dim=-1)
        pos_logits = self.pos_head(pos_input)
        if action_masks is not None and "pos_mask" in action_masks:
            pos_logits = pos_logits.masked_fill(~action_masks["pos_mask"], float("-inf"))
        pos_dist = Categorical(logits=pos_logits, validate_args=False)
        pos = pos_dist.sample()

        # Head 3: token (conditioned on op + pos, skipped for DEL/STOP)
        pos_emb = self.pos_embed(pos)
        tok_input = torch.cat([features, op_emb, pos_emb], dim=-1)
        tok_logits = self.token_head(tok_input)
        if action_masks is not None and "token_mask" in action_masks:
            batch_idx = torch.arange(B, device=device)
            selected_token_mask = action_masks["token_mask"][batch_idx, pos]
            apply_token_mask = (op == OP_SUB).unsqueeze(-1)
            tok_logits = tok_logits.masked_fill(
                apply_token_mask & ~selected_token_mask,
                float("-inf"),
            )
        tok_dist = Categorical(logits=tok_logits, validate_args=False)
        tok = tok_dist.sample()

        # Value
        value = self.value_head(features).squeeze(-1)

        return op, pos, tok, value

    def _evaluate(
        self,
        features: torch.Tensor,
        action_masks: Optional[Dict[str, torch.Tensor]],
        actions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-probs and entropy for given actions."""
        op, pos, tok = actions

        # Head 1: op_type
        op_logits = self.op_head(features)
        if action_masks is not None and "op_mask" in action_masks:
            op_logits = op_logits.masked_fill(~action_masks["op_mask"], float("-inf"))
        op_dist = Categorical(logits=op_logits, validate_args=False)
        op_log_prob = op_dist.log_prob(op)
        op_entropy = op_dist.entropy()

        # Head 2: position
        op_emb = self.op_embed(op)
        pos_input = torch.cat([features, op_emb], dim=-1)
        pos_logits = self.pos_head(pos_input)
        if action_masks is not None and "pos_mask" in action_masks:
            pos_logits = pos_logits.masked_fill(~action_masks["pos_mask"], float("-inf"))
        pos_dist = Categorical(logits=pos_logits, validate_args=False)
        pos_log_prob = pos_dist.log_prob(pos)
        pos_entropy = pos_dist.entropy()

        # Head 3: token
        pos_emb = self.pos_embed(pos)
        tok_input = torch.cat([features, op_emb, pos_emb], dim=-1)
        tok_logits = self.token_head(tok_input)
        if action_masks is not None and "token_mask" in action_masks:
            batch_idx = torch.arange(features.shape[0], device=features.device)
            selected_token_mask = action_masks["token_mask"][batch_idx, pos]
            apply_token_mask = (op == OP_SUB).unsqueeze(-1)
            tok_logits = tok_logits.masked_fill(
                apply_token_mask & ~selected_token_mask,
                float("-inf"),
            )
        tok_dist = Categorical(logits=tok_logits, validate_args=False)
        tok_log_prob = tok_dist.log_prob(tok)
        tok_entropy = tok_dist.entropy()

        # Mask token log-prob for DEL/STOP (not used)
        is_del_or_stop = (op == OP_DEL) | (op == OP_STOP)
        tok_log_prob = tok_log_prob.masked_fill(is_del_or_stop, 0.0)
        tok_entropy = tok_entropy.masked_fill(is_del_or_stop, 0.0)

        # Total log-prob and entropy
        log_prob = op_log_prob + pos_log_prob + tok_log_prob
        entropy = op_entropy + pos_entropy + tok_entropy

        # Value
        value = self.value_head(features).squeeze(-1)

        return log_prob, entropy, value, None

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate only (for GAE computation)."""
        features = self._encode_obs(obs)
        return self.value_head(features).squeeze(-1)
