"""Actor-Critic policy with 3-head autoregressive action space.

Heads:
  1. op_type: {SUB=0, INS=1, DEL=2, STOP=3}  — 4-way categorical
  2. position: [0, max_tcr_len-1]               — max_tcr_len-way categorical
  3. token: [0, 19]                              — 20-way categorical (skipped for DEL/STOP)

Autoregressive: head2 conditioned on head1, head3 conditioned on head1+head2.
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


class ActorCritic(nn.Module):
    """Actor-Critic with 3-head autoregressive action space and action masking."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 512,
        max_tcr_len: int = MAX_TCR_LEN,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.max_tcr_len = max_tcr_len

        # Shared backbone
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
        features = self.backbone(obs)

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
        op_dist = Categorical(logits=op_logits)
        op = op_dist.sample()

        # Head 2: position (conditioned on op)
        op_emb = self.op_embed(op)
        pos_input = torch.cat([features, op_emb], dim=-1)
        pos_logits = self.pos_head(pos_input)
        if action_masks is not None and "pos_mask" in action_masks:
            pos_logits = pos_logits.masked_fill(~action_masks["pos_mask"], float("-inf"))
        pos_dist = Categorical(logits=pos_logits)
        pos = pos_dist.sample()

        # Head 3: token (conditioned on op + pos, skipped for DEL/STOP)
        pos_emb = self.pos_embed(pos)
        tok_input = torch.cat([features, op_emb, pos_emb], dim=-1)
        tok_logits = self.token_head(tok_input)
        tok_dist = Categorical(logits=tok_logits)
        tok = tok_dist.sample()

        # Value
        value = self.value_head(features).squeeze(-1)

        return op, pos, tok, value

    def _evaluate(
        self,
        features: torch.Tensor,
        action_masks: Optional[Dict[str, torch.Tensor]],
        actions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Evaluate log-probs and entropy for given actions."""
        op_actions, pos_actions, tok_actions = actions
        device = features.device

        # Head 1: op
        op_logits = self.op_head(features)
        if action_masks is not None and "op_mask" in action_masks:
            op_logits = op_logits.masked_fill(~action_masks["op_mask"], float("-inf"))
        op_dist = Categorical(logits=op_logits)
        op_log_prob = op_dist.log_prob(op_actions)
        op_entropy = op_dist.entropy()

        # Head 2: position
        op_emb = self.op_embed(op_actions)
        pos_input = torch.cat([features, op_emb], dim=-1)
        pos_logits = self.pos_head(pos_input)
        if action_masks is not None and "pos_mask" in action_masks:
            pos_logits = pos_logits.masked_fill(~action_masks["pos_mask"], float("-inf"))
        pos_dist = Categorical(logits=pos_logits)
        pos_log_prob = pos_dist.log_prob(pos_actions)
        pos_entropy = pos_dist.entropy()

        # Head 3: token
        pos_emb = self.pos_embed(pos_actions)
        tok_input = torch.cat([features, op_emb, pos_emb], dim=-1)
        tok_logits = self.token_head(tok_input)
        tok_dist = Categorical(logits=tok_logits)
        tok_log_prob = tok_dist.log_prob(tok_actions)
        tok_entropy = tok_dist.entropy()

        # Mask token log-prob for DEL/STOP (token not sampled)
        needs_token = (op_actions == OP_SUB) | (op_actions == OP_INS)
        tok_log_prob = tok_log_prob * needs_token.float()
        tok_entropy = tok_entropy * needs_token.float()

        # Mask position log-prob for STOP (position is meaningless)
        needs_position = (op_actions != OP_STOP)
        pos_log_prob = pos_log_prob * needs_position.float()
        pos_entropy = pos_entropy * needs_position.float()

        # Total log-prob = sum of all heads
        total_log_prob = op_log_prob + pos_log_prob + tok_log_prob
        total_entropy = op_entropy + pos_entropy + tok_entropy

        # Value
        value = self.value_head(features).squeeze(-1)

        return total_log_prob, total_entropy, value, None

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate only."""
        features = self.backbone(obs)
        return self.value_head(features).squeeze(-1)
