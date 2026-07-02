"""RSMH: Residual State-Space Markov Head.

Pro-designed (2026-07-02, plan/gpt55pro-sequential-head-design-2026-07-02.md).

Keeps a 256-d recurrent state in the same latent space as Markov embeddings.
At initialization all new matrices are zero, so the head is EXACTLY VanillaMarkov.
The recurrent state is a RESIDUAL correction that learns to propagate branch
information across positions within a 7-token draft block.

Key difference from the previous RNN attempt:
  - Warm-starts from trained Markov (not from scratch)
  - Zero-init means exact Markov at step 0
  - State exists at init (EMA of prev embeddings) but δ_k = 0 → no effect on logits
  - Only ~2.4M new trainable parameters (W1/W2 frozen)
"""

from typing import Optional

import torch
from torch import nn

from deepspec.modeling.dspark.markov_head import VanillaMarkov
from deepspec.utils.sampling import sample_tokens


class RSMHHead(VanillaMarkov):
    """Residual State-Space Markov Head."""

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        hidden_size: int,
        num_positions: int = 7,
    ):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "rsmh"
        self.hidden_size = hidden_size
        r = markov_rank
        d = hidden_size

        input_dim = 2 * r + d

        self.update_proj = nn.Linear(input_dim, r, bias=True)
        self.gate_proj = nn.Linear(input_dim, r, bias=True)
        self.residual_proj = nn.Linear(input_dim, r, bias=True)
        self.pos_residual = nn.Parameter(torch.zeros(num_positions, r))

        self._zero_init()

    def _zero_init(self):
        for proj in [self.update_proj, self.gate_proj, self.residual_proj]:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def _step(
        self,
        e_k: torch.Tensor,
        s_prev: torch.Tensor,
        h_k: torch.Tensor,
        pos_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([e_k, s_prev, h_k], dim=-1)

        u_k = e_k + self.update_proj(z)
        g_k = torch.sigmoid(self.gate_proj(z))
        s_k = g_k * s_prev + (1.0 - g_k) * u_k

        pos_r = self.pos_residual[pos_idx] if pos_idx < self.pos_residual.size(0) else 0.0
        delta_k = self.residual_proj(torch.cat([e_k, s_k, h_k], dim=-1)) + pos_r

        return s_k, delta_k

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        e_k = self.get_prev_embeddings(token_ids)
        if hidden_states is None:
            return self.project_bias(e_k)
        s = torch.zeros_like(e_k)
        _, delta = self._step(e_k, s, hidden_states, 0)
        return self.project_bias(e_k + delta)

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert hidden_states is not None
        block_size = base_logits.size(-2)
        if block_size == 0:
            return base_logits

        leading_shape = base_logits.shape[:-2]
        s = torch.zeros(
            *leading_shape,
            self.markov_rank,
            device=base_logits.device,
            dtype=hidden_states.dtype,
        )

        output_logits = []
        for k in range(block_size):
            e_k = self.get_prev_embeddings(token_ids[..., k])
            h_k = hidden_states[..., k, :]
            s, delta = self._step(e_k, s, h_k, k)
            output_logits.append(base_logits[..., k, :] + self.project_bias(e_k + delta))

        return torch.stack(output_logits, dim=-2)

    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden_states is not None
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty = torch.empty(batch_size, 0, dtype=torch.long, device=base_logits.device)
            return empty, base_logits

        s = torch.zeros(
            batch_size,
            self.markov_rank,
            device=base_logits.device,
            dtype=hidden_states.dtype,
        )

        sampled_tokens = []
        corrected_logits = []
        prev_token_ids = first_prev_token_ids.long()

        for k in range(proposal_len):
            e_k = self.get_prev_embeddings(prev_token_ids)
            h_k = hidden_states[:, k, :]
            s, delta = self._step(e_k, s, h_k, k)

            step_logits = base_logits[:, k, :] + self.project_bias(e_k + delta)
            corrected_logits.append(step_logits.unsqueeze(1))

            next_token_ids = sample_tokens(
                step_logits.unsqueeze(1), temperature=temperature,
            ).squeeze(1)
            sampled_tokens.append(next_token_ids)
            prev_token_ids = next_token_ids

        return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)
