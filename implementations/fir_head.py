"""FIR-Markov: Causal Finite Impulse Response head.

Pro-designed (Rank 3, plan/gpt55pro-sequential-head-design-2026-07-02.md).

Maintains a ring buffer of previous Markov embeddings and computes a
weighted sum with hidden-state-gated lag coefficients. No recurrence,
no attention — just a causal filter over previous draft tokens.

At zero-init, δ_k = 0, so the head is exactly VanillaMarkov.
"""

from typing import Optional

import torch
from torch import nn

from deepspec.modeling.dspark.markov_head import VanillaMarkov
from deepspec.utils.sampling import sample_tokens


class FIRMarkovHead(VanillaMarkov):

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        hidden_size: int,
        max_lags: int = 6,
    ):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "fir"
        self.max_lags = max_lags
        self.hidden_size = hidden_size
        r = markov_rank

        self.lag_transforms = nn.ModuleList([
            nn.Linear(r, r, bias=False) for _ in range(max_lags)
        ])
        self.lag_gate_proj = nn.Linear(hidden_size, max_lags, bias=True)
        self.hidden_residual = nn.Linear(hidden_size, r, bias=False)

        self._zero_init()

    def _zero_init(self):
        for t in self.lag_transforms:
            nn.init.zeros_(t.weight)
        nn.init.zeros_(self.lag_gate_proj.weight)
        nn.init.zeros_(self.lag_gate_proj.bias)
        nn.init.zeros_(self.hidden_residual.weight)

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        e_k = self.get_prev_embeddings(token_ids)
        if hidden_states is None:
            return self.project_bias(e_k)
        delta = self.hidden_residual(hidden_states)
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
        embeddings = []
        for k in range(block_size):
            embeddings.append(self.get_prev_embeddings(token_ids[..., k]))

        output_logits = []
        for k in range(block_size):
            e_k = embeddings[k]
            h_k = hidden_states[..., k, :]

            gates = torch.sigmoid(self.lag_gate_proj(h_k))
            delta = self.hidden_residual(h_k)

            for lag_idx in range(min(k, self.max_lags)):
                prev_e = embeddings[k - lag_idx - 1]
                lag_coeff = self.lag_transforms[lag_idx](prev_e)
                delta = delta + gates[..., lag_idx:lag_idx+1] * lag_coeff

            output_logits.append(
                base_logits[..., k, :] + self.project_bias(e_k + delta)
            )

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

        prev_embeddings = []
        sampled_tokens = []
        corrected_logits = []
        prev_token_ids = first_prev_token_ids.long()

        for k in range(proposal_len):
            e_k = self.get_prev_embeddings(prev_token_ids)
            h_k = hidden_states[:, k, :]

            gates = torch.sigmoid(self.lag_gate_proj(h_k))
            delta = self.hidden_residual(h_k)

            for lag_idx in range(min(k, self.max_lags)):
                prev_e = prev_embeddings[k - lag_idx - 1]
                lag_coeff = self.lag_transforms[lag_idx](prev_e)
                delta = delta + gates[:, lag_idx:lag_idx+1] * lag_coeff

            step_logits = base_logits[:, k, :] + self.project_bias(e_k + delta)
            corrected_logits.append(step_logits.unsqueeze(1))

            next_token_ids = sample_tokens(
                step_logits.unsqueeze(1), temperature=temperature,
            ).squeeze(1)
            sampled_tokens.append(next_token_ids)
            prev_embeddings.append(e_k)
            prev_token_ids = next_token_ids

        return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)
