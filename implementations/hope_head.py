"""HOPE-DSpark: Higher-Order Prefix Encoder head.

Extends DSpark's VanillaMarkov with:
  A. 1st-order Markov path (warm-start from trained DSpark basis)
  B. Factorized 2nd-order pair path: MLP(embed(x_{k-2}), embed(x_{k-1})) -> rank-r coefficient
  C. Entropy/uncertainty gate: fires the higher-order correction only where DFlash is uncertain

All higher-order terms emit into the SAME rank-r vocabulary-bias basis U as the Markov head,
so the expensive V x r multiply is shared. Only the coefficient computation changes.

Design: GPT-5.5 Pro (2026-07-01, plan/gpt55pro-hope-dspark-2026-07-01.md).
"""

from typing import Optional

import torch
from torch import nn

from deepspec.modeling.dspark.markov_head import VanillaMarkov
from deepspec.utils.sampling import sample_tokens


class HOPEHead(VanillaMarkov):
    """Higher-Order Prefix Encoder: 2nd-order pair path + entropy gate on top of Markov."""

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        pair_hidden: int = 128,
        gate_bias_init: float = -5.0,
    ):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "hope"

        # B. Pair path: embed(x_{k-2}) + embed(x_{k-1}) -> MLP -> rank-r coefficient
        self.pair_embed = nn.Embedding(vocab_size, markov_rank)
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * markov_rank, pair_hidden),
            nn.GELU(),
            nn.Linear(pair_hidden, markov_rank),
        )

        # C. Entropy gate: scalar gate per position based on DFlash logit entropy
        self.gate_proj = nn.Linear(2, 1)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, gate_bias_init)

        # Zero-init the pair path output so HOPE starts as pure Markov
        nn.init.zeros_(self.pair_mlp[-1].weight)
        nn.init.zeros_(self.pair_mlp[-1].bias)

    def _compute_entropy_features(self, base_logits: torch.Tensor) -> torch.Tensor:
        """Compute (entropy, margin) from base DFlash logits for gating."""
        probs = torch.softmax(base_logits.float(), dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1, keepdim=True)
        top2 = torch.topk(probs, 2, dim=-1).values
        margin = (top2[..., 0:1] - top2[..., 1:2])
        return torch.cat([entropy, margin], dim=-1).to(base_logits.dtype)

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        *,
        prev_prev_token_ids: Optional[torch.Tensor] = None,
        base_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # A. Standard 1st-order Markov
        markov_coeff = self.get_prev_embeddings(token_ids)

        if prev_prev_token_ids is None or base_logits is None:
            return self.project_bias(markov_coeff)

        # B. 2nd-order pair path
        prev_prev_emb = self.pair_embed(prev_prev_token_ids.long())
        prev_emb = self.pair_embed(token_ids.long())
        pair_input = torch.cat([prev_prev_emb, prev_emb], dim=-1)
        pair_coeff = self.pair_mlp(pair_input)

        # C. Entropy gate
        ent_features = self._compute_entropy_features(base_logits)
        gate = torch.sigmoid(self.gate_proj(ent_features))

        # Combined: Markov + gated pair correction
        combined_coeff = markov_coeff + gate * pair_coeff
        return self.project_bias(combined_coeff)

    def apply_step_logits(
        self,
        logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        prev_prev_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bias = self.compute_step_bias(
            token_ids, hidden_states,
            prev_prev_token_ids=prev_prev_token_ids,
            base_logits=logits,
        )
        return logits + bias

    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        temperature: float = 0.0,
        first_prev_prev_token_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty = torch.empty(batch_size, 0, dtype=torch.long, device=base_logits.device)
            return empty, base_logits

        sampled_tokens = []
        corrected_logits = []
        prev_token_ids = first_prev_token_ids.long()
        prev_prev_token_ids = (
            first_prev_prev_token_ids.long()
            if first_prev_prev_token_ids is not None
            else prev_token_ids
        )

        for step_idx in range(proposal_len):
            step_hidden = None if hidden_states is None else hidden_states[:, step_idx, ...]
            step_base = base_logits[:, step_idx, :]

            step_logits = step_base + self.compute_step_bias(
                prev_token_ids, step_hidden,
                prev_prev_token_ids=prev_prev_token_ids,
                base_logits=step_base,
            )
            corrected_logits.append(step_logits.unsqueeze(1))

            next_token_ids = sample_tokens(
                step_logits.unsqueeze(1), temperature=temperature,
            ).squeeze(1)
            sampled_tokens.append(next_token_ids)

            prev_prev_token_ids = prev_token_ids
            prev_token_ids = next_token_ids

        return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        prev_prev_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Training forward: teacher-forced, unrolled over block_size."""
        block_size = base_logits.size(-2)
        if block_size == 0:
            return base_logits

        leading_shape = base_logits.shape[:-2]
        output_logits = []

        for k in range(block_size):
            step_base = base_logits[..., k, :]
            step_prev = token_ids[..., k]

            if k == 0:
                step_prev_prev = prev_prev_token_ids if prev_prev_token_ids is not None else step_prev
            else:
                step_prev_prev = token_ids[..., k - 1]

            step_hidden = None if hidden_states is None else hidden_states[..., k, :]
            bias = self.compute_step_bias(
                step_prev, step_hidden,
                prev_prev_token_ids=step_prev_prev,
                base_logits=step_base,
            )
            output_logits.append(step_base + bias)

        return torch.stack(output_logits, dim=-2)
