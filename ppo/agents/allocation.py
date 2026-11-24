"""Sequential PRB allocation with shared target/slice constraints."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class PRBAllocation(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = config["system_parameters"]
        self.n_prb = tuple(cfg["nPrb"])
        self.num_entities = cfg["num_target"] + cfg["num_embb"] + cfg["num_urllc"]
        entities = [self.num_entities] + list(range(cfg["num_target"]))
        for s in range(cfg["num_embb"] + cfg["num_urllc"]):
            count = cfg["num_ue_1embb"] if s < cfg["num_embb"] else cfg["num_ue_1urllc"]
            entities.extend([cfg["num_target"] + s] * count)
        self.register_buffer(
            "entity_of_destination", torch.tensor(entities, dtype=torch.long)
        )
        self.num_choices = len(entities)
        self.prb_to_bwp = tuple(w for w, n in enumerate(self.n_prb) for _ in range(n))

    def forward(
        self,
        logits,
        actions=None,
        *,
        budget=None,
        single_bwp=False,
        deterministic=False,
        epsilon=0.0,
    ):
        """Return action, conditional log-probability and entropy per batch row.

        Each PRB conditions its mask on preceding choices. The same prefix is
        replayed when evaluating PPO actions; idle remains valid in every mask.
        """
        batch, prbs, choices = logits.shape
        if (prbs, choices) != (sum(self.n_prb), self.num_choices):
            raise ValueError("PRB logits have incompatible dimensions")
        if not torch.isfinite(logits).all():
            raise FloatingPointError("PRB logits must be finite")
        if actions is not None:
            actions = torch.as_tensor(actions, device=logits.device, dtype=torch.long)
            if actions.shape != (batch, prbs) or torch.any(
                (actions < 0) | (actions >= choices)
            ):
                raise ValueError("Invalid PRB action shape or destination")
        if budget is None:
            caps = torch.tensor(self.n_prb, device=logits.device).view(1, -1, 1)
            caps = caps.expand(batch, -1, self.num_entities).clone()
        else:
            caps = torch.as_tensor(budget, device=logits.device)
            if caps.shape != (batch, len(self.n_prb), self.num_entities):
                raise ValueError("Budget must be [batch,BWP,target+slice]")
            if (
                not torch.isfinite(caps).all()
                or torch.any(caps < 0)
                or torch.any(caps != caps.floor())
            ):
                raise ValueError("Budget must contain non-negative integer caps")
            caps = caps.long().clone()
        caps = torch.cat([caps, torch.zeros_like(caps[:, :, :1])], dim=-1)
        selected_bwp = torch.full(
            (batch, self.num_entities + 1), -1, device=logits.device, dtype=torch.long
        )
        rows = torch.arange(batch, device=logits.device)
        entities = self.entity_of_destination
        outputs, log_probs, entropies, validity = [], [], [], []
        for r, w in enumerate(self.prb_to_bwp):
            valid = caps[:, w, entities] > 0
            if single_bwp:
                selected = selected_bwp[:, entities]
                valid = valid & ((selected < 0) | (selected == w))
            valid[:, 0] = True
            dist = Categorical(
                logits=logits[:, r].masked_fill(~valid, -torch.inf), validate_args=False
            )
            if actions is not None:
                choice = actions[:, r]
                validity.append(valid[rows, choice])
            elif deterministic:
                choice = dist.logits.argmax(dim=-1)
            elif epsilon > 0:
                greedy = dist.logits.argmax(dim=-1)
                exploratory = Categorical(
                    probs=valid.float(), validate_args=False
                ).sample()
                choice = torch.where(
                    torch.rand(batch, device=logits.device) < epsilon,
                    exploratory,
                    greedy,
                )
            else:
                choice = dist.sample()
            outputs.append(choice)
            log_probs.append(dist.log_prob(choice))
            entropies.append(dist.entropy())
            entity = entities[choice]
            caps[rows, w, entity] -= (choice != 0).long()
            selected_bwp[rows, entity] = w
        if validity and not torch.stack(validity).all():
            raise ValueError("PRB action violates its quota or single-BWP constraint")
        return (
            torch.stack(outputs, dim=1),
            torch.stack(log_probs, dim=1).sum(dim=1),
            torch.stack(entropies, dim=1).sum(dim=1),
        )


def sac_ratios(action, shape, single_bwp=False):
    """Map raw tanh actions to Env ratios; replay and critics retain raw actions."""
    raw = np.asarray(action, dtype=np.float32)
    if (
        raw.size != int(np.prod(shape))
        or not np.isfinite(raw).all()
        or np.any(np.abs(raw) > 1)
    ):
        raise ValueError("SAC action must be finite in [-1,1] with the configured size")
    ratios = ((raw + 1.0) * 0.5).reshape(shape)
    if single_bwp:
        best = ratios.argmax(axis=1)
        keep = np.arange(shape[1])[None, :, None] == best[:, None, :]
        ratios = np.where(keep, ratios, 0.0)
    return ratios
