"""PRB-only policy, centralized value network and on-policy rollout storage."""

import numpy as np
import torch
from torch import nn
from comUtils.common import build_mlp
from ppo.agents.allocation import PRBAllocation


class Actor(nn.Module):
    def __init__(self, local_obs_dim, config):
        super().__init__()
        net = config["mappo_parameters"]["network"]["actor"]
        self.allocation = PRBAllocation(config)
        self.n_prb = sum(self.allocation.n_prb)
        self.num_choices = len(self.allocation.entity_of_destination)
        self.backbone, gain = build_mlp(
            input_dim=local_obs_dim,
            hiddens_dim=net["hidden_dims"],
            activation=net["activation"],
            use_layer_norm=net["use_layer_norm"],
            orthogonal_init=net["orthogonal_init"],
            gain=net["gain"],
            isActor=True,
        )
        self.prb_head = nn.Linear(net["hidden_dims"][-1], self.n_prb * self.num_choices)
        if net["orthogonal_init"]:
            nn.init.orthogonal_(self.prb_head.weight, gain=gain)
            nn.init.zeros_(self.prb_head.bias)

    def forward(self, obs, action_prb=None, *, single_bwp=False, deterministic=False):
        logits = self.prb_head(self.backbone(obs)).reshape(
            -1, self.n_prb, self.num_choices
        )
        return self.allocation(
            logits,
            actions=action_prb,
            single_bwp=single_bwp,
            deterministic=deterministic,
        )


class Critic(nn.Module):
    """
    Centralized Critic network for MAPPO.
    Estimates the expected global state-value V(S) for variance reduction during centralized training.

    Note:
        Operates under centralized training: accepts global state containing all BSs, targets, and channel info.
    """

    def __init__(self, global_state_dim, net_conf):
        """
        Initialize the Centralized Critic network backbone.

        Argument:
            global_state_dim (int): Dimension of the global state vector.
            net_conf (dict): Network configuration

        Return:
            None.
        """
        super().__init__()
        self.backbone, self.out_gain = build_mlp(
            input_dim=global_state_dim,
            hiddens_dim=net_conf["hidden_dims"],
            activation=net_conf["activation"],
            use_layer_norm=net_conf["use_layer_norm"],
            orthogonal_init=net_conf["orthogonal_init"],
            gain=net_conf["gain"],
            isActor=False,
        )
        self.value_head = nn.Linear(net_conf["hidden_dims"][-1], 1)
        if net_conf["orthogonal_init"]:
            nn.init.orthogonal_(self.value_head.weight, gain=self.out_gain)
            nn.init.zeros_(self.value_head.bias)

    def forward(self, global_state):
        """
        Forward pass to compute state-value estimates.

        Argument:
            global_state (torch.Tensor)[batch, global_state_dim]: Global environment state vector.

        Return:
            value (torch.Tensor)[batch, 1]: Estimated scalar state-value V(S).
        """
        return self.value_head(self.backbone(global_state))


class MultiAgentRolloutBuffer:
    """One transition per physical step; timeout bootstraps but ends the GAE trace."""

    fields = (
        "local_obs",
        "global_obs",
        "actions_prb",
        "log_probs",
        "rewards",
        "terminateds",
        "truncateds",
        "values",
        "next_values",
    )

    def __init__(self):
        self.clear()

    def push(
        self,
        local_obs,
        global_state,
        act_prb,
        logp,
        reward,
        terminated,
        truncated,
        value,
        next_value,
    ):
        values = (
            np.array(local_obs, copy=True),
            np.array(global_state, copy=True),
            np.array(act_prb, copy=True),
            np.array(logp, copy=True),
            float(reward),
            bool(terminated),
            bool(truncated),
            float(value),
            float(next_value),
        )
        for name, value in zip(self.fields, values):
            getattr(self, name).append(value)

    def clear(self):
        for name in self.fields:
            setattr(self, name, [])

    def get(self, device="cpu"):
        return tuple(
            torch.as_tensor(
                np.asarray(getattr(self, name)),
                device=device,
                dtype=(
                    torch.long
                    if name == "actions_prb"
                    else torch.bool
                    if name in ("terminateds", "truncateds")
                    else torch.float32
                ),
            )
            for name in self.fields
        )

    def __len__(self):
        return len(self.rewards)
