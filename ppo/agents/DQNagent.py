"""Factorized PRB Q-network with feasible sequential action selection."""

import copy
import torch
from torch import nn
from torch.nn import functional as F
from comUtils.common import build_mlp
from ppo.utils.project_config import hier_hidden_dims
from ppo.agents.allocation import PRBAllocation
from ppo.agents.replay import ReplayBuffer


class DQNAgent:
    def __init__(self, config, state_dim, device="cpu", *, single_bwp=False):
        cfg = config["hierarchical_parameters"]["dqn"]
        self.device = torch.device(device)
        self.single_bwp = bool(single_bwp)
        self.allocation = PRBAllocation(config).to(self.device)
        self.n_prb = sum(self.allocation.n_prb)
        self.num_choices = len(self.allocation.entity_of_destination)
        hidden = hier_hidden_dims(cfg)
        backbone, gain = build_mlp(state_dim, hidden, "relu", True, True, 1.0, False)
        head = nn.Linear(hidden[-1], self.n_prb * self.num_choices)
        nn.init.orthogonal_(head.weight, gain=gain)
        nn.init.zeros_(head.bias)
        self.q_net = nn.Sequential(backbone, head).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=cfg["lr"])
        self.replay_buffer = ReplayBuffer(cfg["replay_capacity"])
        self.gamma = cfg["gamma"]
        self.epsilon = cfg["epsilon_start"]
        self.epsilon_min = cfg["epsilon_min"]
        self.epsilon_decay = cfg["epsilon_decay"]
        self.batch_size = cfg["batch_size"]
        self.target_update_freq = cfg["target_update_freq"]
        self.grad_clip = cfg.get("grad_clip", cfg.get("max_grad_norm", 5.0))
        self.update_step = 0

    def select_action(self, state, *, budget, bwp_mode=None, deterministic=False):
        with torch.no_grad():
            states = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            q = self.q_net(states).view(-1, self.n_prb, self.num_choices)
            actions, _, _ = self.allocation(
                q,
                budget=budget,
                single_bwp=self.single_bwp if bwp_mode is None else not bwp_mode,
                deterministic=deterministic or self.epsilon == 0,
                epsilon=0.0 if deterministic else self.epsilon,
            )
        return actions.cpu().numpy()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update(self, batch_size=None):
        size = self.batch_size if batch_size is None else batch_size
        if len(self.replay_buffer) < size:
            return {"dqn_loss": 0.0, "updated": False}
        state, action, reward, next_state, terminated, budget, next_budget = (
            self.replay_buffer.sample(size, self.device)
        )
        q = self.q_net(state).view(-1, self.n_prb, self.num_choices)
        # Shared scalar return uses the mean of selected PRB Q-heads. The
        # constrained greedy decoder is an approximation to the joint argmax.
        selected_q = q.gather(2, action.long().unsqueeze(-1)).squeeze(-1).mean(dim=1)
        with torch.no_grad():
            next_q = self.target_net(next_state).view(-1, self.n_prb, self.num_choices)
            next_action, _, _ = self.allocation(
                next_q,
                budget=next_budget,
                single_bwp=self.single_bwp,
                deterministic=True,
            )
            next_value = (
                next_q.gather(2, next_action.unsqueeze(-1)).squeeze(-1).mean(dim=1)
            )
            target = reward + self.gamma * (1.0 - terminated) * next_value
        loss = F.smooth_l1_loss(selected_q, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite DQN loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.q_net.parameters(), self.grad_clip, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.update_step += 1
        if self.update_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return {"dqn_loss": float(loss.item()), "updated": True}
