import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random

from comUtils.common import build_mlp


class DQNReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, local_obs, global_state, action, reward,
             next_local_obs, next_global_state, done, budget_feat):
        self.buffer.append((
            np.asarray(local_obs, dtype=np.float32),
            np.asarray(global_state, dtype=np.float32),
            np.asarray(action, dtype=np.int64),
            float(reward),
            np.asarray(next_local_obs, dtype=np.float32),
            np.asarray(next_global_state, dtype=np.float32),
            float(done),
            np.asarray(budget_feat, dtype=np.float32),
        ))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        lo, gs, act, rew, nlo, ngs, done, bud = zip(*batch)
        return (
            torch.as_tensor(np.stack(lo), dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(gs), dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(act), dtype=torch.int64, device=device),
            torch.as_tensor(rew, dtype=torch.float32, device=device).unsqueeze(1),
            torch.as_tensor(np.stack(nlo), dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(ngs), dtype=torch.float32, device=device),
            torch.as_tensor(done, dtype=torch.float32, device=device).unsqueeze(1),
            torch.as_tensor(np.stack(bud), dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:

    def __init__(self, config, device="cpu"):
        cfg = config.get("dqn", config)

        self.local_obs_dim = int(cfg["local_obs_dim"])
        self.global_state_dim = int(cfg["global_state_dim"])
        self.num_bs = int(cfg["num_bs"])
        self.num_bwp = int(cfg["num_bwp"])
        self.num_targets = int(cfg["num_targets"])
        self.num_slices = int(cfg["num_slices"])
        self.total_prb = int(cfg["total_prb"])
        self.num_destinations = int(cfg["num_destinations"])
        self.num_columns = self.num_targets + self.num_slices
        self.budget_dim = self.num_bwp * self.num_columns
        self.n_prb_per_bwp = np.asarray(cfg["n_prb_per_bwp"], dtype=np.int64)
        # Budget column (target m or slice s) of each destination; -1 for idle.
        self._dest_column = np.asarray(cfg["dest_column"], dtype=np.int64)
        if (self.n_prb_per_bwp.shape != (self.num_bwp,)
                or self.n_prb_per_bwp.sum() != self.total_prb
                or self._dest_column.shape != (self.num_destinations,)):
            raise ValueError("n_prb_per_bwp / dest_column do not match the PRB and destination layout")
        hidden_dims = list(cfg["hidden_dims"])

        self.gamma = float(cfg["gamma"])
        self.lr = float(cfg["lr"])
        self.eps = float(cfg["epsilon_start"])
        self.eps_end = float(cfg["epsilon_min"])
        self.eps_decay = float(cfg["epsilon_decay"])
        self.target_update = int(cfg["target_update_freq"])
        self.grad_clip = float(cfg["grad_clip"])
        self.batch_size = int(cfg["batch_size"])
        self.warmup_steps = int(cfg["warmup_steps"])

        self.device = torch.device(device)
        input_dim = self.local_obs_dim + self.global_state_dim + self.budget_dim

        backbone, out_gain = build_mlp(
            input_dim=input_dim,
            hiddens_dim=hidden_dims,
            activation="relu",
            use_layer_norm=True,
            orthogonal_init=True,
            gain=1.0,
            isActor=False,
        )
        head = nn.Linear(hidden_dims[-1], self.total_prb * self.num_destinations)
        nn.init.orthogonal_(head.weight, gain=out_gain)
        nn.init.zeros_(head.bias)
        self.policy_net = nn.Sequential(*backbone, head).to(self.device)

        backbone_t, out_gain_t = build_mlp(
            input_dim=input_dim,
            hiddens_dim=hidden_dims,
            activation="relu",
            use_layer_norm=True,
            orthogonal_init=True,
            gain=1.0,
            isActor=False,
        )
        head_t = nn.Linear(hidden_dims[-1], self.total_prb * self.num_destinations)
        nn.init.orthogonal_(head_t.weight, gain=out_gain_t)
        nn.init.zeros_(head_t.bias)
        self.target_net = nn.Sequential(*backbone_t, head_t).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        critic_backbone, critic_gain = build_mlp(
            input_dim=self.global_state_dim,
            hiddens_dim=hidden_dims,
            activation="relu",
            use_layer_norm=True,
            orthogonal_init=True,
            gain=1.0,
            isActor=False,
        )
        critic_head = nn.Linear(hidden_dims[-1], 1)
        nn.init.orthogonal_(critic_head.weight, gain=critic_gain)
        nn.init.zeros_(critic_head.bias)
        self.critic_net = nn.Sequential(*critic_backbone, critic_head).to(self.device)

        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.critic_net.parameters()),
            lr=self.lr,
        )

        self.actornetwork = self.policy_net
        self.criticnetwork = self.critic_net
        self.actor_optimizer = self.optimizer
        self.critic_optimizer = self.optimizer
        self.q_net = self.policy_net

        self.replay_buffer = DQNReplayBuffer(int(cfg["replay_capacity"]))
        self._update_step = 0

        self._bwp_of_prb = np.repeat(np.arange(self.num_bwp), self.n_prb_per_bwp)
        self._bwp_of_prb_t = torch.as_tensor(self._bwp_of_prb, device=self.device)
        self._dest_column_t = torch.as_tensor(np.maximum(self._dest_column, 0), device=self.device)

    @property
    def epsilon(self):
        return self.eps

    @epsilon.setter
    def epsilon(self, value):
        self.eps = float(value)

    def budget_features(self, budget):
        """SAC integer budget [B, W, M+S] -> per-BS share of each BWP, shape [B, W*(M+S)]."""
        budget = np.asarray(budget, dtype=np.float32)
        return (budget / self.n_prb_per_bwp[None, :, None]).reshape(budget.shape[0], -1)

    def _assign_within_budget(self, q, quota, explore):
        """Pick one destination per PRB without exceeding the SAC quota of any column.

        Greedy PRBs with the strongest preference choose first; exploring PRBs pick
        uniformly among idle and destinations whose column still has quota.
        """
        quota = np.array(quota, dtype=np.int64, copy=True)
        cols = self._dest_column[1:]
        actions = np.zeros(self.total_prb, dtype=np.int64)
        order = (np.random.permutation(self.total_prb) if explore
                 else np.argsort(-q.max(axis=1), kind="stable"))
        for r in order:
            w = self._bwp_of_prb[r]
            allowed = np.flatnonzero(quota[w, cols] > 0) + 1
            if allowed.size == 0:
                continue
            if explore:
                d = int(np.random.choice(np.append(allowed, 0)))
            else:
                d = int(allowed[np.argmax(q[r, allowed])])
                if q[r, 0] >= q[r, d]:
                    d = 0
            if d:
                quota[w, cols[d - 1]] -= 1
                actions[r] = d
        return actions

    def _feasible_mask(self, budget_feat):
        """[N, P, D] mask of destinations whose column has quota in the PRB's BWP (idle always)."""
        has_quota = budget_feat.view(-1, self.num_bwp, self.num_columns) > 0
        mask = has_quota[:, self._bwp_of_prb_t, :][:, :, self._dest_column_t]
        mask[:, :, 0] = True
        return mask

    def select_action(self, local_obs, global_state, budget):
        B = local_obs.shape[0]
        gs = np.asarray(global_state, dtype=np.float32)
        if gs.ndim == 1:
            gs = np.tile(gs, (B, 1))
        inp = np.concatenate([local_obs, gs, self.budget_features(budget)], axis=1)

        with torch.no_grad():
            q_all = self.policy_net(
                torch.as_tensor(inp, dtype=torch.float32, device=self.device)
            ).view(B, self.total_prb, self.num_destinations).cpu().numpy()

        budget = np.asarray(budget)
        actions_prb = np.stack([
            self._assign_within_budget(q_all[b], budget[b], random.random() < self.eps)
            for b in range(B)
        ])

        actions_pwr = np.ones((B, self.total_prb), dtype=np.float32)
        log_probs = np.zeros(B, dtype=np.float32)

        gs_t = torch.as_tensor(gs[0], dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            value = self.critic_net(gs_t).item()

        return actions_prb, actions_pwr, log_probs, value

    def estimate_value(self, global_state):
        gs = np.asarray(global_state, dtype=np.float32)
        gs_t = torch.as_tensor(gs, dtype=torch.float32, device=self.device)
        if gs_t.dim() == 1:
            gs_t = gs_t.unsqueeze(0)
        with torch.no_grad():
            return self.critic_net(gs_t).squeeze().item()

    def store_transition(self, local_obs, global_state, action, reward,
                         next_local_obs, next_global_state, done, budget):
        # The same frame budget conditions s and s' (goal-conditioned convention).
        self.replay_buffer.push(
            local_obs, global_state, action, reward,
            next_local_obs, next_global_state, float(done),
            self.budget_features(budget),
        )

    def update(self, buffer=None):
        zeros = {
            "dqn/q_loss": 0.0,
            "dqn/epsilon": self.eps,
            "dqn/q_mean": 0.0,
        }
        if len(self.replay_buffer) < max(self.warmup_steps, self.batch_size):
            return zeros

        lo, gs, actions, rewards, nlo, ngs, dones, bud = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        B_batch, num_bs = lo.shape[0], lo.shape[1]
        lo_flat = lo.view(B_batch * num_bs, -1)
        gs_rep = gs.unsqueeze(1).expand(-1, num_bs, -1).reshape(B_batch * num_bs, -1)
        nlo_flat = nlo.view(B_batch * num_bs, -1)
        ngs_rep = ngs.unsqueeze(1).expand(-1, num_bs, -1).reshape(B_batch * num_bs, -1)
        bud_flat = bud.view(B_batch * num_bs, -1)
        acts_flat = actions.view(B_batch * num_bs, -1)
        # Team reward/done are shared by every BS of the same transition.
        rewards = rewards.repeat_interleave(num_bs, dim=0)
        dones = dones.repeat_interleave(num_bs, dim=0)

        inp = torch.cat([lo_flat, gs_rep, bud_flat], dim=-1)
        q_all = self.policy_net(inp).view(-1, self.total_prb, self.num_destinations)
        q_selected = q_all.gather(2, acts_flat.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            inp_next = torch.cat([nlo_flat, ngs_rep, bud_flat], dim=-1)
            q_next_all = self.target_net(inp_next).view(-1, self.total_prb, self.num_destinations)
            # Per-PRB feasibility only; the joint per-column count limit is not enforced here.
            q_next_all = q_next_all.masked_fill(~self._feasible_mask(bud_flat), float("-inf"))
            q_next_max = q_next_all.max(dim=-1).values
            td_target = rewards.expand_as(q_selected) + self.gamma * (1.0 - dones.expand_as(q_selected)) * q_next_max

        q_loss = F.smooth_l1_loss(q_selected, td_target)

        self.optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self._update_step += 1
        if self._update_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return {
            "dqn/q_loss": q_loss.item(),
            "dqn/epsilon": self.eps,
            "dqn/q_mean": q_selected.mean().item(),
        }

    def update_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def save(self, path):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "critic_net": self.critic_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "eps": self.eps,
            "_update_step": self._update_step,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.critic_net.load_state_dict(ckpt["critic_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.eps = ckpt.get("eps", self.eps)
        self._update_step = ckpt.get("_update_step", 0)
