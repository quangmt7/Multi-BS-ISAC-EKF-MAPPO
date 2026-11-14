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
             next_local_obs, next_global_state, done):
        self.buffer.append((
            np.asarray(local_obs, dtype=np.float32),
            np.asarray(global_state, dtype=np.float32),
            np.asarray(action, dtype=np.int64),
            float(reward),
            np.asarray(next_local_obs, dtype=np.float32),
            np.asarray(next_global_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        lo, gs, act, rew, nlo, ngs, done = zip(*batch)
        return (
            torch.as_tensor(np.stack(lo), dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(gs), dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(act), dtype=torch.int64, device=device),
            torch.as_tensor(rew, dtype=torch.float32, device=device).unsqueeze(1),
            torch.as_tensor(np.stack(nlo), dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(ngs), dtype=torch.float32, device=device),
            torch.as_tensor(done, dtype=torch.float32, device=device).unsqueeze(1),
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
        hidden_dims = list(cfg["hidden_dims"])

        self.gamma = float(cfg["gamma"])
        self.lr = float(cfg["lr"])
        self.eps = float(cfg["eps_start"])
        self.eps_end = float(cfg["eps_end"])
        self.eps_decay = float(cfg["eps_decay"])
        self.target_update = int(cfg["target_update"])
        self.grad_clip = float(cfg["grad_clip"])
        self.batch_size = int(cfg["batch_size"])
        self.warmup_steps = int(cfg["warmup_steps"])

        self.device = torch.device(device)
        input_dim = self.local_obs_dim + self.global_state_dim

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

        self.replay_buffer = DQNReplayBuffer(int(cfg["buffer_capacity"]))
        self._update_step = 0

        nprb_per_bwp = self.total_prb // self.num_bwp
        self._bwp_of_prb = np.array(
            [w for w in range(self.num_bwp) for _ in range(nprb_per_bwp)],
            dtype=np.int64,
        )

    def select_action(self, local_obs, global_state, single_bwp=False):
        B = local_obs.shape[0]
        gs = np.asarray(global_state, dtype=np.float32)
        if gs.ndim == 1:
            gs_expanded = np.broadcast_to(gs, (B, gs.shape[0]))
        else:
            gs_expanded = gs

        actions_list = []
        with torch.no_grad():
            for b in range(B):
                inp = np.concatenate([local_obs[b], gs_expanded[b]])
                inp_t = torch.as_tensor(inp, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_flat = self.policy_net(inp_t).squeeze(0)
                q_values = q_flat.view(self.total_prb, self.num_destinations)

                if random.random() < self.eps:
                    act = np.random.randint(0, self.num_destinations, size=self.total_prb)
                else:
                    act = q_values.argmax(dim=-1).cpu().numpy()

                actions_list.append(act)

        actions_prb = np.stack(actions_list).astype(np.int64)

        if single_bwp:
            for dest in range(1, self.num_destinations):
                for b in range(B):
                    prb_mask = actions_prb[b] == dest
                    if not prb_mask.any():
                        continue
                    bwp_ids = self._bwp_of_prb[prb_mask]
                    counts = np.bincount(bwp_ids, minlength=self.num_bwp)
                    best_bwp = int(np.argmax(counts))
                    kill = prb_mask & (self._bwp_of_prb != best_bwp)
                    actions_prb[b, kill] = 0

        actions_pwr = np.ones((B, self.total_prb), dtype=np.float32)
        log_probs = np.zeros(B, dtype=np.float32)

        gs_t = torch.as_tensor(gs if gs.ndim == 1 else gs[0],
                               dtype=torch.float32, device=self.device).unsqueeze(0)
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
                         next_local_obs, next_global_state, done):
        self.replay_buffer.push(
            local_obs, global_state, action, reward,
            next_local_obs, next_global_state, float(done),
        )

    def update(self, buffer=None):
        zeros = {
            "dqn/q_loss": 0.0,
            "dqn/epsilon": self.eps,
            "dqn/q_mean": 0.0,
        }
        if len(self.replay_buffer) < self.warmup_steps:
            return zeros

        lo, gs, actions, rewards, nlo, ngs, dones = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        B_batch = lo.shape[0]
        num_bs = lo.shape[1] if lo.dim() == 3 else 1

        if lo.dim() == 3:
            lo_flat = lo.view(B_batch * num_bs, -1)
            gs_rep = gs.unsqueeze(1).expand(-1, num_bs, -1).reshape(B_batch * num_bs, -1)
            nlo_flat = nlo.view(B_batch * num_bs, -1)
            ngs_rep = ngs.unsqueeze(1).expand(-1, num_bs, -1).reshape(B_batch * num_bs, -1)
            acts_flat = actions.view(B_batch * num_bs, -1)
        else:
            lo_flat = lo
            gs_rep = gs
            nlo_flat = nlo
            ngs_rep = ngs
            acts_flat = actions

        inp = torch.cat([lo_flat, gs_rep], dim=-1)
        q_all = self.policy_net(inp).view(-1, self.total_prb, self.num_destinations)
        q_selected = q_all.gather(2, acts_flat.unsqueeze(-1)).squeeze(-1)
        q_loss_per_sample = q_selected.mean(dim=-1)

        with torch.no_grad():
            inp_next = torch.cat([nlo_flat, ngs_rep], dim=-1)
            q_next_all = self.target_net(inp_next).view(-1, self.total_prb, self.num_destinations)
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
