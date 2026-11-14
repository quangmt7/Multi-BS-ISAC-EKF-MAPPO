import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import math

from comUtils.common import build_mlp


class GaussianActor(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dims,
                 log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.backbone, out_gain = build_mlp(
            input_dim=state_dim,
            hiddens_dim=hidden_dims,
            activation="relu",         
            use_layer_norm=True,
            orthogonal_init=True,
            gain=0.01,
            isActor=True,
        )
        self.head = nn.Linear(hidden_dims[-1], 2 * action_dim)
        nn.init.orthogonal_(self.head.weight, gain=out_gain)
        nn.init.zeros_(self.head.bias)

    def forward(self, state):
        features = self.backbone(state)
        mean, _ = torch.chunk(self.head(features), 2, dim=-1)
        return torch.tanh(mean)

    def sample(self, state):
        features = self.backbone(state)
        mean, log_std = torch.chunk(self.head(features), 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std)
        x_t = dist.rsample()
        x_t = torch.clamp(x_t, -6.0, 6.0)
        action = torch.tanh(x_t)

        log_prob = dist.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mean


class SACReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.as_tensor(np.stack(states), dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(actions), dtype=torch.float32, device=device),
            torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1),
            torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device),
            torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class _SACCritic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        self.backbone, out_gain = build_mlp(
            input_dim=state_dim + action_dim,
            hiddens_dim=hidden_dims,
            activation="relu",
            use_layer_norm=True,
            orthogonal_init=True,
            gain=1.0,
            isActor=False,
        )
        self.value_head = nn.Linear(hidden_dims[-1], 1)
        nn.init.orthogonal_(self.value_head.weight, gain=out_gain)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.value_head(self.backbone(x))


class SACAgent:

    def __init__(self, config, device="cpu"):
        cfg = config.get("sac", config)

        self.state_dim = int(cfg["state_dim"])
        self.num_bs = int(cfg["num_bs"])
        self.num_bwp = int(cfg["num_bwp"])
        self.num_targets = int(cfg["num_targets"])
        self.num_slices = int(cfg["num_slices"])
        self.action_dim = self.num_bs * self.num_bwp * (self.num_targets + self.num_slices)
        self.action_shape = (self.num_bs, self.num_bwp, self.num_targets + self.num_slices)

        hidden_dims = list(cfg["hidden_dims"])
        self.gamma = float(cfg["gamma"])
        self.tau = float(cfg["tau"])
        self.policy_delay = int(cfg["policy_delay"])
        self.grad_clip = float(cfg["grad_clip"])
        self.batch_size = int(cfg["batch_size"])
        self.warmup_steps = int(cfg["warmup_steps"])

        self.device = torch.device(device)

        self.actor = GaussianActor(
            self.state_dim, self.action_dim, hidden_dims,
        ).to(self.device)

        self.critic_1 = _SACCritic(self.state_dim, self.action_dim, hidden_dims).to(self.device)
        self.critic_2 = _SACCritic(self.state_dim, self.action_dim, hidden_dims).to(self.device)
        self.target_critic_1 = _SACCritic(self.state_dim, self.action_dim, hidden_dims).to(self.device)
        self.target_critic_2 = _SACCritic(self.state_dim, self.action_dim, hidden_dims).to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        alpha_init = float(cfg.get("alpha_init", 0.2))
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(alpha_init), dtype=torch.float32, device=self.device)
        )
        self.target_entropy = -0.5 * self.action_dim

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=float(cfg["actor_lr"]))
        self.critic_optimizer = optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=float(cfg["critic_lr"]),
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=float(cfg["alpha_lr"]))

        self.replay_buffer = SACReplayBuffer(int(cfg["buffer_capacity"]))
        self._update_step = 0

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def select_action(self, global_state, single_bwp=False, deterministic=False):
        state_t = torch.as_tensor(
            global_state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            if deterministic:
                action = self.actor(state_t)
            else:
                action, _, _ = self.actor.sample(state_t)

        ratios_flat = (action.squeeze(0).cpu().numpy() + 1.0) / 2.0
        ratios = ratios_flat.reshape(self.action_shape)

        if single_bwp:
            for j in range(self.action_shape[2]):
                for b in range(self.num_bs):
                    col = ratios[b, :, j]
                    best = int(np.argmax(col))
                    mask = np.zeros(self.num_bwp, dtype=col.dtype)
                    mask[best] = col[best]
                    ratios[b, :, j] = mask

        ratios = np.maximum(ratios, 0.0)
        return ratios

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(
            state,
            np.asarray(action).flatten(),
            reward,
            next_state,
            float(done),
        )

    def update(self):
        zeros = {
            "sac/critic_loss": 0.0,
            "sac/actor_loss": 0.0,
            "sac/alpha_loss": 0.0,
            "sac/alpha": self.alpha,
            "sac/entropy": 0.0,
        }
        if len(self.replay_buffer) < self.warmup_steps:
            return zeros

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        # --- Critic update ---
        with torch.no_grad():
            a_next, log_pi_next, _ = self.actor.sample(next_states)
            q1_target = self.target_critic_1(next_states, a_next)
            q2_target = self.target_critic_2(next_states, a_next)
            q_target = torch.min(q1_target, q2_target) - self.log_alpha.exp() * log_pi_next
            td_target = rewards + self.gamma * (1.0 - dones) * q_target

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        critic_loss = F.smooth_l1_loss(q1, td_target) + F.smooth_l1_loss(q2, td_target)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            self.grad_clip,
        )
        self.critic_optimizer.step()

        # --- Actor + Alpha update (delayed) ---
        actor_loss_val = 0.0
        alpha_loss_val = 0.0
        entropy_val = 0.0

        self._update_step += 1
        if self._update_step % self.policy_delay == 0:
            a_new, log_pi, _ = self.actor.sample(states)
            q1_new = self.critic_1(states, a_new)
            q2_new = self.critic_2(states, a_new)
            actor_loss = (self.log_alpha.exp() * log_pi - torch.min(q1_new, q2_new)).mean()

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_optimizer.step()

            # Alpha
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

            actor_loss_val = actor_loss.item()
            alpha_loss_val = alpha_loss.item()
            entropy_val = -log_pi.mean().item()

        # --- Soft target update ---
        with torch.no_grad():
            for p, tp in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
            for p, tp in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        return {
            "sac/critic_loss": critic_loss.item(),
            "sac/actor_loss": actor_loss_val,
            "sac/alpha_loss": alpha_loss_val,
            "sac/alpha": self.alpha,
            "sac/entropy": entropy_val,
        }

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "target_critic_1": self.target_critic_1.state_dict(),
            "target_critic_2": self.target_critic_2.state_dict(),
            "log_alpha": self.log_alpha.data,
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "_update_step": self._update_step,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic_1.load_state_dict(ckpt["critic_1"])
        self.critic_2.load_state_dict(ckpt["critic_2"])
        self.target_critic_1.load_state_dict(ckpt["target_critic_1"])
        self.target_critic_2.load_state_dict(ckpt["target_critic_2"])
        self.log_alpha.data.copy_(ckpt["log_alpha"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])
        self._update_step = ckpt.get("_update_step", 0)
