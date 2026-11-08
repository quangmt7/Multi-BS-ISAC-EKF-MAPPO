import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from comUtils.common import get_activation, build_mlp

class SACAgent:
    def __init__(self, state_dim, num_rus, num_bwp_ru, num_slices, train_cons):
        """
        Agent cho SAC
        int state_dim : số chiều của state
        int num_rus : số RU
        list(int) num_bwp_ru : số BWP mỗi RU 
        int num_slices : số slice
        """

        self.state_dim = state_dim
        self.num_rus = num_rus
        self.num_bwp_ru = num_bwp_ru
        self.num_slices = num_slices    
        self.device = train_cons["device"]
        self.gamma = train_cons["gamma"]
        self.tau = train_cons["tau"]
        self.alpha = train_cons["alpha"]
        self.action_scale = train_cons["action_scale"]
        self.action_bias = train_cons["action_bias"]
        self.lr = train_cons["lr"]
        self.actor_lr = train_cons["actor_lr"]
        self.critic_lr = train_cons["critic_lr"]
        self.alpha_lr = train_cons["alpha_lr"]

        self.action_dim = np.sum(num_bwp_ru) * num_slices

        # --- Actor & Critics
        self.actor = GaussianPolicy(state_dim, self.action_dim,
                                    action_scale=self.action_scale,
                                    action_bias=self.action_bias).to(self.device)

        # Critics Q1, Q2
        q_input_dim = state_dim + self.action_dim  
        self.critic_1 = MLP(q_input_dim, 1).to(self.device)
        self.critic_1.init_weights()
        self.critic_2 = MLP(q_input_dim, 1).to(self.device)
        self.critic_2.init_weights()
        self.target_critic_1 = MLP(q_input_dim, 1).to(self.device)
        self.target_critic_1.init_weights()
        self.target_critic_2 = MLP(q_input_dim, 1).to(self.device)
        self.target_critic_2.init_weights()

        # copy weights to targets
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_1_opt = optim.Adam(self.critic_1.parameters(), lr=self.critic_lr)
        self.critic_2_opt = optim.Adam(self.critic_2.parameters(), lr=self.critic_lr)

        # --- Replay buffer (same as before)
        self.replay_buffer = ReplayBuffer(forSAC=True)

        # --- Entropy tuning (correct, register log_alpha as Parameter)
        self.target_entropy = -0.5 * float(self.action_dim)

        init_alpha = float(self.alpha) if hasattr(self, "alpha") else float(0.3)
        # register as nn.Parameter so it's in model.parameters()/state_dict()
        self.log_alpha = torch.nn.Parameter(
            torch.tensor(np.log(init_alpha), dtype=torch.float32, device=self.device)
        )
        # scalar float for immediate use in ops
        self.alpha = float(self.log_alpha.exp().item())

        # optimizer for log_alpha
        self.alpha_opt = optim.Adam([self.log_alpha], lr=self.alpha_lr)


    def select_action(self, state):

        state = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action_sample, _, _ = self.actor.sample(state)

        # Chuyển [-1,1] -> [0,1]
        action = (action_sample + 1.0) * 0.5

        action = action.view(
            self.num_rus,
            max(self.num_bwp_ru),
            self.num_slices
        )

        eps = 1e-8

        for r in range(self.num_rus):
            for b in range(self.num_bwp_ru[r]):

                action[r, b] = torch.clamp(action[r, b], 0.0, 1.0)

                # bỏ allocation quá nhỏ
                action[r, b][action[r, b] < 0.01] = 0.0

                total = action[r, b].sum()

                if total > eps:

                    # pattern
                    pattern = action[r, b] / total

                    # Budget sử dụng
                    usage = total / self.num_slices

                    # Mở rộng khoảng budget
                    usage = torch.pow(usage, 0.7)

                    usage = torch.clamp(usage, 0.0, 1.0)
                    usage = min(1.1 * usage, 1)

                    action[r, b] = pattern * usage

                else:

                    action[r, b].zero_()

        return action.squeeze(0).cpu().numpy()

    def update(self, step, policy_delay, last_actor_loss, batch_size, debug=False):

        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0, last_actor_loss

        # ==========================================================
        # Sample replay
        # ==========================================================
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(batch_size)

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # ==========================================================
        # Critic target
        # ==========================================================
        with torch.no_grad():

            next_action, next_log_pi, _ = self.actor.sample(next_states)

            target_q1 = self.target_critic_1(
                torch.cat([next_states, next_action], dim=-1)
            )

            target_q2 = self.target_critic_2(
                torch.cat([next_states, next_action], dim=-1)
            )

            target_q = torch.min(target_q1, target_q2)

            target = rewards + \
                    (1.0 - dones) * self.gamma * \
                    (target_q - self.alpha * next_log_pi)

        # ==========================================================
        # Critic update
        # ==========================================================
        current_q1 = self.critic_1(
            torch.cat([states, actions], dim=-1)
        )

        current_q2 = self.critic_2(
            torch.cat([states, actions], dim=-1)
        )

        critic_loss1 = F.smooth_l1_loss(current_q1, target)
        critic_loss2 = F.smooth_l1_loss(current_q2, target)

        critic_loss = critic_loss1 + critic_loss2

        self.critic_1_opt.zero_grad()
        critic_loss1.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic_1.parameters(),
            5.0
        )
        self.critic_1_opt.step()

        self.critic_2_opt.zero_grad()
        critic_loss2.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic_2.parameters(),
            5.0
        )
        self.critic_2_opt.step()

        # ==========================================================
        # Actor update
        # ==========================================================
        actor_loss = last_actor_loss

        if step % policy_delay == 0:

            new_action, log_pi, mean = self.actor.sample(states)

            q1 = self.critic_1(
                torch.cat([states, new_action], dim=-1)
            )

            q2 = self.critic_2(
                torch.cat([states, new_action], dim=-1)
            )

            q = torch.min(q1, q2)

            actor_loss = (
                self.alpha * log_pi - q
            ).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                5.0
            )

            self.actor_opt.step()

            last_actor_loss = actor_loss.detach()

        # ==========================================================
        # Alpha update
        # ==========================================================
        _, log_pi, _ = self.actor.sample(states)

        alpha_loss = -(
            self.log_alpha *
            (log_pi + self.target_entropy).detach()
        ).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self.alpha = self.log_alpha.exp().item()

        # ==========================================================
        # Soft update
        # ==========================================================
        with torch.no_grad():

            for tp, p in zip(
                    self.target_critic_1.parameters(),
                    self.critic_1.parameters()):
                tp.data.mul_(1 - self.tau)
                tp.data.add_(self.tau * p.data)

            for tp, p in zip(
                    self.target_critic_2.parameters(),
                    self.critic_2.parameters()):
                tp.data.mul_(1 - self.tau)
                tp.data.add_(self.tau * p.data)

        # ==========================================================
        # Debug
        # ==========================================================
        if debug and step % 500 == 0:

            print("=" * 60)
            print(f"Step: {step}")

            print(f"Reward          : {rewards.mean().item():.4f}")

            print(f"Critic loss     : {critic_loss.item():.4f}")
            print(f"Actor loss      : {actor_loss.item():.4f}")

            print(f"Q mean          : {q.mean().item():.4f}")
            print(f"Target Q mean   : {target.mean().item():.4f}")

            print(f"Alpha           : {self.alpha:.5f}")
            print(f"Alpha loss      : {alpha_loss.item():.4f}")

            print(f"log_pi mean     : {log_pi.mean().item():.4f}")

            print(f"Mean(abs)       : {mean.abs().mean().item():.4f}")

            std = self.actor.forward(states)[1].exp()

            print(f"Std mean        : {std.mean().item():.4f}")

            entropy = -log_pi.mean().item()

            print(f"Entropy         : {entropy:.4f}")

            print("=" * 60)

        return (
            float(actor_loss.item()),
            float(critic_loss.item()),
            last_actor_loss
        )

