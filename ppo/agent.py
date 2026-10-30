import numpy as np
import torch
import torch.nn as nn
# Import components from AgentComponents.py
try:
    from ppo.AgentComponents import Actor, Critic, MultiAgentRolloutBuffer
except ImportError:
    from AgentComponents import Actor, Critic, MultiAgentRolloutBuffer


class MAPPOAgent:
    """
    Multi-Agent Proximal Policy Optimization controller.
    Manages decentralized Actor policies with Parameter Sharing and a Centralized Critic for cooperative ISAC.
    """
    def __init__(self, config: dict, local_obs_dim, global_state_dim):
        """
        Initialize networks, dual optimizers, and algorithmic hyperparameters for MAPPO.

        Argument:
            local_obs_dim (int): Dimension of the local observation vector for each BS.
            global_state_dim (int): Dimension of the global state vector.
            n_prb (int): Number of PRBs per BS.
            num_targets (int): Number of mobile sensing targets.
            num_ues (int): Number of communication users (UEs).
            num_bs (int): Number of base stations (agents).
            lr_actor (float): Learning rate for the Actor optimizer.
            lr_critic (float): Learning rate for the Critic optimizer. 
            gamma (float): Discount factor for future rewards.
            gae_lambda (float): Lambda parameter for Generalized Advantage Estimation.
            clip_coef (float): PPO surrogate clipping threshold epsilon.
            ent_coef (float): Entropy bonus coefficient.
            vf_coef (float): Value loss coefficient.
            max_grad_norm (float): Maximum norm for gradient clipping.
            update_epochs (int): Number of optimization passes over rollout data.
            batch_size (int): Minibatch size for gradient updates.
            device (str, torch.device, or None): Device for tensor computations ('cpu', 'cuda', or None for auto).

        Return:
            None.
        """
        self.config = config
        # get system parameters
        sys_config= config.get("system_parameters")
        self.n_prb= sys_config["num_prb_bs"]
        self.num_targets= sys_config["num_target"]
        self.num_ues= sys_config["num_ue"]
        self.num_bs = sys_config["num_bs"]
        # get mappo parameter
        mappo_config= config.get("mappo_parameters")
        ppo_config= mappo_config.get("ppo", {})
        train_config = mappo_config.get("training", {})
        
        self.gamma= ppo_config.get("gamma")
        self.gae_lambda = ppo_config.get("gae_lambda")
        self.clip_coef = ppo_config.get("clip_epsilon")
        self.ent_coef = ppo_config.get("entropy_coef")
        self.vf_coef = ppo_config.get("value_loss_coef")
        self.max_grad_norm = ppo_config.get("max_grad_norm")
        self.update_epochs = ppo_config.get("ppo_epochs")
        self.batch_size = ppo_config.get("batch_size")
        # get device
        device_str = train_config.get("device", "cpu")
        self.device= torch.device("cuda" if (device_str =="cuda" and torch.cuda.is_available()) else "cpu")
        
        # actor, critic network and optimizers
        network_config = mappo_config["network"]
        self.actornetwork = Actor(local_obs_dim, self.n_prb, self.num_targets, self.num_ues, network_config["actor"]).to(self.device)
        self.criticnetwork= Critic(global_state_dim, network_config["critic"]).to(self.device)
        self.actor_optimizer= torch.optim.Adam(self.actornetwork.parameters(), lr= ppo_config.get("lr_actor"))
        self.critic_optimizer= torch.optim.Adam(self.criticnetwork.parameters(), lr= ppo_config.get("lr_critic"))
        self.epoch = 0
        self.last_change = 0
    
    def select_action(self, local_obs_list, global_state):
        """
        Select actions for all B base stations and evaluate global state-value during environment interaction.

        Argument:
            local_obs_list (np.ndarray)[B, local_obs_dim]: Local observations for all B base stations.
            global_state (np.ndarray)[global_state_dim]: Global network state vector.

        Return:
            act_prb (np.ndarray)[B, n_prb]: Discrete PRB selection decisions for B BSs (dtype int64).
            act_pwr (np.ndarray)[B, n_prb]: Continuous normalized power allocations for B BSs in [0, 1] (dtype float32).
            logp (np.ndarray)[B]: Action log-probabilities for each BS.
            val (float): Scalar state-value V(S) predicted by the Centralized Critic.

        Note:
            Decentralized execution: Actor processes B agents concurrently via parameter sharing.
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(local_obs_list, dtype=torch.float32, device=self.device)
            state_t = torch.as_tensor(global_state, dtype=torch.float32, device=self.device)
            if state_t.ndim ==1:
                state_t=state_t.unsqueeze(0)

            assert obs_t.ndim == 2, (
            f"Expected obs [B, obs_dim], got {obs_t.shape}"
            )
            assert state_t.ndim == 2 and state_t.shape[0] == 1, (
                f"Expected one global state, got {state_t.shape}"
            )
                
            act_prb, act_pwr, logp, _ = self.actornetwork(obs_t)
            val = self.criticnetwork(state_t)
            #if act_pwr is not None:
            #    print(
            #        f"power "
            #        f"min={act_pwr.min().item():.8f} "
            #        f"max={act_pwr.max().item():.8f}"
            #    )
            return (
                act_prb.cpu().numpy(),
                act_pwr.cpu().numpy(),
                logp.cpu().numpy(),
                val.squeeze().item()
            )

    def compute_gae(self, buffer):
        """
        Compute Generalized Advantage Estimation (GAE) and discounted return targets backwards across rollout steps.

        Argument:
            buffer (MultiAgentRolloutBuffer): Rollout buffer holding T timesteps of multi-agent interactions.

        Return:
            advantages_t (torch.Tensor)[T]: Computed Generalized Advantage Estimations.
            returns_t (torch.Tensor)[T]: Discounted target returns for Critic training.
        """
        with torch.no_grad():
            rewards = buffer.rewards
            terminateds = buffer.terminateds
            truncateds = buffer.truncateds
            values = buffer.values
            next_values = buffer.next_values
            T = len(rewards)

            advantages = [0.0] * T
            last_gaelam = 0.0

            for t in reversed(range(T)):
                next_val_step = next_values[t]
                next_nonterminal = 1.0 - float(terminateds[t])
                next_done = float(terminateds[t] or truncateds[t])

                # TD Error: delta = r + gamma * V(s') * (1 - terminated) - V(s)
                delta = rewards[t] + self.gamma * next_val_step * next_nonterminal - values[t]

                # GAE: A_t = delta + gamma * lambda * (1 - done) * A_{t+1}
                advantages[t] = last_gaelam = delta + self.gamma * self.gae_lambda * (1.0 - next_done) * last_gaelam

            returns = [adv + val for adv, val in zip(advantages, values)]
            # convert and return Advantages, Q 
            advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

            return advantages_t, returns_t

    def estimate_value(self, global_state):
        """Estimate the critic value of one global state.

        Argument:
            (global_state) (np.ndarray)[global_state_dim]: State after a transition.

        Return:
            (value) (float): Estimated state value.
        """
        with torch.no_grad():
            state = torch.as_tensor(global_state, dtype=torch.float32, device=self.device)
            return float(self.criticnetwork(state.unsqueeze(0)).item())
        
    def update(self, buffer):
        """
        PPO update for parameter-sharing actor and centralized critic.
        """

        # =====================================================
        # 1. GAE
        # =====================================================
        b_advantages, b_returns = self.compute_gae(
            buffer
        )

        (
            b_obs,
            b_states,
            b_act_prb,
            b_act_pwr,
            b_log_probs,
            _,
            _,
            _,
            b_values,
        ) = buffer.get(self.device)

        # =====================================================
        # 2. Advantage normalization
        # =====================================================
        adv_std = b_advantages.std(unbiased=False)

        if adv_std > 1e-8:
            b_advantages = (
                b_advantages - b_advantages.mean()
            ) / (adv_std + 1e-8)
        else:
            b_advantages = (
                b_advantages - b_advantages.mean()
            )

        # =====================================================
        # 3. Flatten actor samples
        # =====================================================
        T, B = b_obs.shape[:2]

        flat_obs = b_obs.reshape(T * B, -1)

        flat_act_prb = b_act_prb.reshape(
            T * B,
            b_act_prb.shape[-1]
        )

        flat_act_pwr = b_act_pwr.reshape(
            T * B,
            b_act_pwr.shape[-1]
        )

        flat_logp = b_log_probs.reshape(-1)

        # shared team advantage
        flat_adv = (
            b_advantages
            .unsqueeze(1)
            .expand(T, B)
            .reshape(-1)
        )

        # =====================================================
        # DEBUG: CHECK STORED LOGP VS RECOMPUTED LOGP
        # =====================================================
        #with torch.no_grad():

        #    _, _, recomputed_logp, _ = self.actornetwork(
        #        flat_obs,
        #        flat_act_prb,
        #        flat_act_pwr,
        #    )

        #    log_ratio = recomputed_logp - flat_logp
        #    ratio = torch.exp(log_ratio)

            #print("\n================ PPO INITIAL CHECK ================")

        #    print(f"logp_old mean    : {flat_logp.mean().item():.6f}")
        #    print(f"logp_old std     : {flat_logp.std(unbiased=False).item():.6f}")

        #    print(f"logp_new mean    : {recomputed_logp.mean().item():.6f}")
        #    print(f"logp_new std     : {recomputed_logp.std(unbiased=False).item():.6f}")

        #    print(f"log_ratio mean   : {log_ratio.mean().item():.6f}")
        #    print(f"log_ratio std    : {log_ratio.std(unbiased=False).item():.6f}")

        #    print(f"ratio mean       : {ratio.mean().item():.6f}")
        #    print(f"ratio std        : {ratio.std(unbiased=False).item():.6f}")
        #    print(f"ratio min        : {ratio.min().item():.6f}")
        #    print(f"ratio max        : {ratio.max().item():.6f}")

        #    print("===================================================\n")

        # =====================================================
        # Metrics
        # =====================================================
        pg_losses = []
        v_losses = []
        ent_losses = []

        ratio_means = []
        ratio_stds = []
        clip_fractions = []

        # =====================================================
        # 4. ACTOR UPDATE
        # =====================================================
        total_actor_samples = T * B
        actor_indices = np.arange(total_actor_samples)

        for _ in range(self.update_epochs):

            np.random.shuffle(actor_indices)

            for start in range(
                0,
                total_actor_samples,
                self.batch_size
            ):

                end = min(
                    start + self.batch_size,
                    total_actor_samples
                )

                mb_inds = actor_indices[start:end]

                mb_obs = flat_obs[mb_inds]
                mb_act_prb = flat_act_prb[mb_inds]
                mb_act_pwr = flat_act_pwr[mb_inds]

                mb_old_logp = flat_logp[mb_inds]
                mb_adv = flat_adv[mb_inds]

                (
                    _,
                    _,
                    new_logp,
                    entropy
                ) = self.actornetwork(
                    mb_obs,
                    mb_act_prb,
                    mb_act_pwr,
                )

                # PPO ratio
                log_ratio = new_logp - mb_old_logp
                ratio = torch.exp(log_ratio)

                surr1 = ratio * mb_adv

                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_coef,
                        1.0 + self.clip_coef,
                    )
                    * mb_adv
                )

                policy_loss = (
                    -torch.minimum(
                        surr1,
                        surr2
                    ).mean()
                )

                entropy_bonus = entropy.mean()

                actor_loss = (
                    policy_loss
                    - self.ent_coef * entropy_bonus
                )

                self.actor_optimizer.zero_grad(
                    set_to_none=True
                )

                actor_loss.backward()

                nn.utils.clip_grad_norm_(
                    self.actornetwork.parameters(),
                    self.max_grad_norm,
                )

                self.actor_optimizer.step()

                # metrics
                clip_fraction = (
                    (ratio - 1.0).abs()
                    > self.clip_coef
                ).float().mean()

                pg_losses.append(
                    policy_loss.item()
                )

                ent_losses.append(
                    entropy_bonus.item()
                )

                ratio_means.append(
                    ratio.mean().item()
                )

                ratio_stds.append(
                    ratio.std(
                        unbiased=False
                    ).item()
                )

                clip_fractions.append(
                    clip_fraction.item()
                )


        # =====================================================
        # 5. CRITIC UPDATE
        # =====================================================
        critic_indices = np.arange(T)

        for _ in range(self.update_epochs):

            np.random.shuffle(
                critic_indices
            )

            for start in range(
                0,
                T,
                self.batch_size
            ):

                end = min(
                    start + self.batch_size,
                    T
                )

                mb_inds = critic_indices[start:end]

                mb_states = b_states[mb_inds]
                mb_returns = b_returns[mb_inds]

                new_values = (
                    self.criticnetwork(
                        mb_states
                    )
                    .squeeze(-1)
                )

                value_loss = (
                    0.5
                    * (
                        (new_values - mb_returns)
                        ** 2
                    ).mean()
                )

                critic_loss = (
                    self.vf_coef
                    * value_loss
                )

                self.critic_optimizer.zero_grad(
                    set_to_none=True
                )

                critic_loss.backward()

                nn.utils.clip_grad_norm_(
                    self.criticnetwork.parameters(),
                    self.max_grad_norm,
                )

                self.critic_optimizer.step()

                v_losses.append(
                    critic_loss.item()
                )

        # =====================================================
        # 6. CLEAR BUFFER
        # =====================================================
        buffer.clear()

        # =====================================================
        # 7. RETURN
        # =====================================================
        return {
            "policy_loss": float(np.mean(pg_losses)),
            "critic_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(ent_losses)),
            "ratio_mean": float(np.mean(ratio_means)),
            "ratio_std": float(np.mean(ratio_stds)),
            "clip_fraction": float(np.mean(clip_fractions)),
        }