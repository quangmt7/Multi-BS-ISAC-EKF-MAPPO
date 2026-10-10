import numpy as np
import torch
import torch.nn as nn
# Import components from AgentComponents.py
try:
    from PPO.AgentComponents import Actor, Critic, MultiAgentRolloutBuffer
except ImportError:
    from AgentComponents import Actor, Critic, MultiAgentRolloutBuffer


class MAPPOAgent:
    """
    Multi-Agent Proximal Policy Optimization controller.
    Manages decentralized Actor policies with Parameter Sharing and a Centralized Critic for cooperative ISAC.
    """
    def __init__(self, config: dict, local_obs_dim, global_state_dim ):
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
        self.n_prb= sys_config["num_prb"]
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
        self.actornetwork = Actor(local_obs_dim, self.n_prb, self.num_targets, self.num_ues).to(self.device)
        self.criticnetwork= Critic(global_state_dim).to(self.device)
        self.actor_optimizer= torch.optim.Adam(self.actornetwork.parameters(), lr= ppo_config.get("lr_actor"))
        self.critic_optimizer= torch.optim.Adam(self.criticnetwork.parameters(), lr= ppo_config.get("lr_critic"))
    
    
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
                
            act_prb, act_pwr, logp, _ = self.actornetwork(obs_t)
            val = self.criticnetwork(state_t)
            return (
                act_prb.cpu().numpy(),
                act_pwr.cpu().numpy(),
                logp.cpu().numpy(),
                val.squeeze().item()
            )

    def compute_gae(self, buffer, next_global_state):
        """
        Compute Generalized Advantage Estimation (GAE) and discounted return targets backwards across rollout steps.

        Argument:
            buffer (MultiAgentRolloutBuffer): Rollout buffer holding T timesteps of multi-agent interactions.
            next_global_state (np.ndarray)[global_state_dim]: Global state of the timestep following the rollout (S_T).

        Return:
            advantages_t (torch.Tensor)[T]: Computed Generalized Advantage Estimations.
            returns_t (torch.Tensor)[T]: Discounted target returns for Critic training.
        """
        with torch.no_grad():
            next_state_t = torch.as_tensor(next_global_state, dtype=torch.float32, device=self.device)
            if next_state_t.ndim == 1:
                next_state_t = next_state_t.unsqueeze(0)

            next_val = self.criticnetwork(next_state_t).squeeze().item()
            rewards = buffer.rewards
            terminateds = buffer.terminateds
            truncateds = buffer.truncateds
            values = buffer.values
            T = len(rewards)

            advantages = [0.0] * T
            last_gaelam = 0.0

            for t in reversed(range(T)):
                next_val_step = next_val if (t == T - 1) else values[t + 1]
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
        
    def update(self, buffer, next_global_state):
        """
        Update Actor and Centralized Critic networks using PPO clipped loss over multiple epochs.

        Argument:
            buffer (MultiAgentRolloutBuffer): Buffer containing on-policy rollout transitions.
            next_global_state (np.ndarray)[global_state_dim]: Global state after rollout horizon.

        Return:
            metrics (dict): Dictionary containing average training losses:
                - 'policy_loss' (float): Average PPO clipped policy loss.
                - 'critic_loss' (float): Average mean-squared error value loss.
                - 'entropy' (float): Average policy distribution entropy.

        Note:
            Parameter Sharing flattens Actor samples into (T * B), whereas Centralized Critic trains on T global states.
        """
        
        b_advantages, b_returns = self.compute_gae(buffer, next_global_state)
        b_obs, b_states, b_act_prb, b_act_pwr, b_log_probs, _, _, _, b_values = buffer.get(self.device)        
        #normalization
        b_advantages = (b_advantages -b_advantages.mean())/ (b_advantages.std() + 1e-8)
        
        #Flatten data : flat T and B 
        T,B =b_obs.shape[0], b_obs.shape[1]
        # [T,B,...] --> [T*B,...]
        flat_obs = b_obs.reshape(-1, b_obs.shape[-1])
        flat_act_prb = b_act_prb.reshape(-1, b_act_prb.shape[-1])
        flat_act_pwr = b_act_pwr.reshape(-1, b_act_pwr.shape[-1])
        flat_logp = b_log_probs.reshape(-1)
        # 
        flat_adv = b_advantages.unsqueeze(1).repeat(1, B).reshape(-1)     
        
        
        
        # define statistical array
        pg_losses = []   # Policy Gradient Loss (L_clip)
        v_losses = []    # Value Loss (L_VF)
        ent_losses = []  # Entropy Loss
        
        total_actor_samples = T*B
        actor_indices= np.arange(total_actor_samples)
        critic_indices=np.arange(T)
        
        # epochs loop
        for epoch in range(self.update_epochs):
            np.random.shuffle(actor_indices)
            np.random.shuffle(critic_indices)

            # UPDATE ACTOR  
            for start in range(0, total_actor_samples, self.batch_size):
                end = start + self.batch_size
                mb_inds = actor_indices[start:end]

                mb_obs = flat_obs[mb_inds]
                mb_act_prb = flat_act_prb[mb_inds]
                mb_act_pwr = flat_act_pwr[mb_inds]
                mb_logp = flat_logp[mb_inds]
                mb_advantages = flat_adv[mb_inds]

                # add to actor network 
                _, _, new_logp, entropy = self.actornetwork(mb_obs, mb_act_prb, mb_act_pwr)
                
                # calculate prob ratio
                ratio = torch.exp(new_logp - mb_logp)

                # ppo CLIP
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # entropy bonus
                entropy_loss = -entropy.mean()
                # total actor loss
                actor_loss = policy_loss + self.ent_coef * entropy_loss

                # Backward and optim
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actornetwork.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                pg_losses.append(policy_loss.item())
                ent_losses.append(entropy.mean().item())

            # UPDATE CRITIC
            for start in range(0, T, self.batch_size):
                end = start + self.batch_size
                mb_inds = critic_indices[start:end]

                mb_states = b_states[mb_inds]
                mb_returns = b_returns[mb_inds]
                new_val = self.criticnetwork(mb_states).squeeze(-1)

                # cal MSE loss
                critic_loss = 0.5 * ((new_val - mb_returns) ** 2).mean()

                # backward and optim
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.criticnetwork.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                v_losses.append(critic_loss.item())
        # CLEAR BUFFER AND RETURN LOG
        buffer.clear()

        return {
            "policy_loss": np.mean(pg_losses),
            "critic_loss": np.mean(v_losses),
            "entropy": np.mean(ent_losses)
        }
