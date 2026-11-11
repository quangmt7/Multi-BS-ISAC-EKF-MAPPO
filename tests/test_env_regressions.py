"""Regression checks for the BWP environment kernel (no training required)."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def configuration():
    """Small unequal-BWP/UE fixture; values here are test data, not defaults."""
    config = {}
    for filename in ("system.yaml", "ekf.yaml", "mappo.yaml"):
        with (ROOT / "config" / filename).open(encoding="utf-8") as stream:
            config.update(yaml.safe_load(stream))
    config["system_parameters"].update(
        num_bs=3,
        num_embb=1,
        num_urllc=1,
        num_ue_1embb=2,
        num_ue_1urllc=3,
        num_target=4,
        bwp=[1, 3],
        nPrb=[2, 3],
        pEach=[5, 5],
        fStart=[0, 10],
        cost_gb=1.0,
        cost_switch=1.0,
        urllc_packet_size_bits=256,
    )
    config["system_parameters"]["simulation"]["num_slots_per_episode"] = 8
    return config


def environment(config=None, hierarchical=False, **kwargs):
    # Keep imports inside the helper so the missing optional plotting dependency
    # is reported as an environment regression, rather than hiding collection.
    if hierarchical:
        from ppo.envs.HIERenv import HierISACEnv

        env_class = HierISACEnv
    else:
        from ppo.envs.PPOenv import PPOISACEnv

        env_class = PPOISACEnv
    return env_class(configuration() if config is None else config, **kwargs)


def idle(env):
    return np.zeros((env.num_bs, len(env.bwp), max(env.nPrb)), dtype=np.int64)


class EnvironmentRegressionTests(unittest.TestCase):
    def test_trainer_uses_current_env_and_rejects_legacy_learned_power(self):
        from types import SimpleNamespace
        from ppo.utils.mappo_trainer import MAPPOTrainer
        from ppo.envs.PPOenv import PPOISACEnv
        trainer = MAPPOTrainer(configuration())
        env = trainer._create_environment(bwp_mode=True)
        self.assertIsInstance(env, PPOISACEnv)
        trainer.env = env
        trainer.agent = SimpleNamespace(actornetwork=SimpleNamespace(power_mean=object()))
        trainer.buffer_factory = list
        with self.assertRaisesRegex(RuntimeError, "fixed pEach power"):
            trainer.run()

    def test_reset_and_idle_step_work_for_both_subclasses(self):
        for hierarchical in (False, True):
            with self.subTest(hierarchical=hierarchical):
                env = environment(hierarchical=hierarchical)
                observations, state = env.reset(seed=7)
                self.assertEqual(np.asarray(observations).shape[0], 3)
                self.assertTrue(np.isfinite(observations).all())
                self.assertTrue(np.isfinite(state).all())
                result = env.step(idle(env))
                self.assertEqual(len(result), 6)
                self.assertIsInstance(result[2], float)
                self.assertTrue(np.isfinite(result[2]))
                self.assertEqual(result[5]["energy_cost"], 0.0)

    def test_mapping_preserves_global_ue_identity_and_excludes_padding(self):
        env = environment()
        env.reset(seed=7)
        actions = idle(env)
        actions[0, 0, 0] = 1  # first target
        actions[1, 1, 1] = 5  # first eMBB UE, after four targets
        actions[2, 1, 2] = 9  # final URLLC UE (global UE index four)
        x_c, x_s = env.action_mapping(actions)
        self.assertEqual(x_c.shape, (3, 2, 2, 3, 3))
        self.assertEqual(x_s.shape, (3, 2, 4, 3))
        self.assertEqual(x_s[0, 0, 0, 0], 1)
        self.assertEqual(x_c[1, 1, 0, 0, 1], 1)
        self.assertEqual(x_c[2, 1, 1, 2, 2], 1)
        self.assertEqual(np.sum(x_c), 2)
        self.assertEqual(np.sum(x_s), 1)
        self.assertFalse(x_c[:, :, 0, 2].any())  # padded eMBB UE
        self.assertFalse(x_c[:, 0, :, :, 2].any())  # padded PRB
        self.assertFalse(x_s[:, 0, :, 2].any())

    def test_invalid_actions_fail_before_mutating_the_environment(self):
        env = environment()
        env.reset(seed=7)
        invalid_actions = []
        padding = idle(env)
        padding[0, 0, 2] = 1
        invalid_actions.append(padding)
        out_of_range = idle(env)
        out_of_range[0, 1, 0] = 10
        invalid_actions.append(out_of_range)
        fractional = idle(env).astype(float)
        fractional[0, 0, 0] = 0.5
        invalid_actions.append(fractional)
        for actions in invalid_actions:
            with self.subTest(actions=actions.tolist()):
                with self.assertRaises((ValueError, TypeError)):
                    env.step(actions)
                self.assertEqual(env.current_step, 0)

    def test_single_bwp_rejects_one_service_using_two_bwps(self):
        env = environment(bwp_mode=False)
        env.reset(seed=7)
        actions = idle(env)
        actions[0, 0, 0] = 1
        actions[0, 1, 0] = 1
        with self.assertRaises(ValueError):
            env.action_mapping(actions)

    def test_embb_throughput_is_instantaneous(self):
        env = environment()
        env.reset(seed=7)
        actions = idle(env)
        actions[0, 0, 0] = 5
        env.step(actions)
        self.assertGreater(env.eMBB_Thr[0, 0], 0.0)
        env.step(idle(env))
        np.testing.assert_array_equal(env.eMBB_Thr, 0.0)

    def test_flat_mapping_fixed_power_and_global_energy_agree(self):
        env = environment()
        env.reset(seed=7)
        actions = np.zeros((3, 5), dtype=np.int64)
        actions[0, [0, 2]] = [1, 5]  # one real PRB in each BWP
        x_c, x_s = env.action_mapping(actions)
        np.testing.assert_allclose(env.n_res, [3, 5, 5])
        np.testing.assert_allclose(env.p_res, [0.015, 0.025, 0.025])
        self.assertAlmostEqual(env.calculateEnergy(x_c, x_s), 2 * 0.005 * 0.01)
        info = env.step(actions)[5]
        self.assertAlmostEqual(info["energy_cost"], 0.0001)
        np.testing.assert_allclose(info["cost_by_bs"].sum(axis=0), info["cost"])

    def test_budget_is_enforced_before_advancing_time_and_cleared_on_reset(self):
        env = environment(hierarchical=True)
        env.reset(seed=7)
        budget = np.zeros_like(env.BWP_slice, dtype=np.int64)
        budget[0, 0, env.num_slices] = 1  # first target
        env.set_budget(budget)
        budget[:] = 0  # caller changes must not change the held budget
        actions = idle(env)
        actions[0, 0, 0] = 1
        env.step(actions)
        actions[0, 0, 1] = 1
        with self.assertRaisesRegex(ValueError, "budget"):
            env.step(actions)
        self.assertEqual(env.current_step, 1)
        env.reset(seed=7)
        env.step(actions)  # reset removed old caps

    def test_active_sensing_logs_posterior_error_and_keeps_covariance_psd(self):
        env = environment()
        env.reset(seed=7)
        actions = idle(env)
        actions[:, 0, :2] = 1
        actions[:, 1, :] = 1
        with patch.object(env.detection_list[0], "detect", return_value=(True, 1.0)):
            info = env.step(actions)[5]
        self.assertTrue(env.detect_mask[0, 0])
        self.assertEqual(env.aoi[0], 0)
        posterior = np.asarray(env.ekf_common.state_list)
        expected_error = np.linalg.norm(posterior[:, :3] - env.target_states[:, :3], axis=1).mean()
        self.assertAlmostEqual(info["pos_error_mean_m"], expected_error)
        for covariance in env.ekf_common.covariance_list:
            self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-10)

    def test_reset_clears_traffic_and_bwp_history(self):
        env = environment()
        env.reset(seed=7)
        env.numBit[:] = 100
        env.URLLC_Latency[:] = 20
        env.eMBB_Thr[:] = 1e6
        env.BWP_slice[:] = 1
        env.last_BWP_slice[:] = 1
        env.reset(seed=8)
        for values in (
            env.numBit, env.URLLC_Latency, env.eMBB_Thr,
            env.BWP_slice, env.last_BWP_slice,
        ):
            np.testing.assert_array_equal(values, 0)
        self.assertEqual(env.current_step, 0)

    def test_switch_cost_charges_only_new_activation(self):
        env = environment()
        env.reset(seed=7)
        actions = idle(env)
        actions[0, 0, 0] = 5
        first = env.step(actions)[5]["switch_cost"]
        same = env.step(actions)[5]["switch_cost"]
        off = env.step(idle(env))[5]["switch_cost"]
        again = env.step(actions)[5]["switch_cost"]
        self.assertEqual(first, env.cost_switch)
        self.assertEqual(same, 0.0)
        self.assertEqual(off, 0.0)
        self.assertEqual(again, first)

    def test_guard_cost_counts_an_unordered_numerology_pair_once(self):
        env = environment()
        env.reset(seed=7)
        env.BWP_slice[:] = 0
        env.BWP_slice[0, :, 0] = 1
        # One service on numerologies one and three: ceil(|1 - 3|) = 2.
        self.assertEqual(env.calculateGuardBand(), 2.0)
        self.assertEqual(env.calculateFrag(), 4.0)

    def test_single_physical_bwp_has_finite_reward_with_zero_guard_bound(self):
        config = configuration()
        config["system_parameters"].update(
            bwp=[1], nPrb=[2], pEach=[5], fStart=[0]
        )
        env = environment(config=config)
        env.reset(seed=7)
        result = env.step(idle(env))
        self.assertEqual(env.calculateGuardBand(), 0.0)
        self.assertTrue(np.isfinite(result[2]))

    def test_reward_reduction_is_scalar_with_unequal_service_ue_counts(self):
        env = environment()
        env.reset(seed=7)
        args = dict(
            user_rates=np.full((1, 2), 1e6),
            user_delay=np.zeros((1, 3)),
            p_suc=np.ones(4),
            u_pos=np.ones(4),
            u_vel=np.ones(4),
        )
        cheap = env._compute_reward(cost=np.zeros(4), **args)[0]
        expensive = env._compute_reward(cost=np.asarray(env.cost_max), **args)[0]
        self.assertIsInstance(cheap, float)
        self.assertTrue(np.isfinite(cheap))
        self.assertGreater(cheap, expensive)

    def test_urllc_packet_completion_keeps_latency_and_starts_next_packet_later(self):
        env = environment()
        env.reset(seed=7)
        bits_per_step = np.full((1, 3), 128.0)
        # Control only the received-bit source; traffic, time, EKF and state
        # transitions remain the real implementations.
        with patch.object(
            env.comm_managers[0], "compute_urllc_bits", return_value=bits_per_step
        ):
            env.step(idle(env))
            np.testing.assert_allclose(env.numBit, 128.0)
            completed = env.step(idle(env))[5]["urllc_completed_latency_ms"]
            np.testing.assert_allclose(completed, 2 * env.slot_duration * 1000)
            np.testing.assert_array_equal(env.numBit, 0.0)
            env.step(idle(env))
            np.testing.assert_allclose(env.numBit, 128.0)
            np.testing.assert_allclose(env.URLLC_Latency, env.slot_duration * 1000)

    def test_seed_reproduces_prior_and_transition(self):
        env = environment()
        first_obs, first_state = env.reset(seed=123)
        first_prior = np.asarray(env.ekf_common.state_list).copy()
        first = env.step(idle(env))
        second_obs, second_state = env.reset(seed=123)
        np.testing.assert_array_equal(first_obs, second_obs)
        np.testing.assert_array_equal(first_state, second_state)
        np.testing.assert_array_equal(first_prior, env.ekf_common.state_list)
        second = env.step(idle(env))
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2:5], second[2:5])

    def test_prior_uses_full_3d_covariance_and_is_not_ground_truth(self):
        env = environment()
        env.reset(seed=7)
        initial_covariance = np.diag(
            env.config["ekf_parameters"]["initial_covariance"]["p_diag_init"]
        )
        u_pos, u_vel = env._get_uncertainty_summary()
        np.testing.assert_allclose(u_pos, np.trace(initial_covariance[:3, :3]))
        np.testing.assert_allclose(u_vel, np.trace(initial_covariance[3:, 3:]))
        self.assertFalse(np.array_equal(env.ekf_common.state_list, env.target_states))
        for covariance in env.ekf_common.covariance_list:
            np.testing.assert_array_equal(covariance, initial_covariance)


if __name__ == "__main__":
    unittest.main()
