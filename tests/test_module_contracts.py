"""Regression checks for the current BWP configuration and PHY interfaces."""
from pathlib import Path
import unittest

import numpy as np
import yaml

from channel.channel import WirelessChannel
from communication.communication import CommunicationManager
from ppo.utils.project_config import ProjectConfig


def configuration():
    config = {}
    for path in (Path(__file__).resolve().parents[1] / "config").glob("*.yaml"):
        config.update(yaml.safe_load(path.read_text(encoding="utf-8")))
    config["system_parameters"].update(
        num_bs=3, num_target=4, num_embb=1, num_urllc=1,
        num_ue_1embb=2, num_ue_1urllc=3, nPrb=[2, 3],
        cost_switch=0.3, cost_gb=2, urllc_packet_size_bits=256,
    )
    return config


class ModuleContractTests(unittest.TestCase):
    def test_current_config_does_not_require_removed_flat_dimensions(self):
        ProjectConfig(configuration()).validate()

    def test_packet_size_must_be_explicit_and_bwp_arrays_must_align(self):
        config = configuration()
        config["system_parameters"]["urllc_packet_size_bits"] = None
        with self.assertRaisesRegex(ValueError, "urllc_packet_size_bits"):
            ProjectConfig(config).validate()
        config = configuration()
        config["system_parameters"]["nPrb"] = [2]
        with self.assertRaisesRegex(ValueError, "nPrb"):
            ProjectConfig(config).validate()

    def test_runtime_contract_derives_flat_dimensions_from_current_config(self):
        from ppo.utils.runtime_contract import RuntimeContract
        contract = RuntimeContract(configuration())
        self.assertEqual(contract.num_prb, 5)
        self.assertEqual(contract.max_destination, 9)

    def test_bandwidth_is_hz_and_urllc_duration_is_seconds(self):
        comm = CommunicationManager(configuration(), bs_id=0)
        allocation = np.zeros((3, 2, 1, 3, 3))
        allocation[0, 0, 0, 0, 0] = 1
        efficiency = np.ones_like(allocation)
        rate = comm.compute_embb_throughput(allocation, efficiency)
        bits = comm.compute_urllc_bits(allocation, efficiency, 0.01)
        self.assertAlmostEqual(float(rate[0, 0]), 360000)
        self.assertAlmostEqual(float(bits[0, 0]), 3600)
        _, satisfied, metrics = comm.evaluate_qos(np.array([[500000.]]), np.array([[0.]]))
        self.assertFalse(satisfied)
        self.assertAlmostEqual(metrics["mean_embb_rate_mbps"], 0.5)

    def test_channel_uses_same_echo_realization_and_preserves_bwp(self):
        config = configuration()
        config["system_parameters"]["channel_and_interference"]["shadowing_std_db"] = 0
        channel = WirelessChannel(config)
        x_c = np.zeros((3, 2, 2, 3, 3))
        x_s = np.zeros((3, 2, 4, 3))
        x_s[0, 1, 0, 0] = 1
        bs = np.array([[0., 0., 25.], [100., 0., 25.], [0., 100., 25.]])
        targets = np.array([[20., 20., 30.], [40., 20., 30.], [20., 40., 30.], [40., 40., 30.]])
        result = channel.step(x_c, x_s, bs, np.full((2, 3, 3), 10.), targets)
        self.assertEqual(result["effective_sensing_sinr"].shape, (3, 4))
        self.assertEqual(result["sensing_sinr_prb"].shape, (3, 2, 4, 3))
        # No other transmitter or communication allocation is active.
        expected = result["sensing_gain"][0, 1, 0, 0] * channel.pEach[1] / channel.noise_power
        self.assertAlmostEqual(float(result["sensing_sinr_prb"][0, 1, 0, 0]) / expected, 1., places=5)
        np.testing.assert_array_equal(result["sensing_gain"][:, 0, :, 2], 0)


if __name__ == "__main__":
    unittest.main()
