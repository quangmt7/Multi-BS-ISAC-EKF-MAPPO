"""Train and evaluate the three allocation methods on paired scenario seeds."""

from pathlib import Path
import argparse
import copy
import json
import subprocess
import numpy as np
import torch
from scipy.stats import t as student_t
from ppo.agents.allocation import sac_ratios
from ppo.envs.PPOenv import PPOISACEnv
from ppo.envs.HIERenv import HierISACEnv
from ppo.utils.project_config import ProjectConfig
from ppo.utils.mappo_trainer import MAPPOTrainer
from ppo.utils.hier_trainer import HierarchicalISACTrainer
from benchmark.tracking_benchmark import TrackingReference


METHODS = ("mappo", "hierarchical", "hierarchical_single_bwp")
METRICS = (
    "mean_embb_rate_mbps",
    "qos_eMBB_rate",
    "qos_URLLC_rate",
    "qos_satisfaction_rate",
    "sensing_success_prob_mean",
    "mean_urllc_latency_ms",
    "aoi_mean",
    "pos_error_mean_m",
    "vel_error_mean_mps",
    "u_pos_mean",
    "u_vel_mean",
    "energy_cost",
    "switch_cost",
    "guardband_cost",
    "conditional_pcrb_pos",
    "conditional_pcrb_vel",
    "episode_length",
    "terminal_failure",
)


def evaluate(config, method, agent, seeds, steps):
    """Frozen deterministic policies; metrics include duration and terminal failures.

    PHY time series share scenario/channel seeds and keyed measurement streams.
    Covariance references are conditional on each policy's actual observations.
    """
    env = (
        PPOISACEnv(config)
        if method == "mappo"
        else HierISACEnv(config, bwp_mode=method != "hierarchical_single_bwp")
    )
    env.max_steps = steps
    rows = []
    for seed in seeds:
        local, state = env.reset(seed=seed)
        reference = TrackingReference(env)
        totals = {name: 0.0 for name in METRICS}
        terminated = False
        completed_packets, latency_sum = 0, 0.0
        for step in range(steps):
            if method == "mappo":
                action = agent.select_action(local, state, deterministic=True)[0]
            else:
                sac, dqn = agent
                if env._frame_steps is None:
                    raw = sac.select_action(env.get_sac_state(), deterministic=True)
                    length = min(
                        config["system_parameters"]["simulation"][
                            "num_subframes_per_frame"
                        ],
                        steps - step,
                    )
                    env.begin_frame(
                        sac_ratios(
                            raw, env.BWP_slice.shape, method.endswith("single_bwp")
                        ),
                        num_subframes=length,
                    )
                action = dqn.select_action(
                    env.get_dqn_state(),
                    budget=env.budget,
                    bwp_mode=not method.endswith("single_bwp"),
                    deterministic=True,
                )
            local, state, _, terminated, truncated, info = env.step(action)
            reference.update(env)
            p, v = reference.pcrb.get_uncer()
            info.update(
                conditional_pcrb_pos=float(np.mean(p)),
                conditional_pcrb_vel=float(np.mean(v)),
            )
            completed_packets += info["urllc_completed_packets"]
            latency_sum += info["urllc_completed_latency_sum_ms"]
            for name in METRICS[:-2]:
                if name != "mean_urllc_latency_ms":
                    totals[name] += float(info[name])
            if terminated or truncated:
                break
        length = step + 1
        row = {
            name: total / length
            for name, total in totals.items()
            if name not in ("episode_length", "terminal_failure")
        }
        row.update(
            episode_length=length,
            terminal_failure=float(terminated),
            eval_seed=seed,
            mean_urllc_latency_ms=latency_sum / completed_packets
            if completed_packets
            else None,
            urllc_completed_packets=completed_packets,
        )
        rows.append(row)
    return rows


def summarize(records):
    """CI replicates are independent training seeds, after averaging eval seeds."""
    summary = {}
    for method in METHODS:
        selected = [row for row in records if row["method"] == method]
        seeds = sorted({row["train_seed"] for row in selected})
        summary[method] = {}
        for metric in METRICS:
            seed_means = []
            for seed in seeds:
                observations = [
                    row[metric]
                    for row in selected
                    if row["train_seed"] == seed and row[metric] is not None
                ]
                if observations:
                    if metric == "mean_urllc_latency_ms":
                        finished = [
                            row
                            for row in selected
                            if row["train_seed"] == seed and row[metric] is not None
                        ]
                        packets = sum(
                            row["urllc_completed_packets"] for row in finished
                        )
                        seed_means.append(
                            sum(
                                row[metric] * row["urllc_completed_packets"]
                                for row in finished
                            )
                            / packets
                        )
                    else:
                        seed_means.append(np.mean(observations))
            values = np.asarray(seed_means, dtype=float)
            mean = float(values.mean()) if len(values) else None
            interval = None
            if len(values) > 1:
                half = float(
                    student_t.ppf(0.975, len(values) - 1)
                    * values.std(ddof=1)
                    / np.sqrt(len(values))
                )
                interval = [mean - half, mean + half]
            summary[method][metric] = {
                "mean": mean,
                "ci95": interval,
                "training_seeds": len(values),
                "training_seeds_total": len(seeds),
            }
    return summary


def run_comparison(config, *, episodes, steps, seeds, eval_seeds, output_dir):
    if (
        not seeds
        or not eval_seeds
        or len(set(seeds)) != len(seeds)
        or len(set(eval_seeds)) != len(eval_seeds)
    ):
        raise ValueError("Provide non-empty unique training and evaluation seeds")
    if set(seeds) & set(eval_seeds):
        raise ValueError("Evaluation seeds must be held out from training seeds")
    if any(
        isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 2**32
        for v in (*seeds, *eval_seeds)
    ):
        raise ValueError("Seeds must be integers in [0,2**32)")
    if episodes <= 0 or steps <= 0:
        raise ValueError("episodes and steps must be positive")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Comparison output directory is not empty; choose a new directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    base = copy.deepcopy(config)
    base["comparison_seeds"] = dict(training=list(seeds), evaluation=list(eval_seeds))
    base["system_parameters"]["simulation"]["num_slots_per_episode"] = steps
    base["mappo_parameters"]["training"].update(
        total_episodes=episodes, episode_length=steps, total_steps=episodes * steps
    )
    hier = base["hierarchical_parameters"]
    hier.get("training", hier["timescale"]).update(
        total_episodes=episodes, episode_length=steps, total_steps=episodes * steps
    )
    records = []
    for seed in seeds:
        for method in METHODS:
            cfg = copy.deepcopy(base)
            cfg["mappo_parameters"]["training"]["seed"] = seed
            cfg["hierarchical_parameters"].get(
                "training", cfg["hierarchical_parameters"]["timescale"]
            )["seed"] = seed
            run_dir = output / f"seed_{seed}" / method
            if method == "mappo":
                trainer = MAPPOTrainer(cfg, output_dir=run_dir)
                training_history = trainer.run()
                agents = trainer.agent
                updates = {
                    "ppo_rollout_updates": sum(
                        "policy_loss" in ep for ep in training_history
                    )
                }
            else:
                trainer = HierarchicalISACTrainer(
                    cfg,
                    output_dir=run_dir,
                    single_bwp_mode=method.endswith("single_bwp"),
                )
                training_history = trainer.run()
                agents = (trainer.sac_agent, trainer.dqn_agent)
                updates = dict(
                    sac_actor=sum(
                        ep["episode_sac_actor_updates"] for ep in training_history
                    ),
                    sac_critic=trainer.sac_agent.update_step,
                    dqn=trainer.dqn_agent.update_step,
                )
            for row in evaluate(cfg, method, agents, eval_seeds, steps):
                records.append(
                    dict(
                        row,
                        method=method,
                        train_seed=seed,
                        training_steps=training_history[-1]["global_step"],
                        training_updates=updates,
                    )
                )
    summary = summarize(records)
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    result = dict(
        config=base,
        commit=revision,
        dirty=dirty,
        torch_version=torch.__version__,
        training_seeds=seeds,
        evaluation_seeds=eval_seeds,
        records=records,
        summary=summary,
        ci_unit="independent training seed, averaged over evaluation seeds",
        metrics_note="Step means over observed episode duration; interpret with terminal_failure and episode_length",
        reference="Conditional covariance recursion per allocation policy; not a common optimal policy",
    )
    (output / "comparison.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    render_summary(summary, output)
    return result


def render_summary(summary, output):
    import matplotlib.pyplot as plt

    for metric in (
        "qos_eMBB_rate",
        "qos_URLLC_rate",
        "pos_error_mean_m",
        "aoi_mean",
        "energy_cost",
        "terminal_failure",
    ):
        values = [summary[name][metric]["mean"] for name in METHODS]
        errors = [
            0.0
            if summary[name][metric]["ci95"] is None
            else (summary[name][metric]["ci95"][1] - summary[name][metric]["ci95"][0])
            / 2
            for name in METHODS
        ]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(METHODS, values, yerr=errors, capsize=4)
        ax.set(ylabel=metric, title=f"Held-out evaluation: {metric}")
        fig.tight_layout()
        fig.savefig(Path(output) / f"compare_{metric}.png", dpi=160)
        plt.close(fig)


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train and evaluate MAPPO, hierarchical multi/single BWP"
    )
    parser.add_argument("--config-dir", type=Path, default=root / "config")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[1042, 1043])
    parser.add_argument(
        "--output-dir", type=Path, default=root / "results" / "comparison"
    )
    args = parser.parse_args(argv)
    config = ProjectConfig.from_directory(args.config_dir, mode="hierarchical").data
    run_comparison(
        config,
        episodes=args.episodes,
        steps=args.episode_length,
        seeds=args.seeds,
        eval_seeds=args.eval_seeds,
        output_dir=args.output_dir,
    )
