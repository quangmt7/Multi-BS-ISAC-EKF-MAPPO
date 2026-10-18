"""Create training figures from recorded episode metrics."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class ResultArtifacts:
    """Manage metric and figure locations for one training output directory.

    Argument:
        (output_dir) (str | Path): Directory containing training results.

    Return:
        (None).
    """

    def __init__(self, output_dir: str | Path):
        """Store the output paths without changing existing results.

        Argument:
            (output_dir) (str | Path): Directory containing training results.

        Return:
            (None).
        """

        self.output_dir = Path(output_dir)
        self.metrics_path = self.output_dir / "metrics" / "metrics.jsonl"
        self.figures_dir = self.output_dir / "figures"

    def prepare_metrics(self, *, resume: bool) -> Path:
        """Select the metric file and preserve an older root-level metric log.

        Argument:
            (resume) (bool): Whether training resumes from a checkpoint.

        Return:
            (metrics_path) (Path): Metric log inside the metrics directory.
        """

        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path = self.output_dir / "metrics.jsonl"
        if resume and not self.metrics_path.exists() and legacy_path.is_file():
            shutil.copyfile(legacy_path, self.metrics_path)
        return self.metrics_path

    def render_existing(self) -> list[Path]:
        """Generate figures from an existing completed training run.

        Argument:
            (None).

        Return:
            (figure_paths) (list[Path]): Written PNG figure paths.
        """

        if not self.metrics_path.is_file():
            legacy_path = self.output_dir / "metrics.jsonl"
            if not legacy_path.is_file():
                raise FileNotFoundError(
                    f"No training metrics found in {self.output_dir}"
                )
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(legacy_path, self.metrics_path)
        return self.render(self.metrics_path)

    def render(self, metrics_path: str | Path) -> list[Path]:
        """Plot recorded training, communication, sensing and PPO metrics as individual figures.

        Argument:
            (metrics_path) (str | Path): JSON Lines episode metrics file.

        Return:
            (figure_paths) (list[Path]): Written PNG figure paths.
        """

        records = self._read_metrics(Path(metrics_path))
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        output_paths: list[Path] = []

        # 1. Training Return (duy nhất đồ thị này dùng moving mean)
        curve = self._series(records, "episode_return")
        if len(curve[0]) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(*curve, color="#94a3b8", linewidth=0.8, label="Episode return")
            if len(curve[1]) >= 2:
                window = min(50, len(curve[1]))
                smoothed = np.convolve(curve[1], np.ones(window) / window, mode="valid")
                ax.plot(curve[0][window - 1 :], smoothed, color="#2563eb",
                        linewidth=2, label=f"Moving mean ({window} episodes)")
            ax.set(title="Training return", xlabel="Episode", ylabel="Return")
            ax.grid(alpha=0.2)
            ax.legend()
            output_paths.append(self._save(fig, "training_curve.png"))

        # 2. Individual Communication figures (đường nét xanh đơn sắc, auto-scale tự nhiên)
        comm_specs = (
            ("comm_rate_mean_mbps", "Mean UE communication rate", "Mean UE rate (Mbps)", "comm_rate.png"),
            ("qos_satisfaction_rate", "QoS satisfaction rate", "QoS satisfaction rate", "comm_qos_satisfaction.png"),
        )
        for key, title, ylabel, fname in comm_specs:
            p = self._single_figure(records, key, title, ylabel, fname)
            if p:
                output_paths.append(p)

        # 3. Individual Sensing and tracking figures (đường nét xanh đơn sắc)
        sensing_specs = (
            ("sensing_success_prob_mean", "Mean sensing success probability", "Mean sensing success probability", "sensing_success_prob.png"),
            ("aoi_mean", "Mean age of information", "Mean age of information", "sensing_aoi.png"),
            ("pos_error_mean_m", "Mean position error", "Mean position error (m)", "sensing_pos_error.png"),
            ("vel_error_mean_mps", "Mean velocity error", "Mean velocity error (m/s)", "sensing_vel_error.png"),
        )
        for key, title, ylabel, fname in sensing_specs:
            p = self._single_figure(records, key, title, ylabel, fname)
            if p:
                output_paths.append(p)

        # 4. Individual PPO update figures (đường nét xanh đơn sắc)
        ppo_specs = (
            ("policy_loss", "PPO policy loss", "Policy loss", "ppo_policy_loss.png"),
            ("critic_loss", "PPO critic loss", "Critic loss", "ppo_critic_loss.png"),
            ("entropy", "PPO policy entropy", "Policy entropy", "ppo_entropy.png"),
        )
        for key, title, ylabel, fname in ppo_specs:
            p = self._single_figure(records, key, title, ylabel, fname)
            if p:
                output_paths.append(p)

        return output_paths

    @staticmethod
    def _read_metrics(path: Path) -> list[dict[str, Any]]:
        """Read episode records while reporting malformed lines precisely.

        Argument:
            (path) (Path): JSON Lines metrics path.

        Return:
            (records) (list[dict[str, Any]]): Episode metric mappings.
        """

        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
                if not isinstance(record, dict) or not isinstance(record.get("episode"), int):
                    raise ValueError(f"Invalid episode record at {path}:{line_number}")
                records.append(record)
        if not records:
            raise ValueError(f"No episode metrics found in {path}")
        return records

    @staticmethod
    def _series(records: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract finite scalar values and their episode numbers.

        Argument:
            (records) (list[dict[str, Any]]): Episode metric mappings.
            (key) (str): Metric name to extract.

        Return:
            (series) (tuple[np.ndarray, np.ndarray]): Episodes and values.
        """

        points = []
        for record in records:
            value = record.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
                points.append((record["episode"], float(value)))
        if not points:
            return np.array([], dtype=int), np.array([], dtype=float)
        return np.asarray(points, dtype=float).T

    def _single_figure(
        self,
        records: list[dict[str, Any]],
        key: str,
        title: str,
        ylabel: str,
        filename: str,
    ) -> Path | None:
        """Draw a single metric with a solid blue line matching the original style.

        Argument:
            (records) (list[dict[str, Any]]): Episode metric mappings.
            (key) (str): Metric key to extract.
            (title) (str): Title of the plot.
            (ylabel) (str): Label for Y-axis.
            (filename) (str): Target filename.

        Return:
            (path) (Path | None): Written PNG path, or None if no points.
        """

        episodes, values = self._series(records, key)
        if len(episodes) == 0:
            return None

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.plot(episodes, values, color="#2563eb", linewidth=1.2)
        ax.set(title=title, xlabel="Episode", ylabel=ylabel)
        ax.grid(alpha=0.2)
        return self._save(fig, filename)

    def _save(self, fig: Any, filename: str) -> Path:
        """Write and close one figure.

        Argument:
            (fig) (matplotlib.figure.Figure): Figure to write.
            (filename) (str): Filename under the figures directory.

        Return:
            (path) (Path): Written PNG path.
        """

        path = self.figures_dir / filename
        try:
            fig.tight_layout()
            fig.savefig(path, dpi=160)
        finally:
            plt.close(fig)
        return path
