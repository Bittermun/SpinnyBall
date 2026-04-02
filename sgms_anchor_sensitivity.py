"""
Sobol sensitivity analysis for the reduced-order dynamic-anchor model.

This module evaluates analytical anchor outputs from ``sgms_anchor_v1.py`` and
uses SALib to quantify parameter importance without running the ODE solver.
That keeps paper-scale sweeps fast and deterministic.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from sgms_anchor_v1 import DEFAULT_PARAMS, analytical_metrics


DEFAULT_PROBLEM = {
    "num_vars": 4,
    "names": ["u", "g_gain", "eps", "lam"],
    "bounds": [
        [5.0, 520.0],
        [0.02, 0.2],
        [0.0, 1e-3],
        [0.1, 2.0],
    ],
}

DEFAULT_OUTPUTS = (
    "force_per_stream_n",
    "k_eff",
    "period_s",
    "static_offset_m",
    "packet_rate_hz",
)


def _copy_params(base_params: dict | None = None) -> dict:
    params = DEFAULT_PARAMS.copy()
    if base_params is not None:
        params.update(base_params)
    return params


def evaluate_parameter_vector(vector: np.ndarray, base_params: dict | None = None) -> dict:
    params = _copy_params(base_params)
    u, g_gain, eps, lam = [float(v) for v in vector]
    params.update({"u": u, "g_gain": g_gain, "eps": eps, "lam": lam})
    return analytical_metrics(params)


def sample_anchor_problem(
    problem: dict | None = None,
    N: int = 256,
    calc_second_order: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    problem = DEFAULT_PROBLEM if problem is None else problem
    return sobol_sample.sample(
        problem,
        N,
        calc_second_order=calc_second_order,
        scramble=True,
        seed=seed,
    )


def evaluate_sample_matrix(
    samples: np.ndarray,
    outputs: tuple[str, ...] = DEFAULT_OUTPUTS,
    base_params: dict | None = None,
) -> dict[str, np.ndarray]:
    values = {output: np.empty(samples.shape[0]) for output in outputs}
    for i, sample in enumerate(samples):
        metrics = evaluate_parameter_vector(sample, base_params=base_params)
        for output in outputs:
            values[output][i] = metrics[output]
    return values


def run_sobol_sensitivity(
    problem: dict | None = None,
    N: int = 256,
    outputs: tuple[str, ...] = DEFAULT_OUTPUTS,
    calc_second_order: bool = False,
    base_params: dict | None = None,
    seed: int | None = None,
) -> dict:
    problem = DEFAULT_PROBLEM if problem is None else problem
    samples = sample_anchor_problem(problem, N=N, calc_second_order=calc_second_order, seed=seed)
    values = evaluate_sample_matrix(samples, outputs=outputs, base_params=base_params)

    indices = {}
    for output in outputs:
        indices[output] = sobol_analyze.analyze(
            problem,
            values[output],
            calc_second_order=calc_second_order,
            print_to_console=False,
            seed=seed,
        )

    return {
        "problem": problem,
        "samples": samples,
        "outputs": values,
        "indices": indices,
        "calc_second_order": calc_second_order,
    }


def export_sobol_indices_csv(indices: dict, names: list[str], filename: str | Path) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for output, data in indices.items():
        st = np.asarray(data["ST"])
        s1 = np.asarray(data["S1"])
        for i, name in enumerate(names):
            rows.append(
                {
                    "output": output,
                    "parameter": name,
                    "S1": float(s1[i]),
                    "ST": float(st[i]),
                }
            )

    with filename.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["output", "parameter", "S1", "ST"])
        writer.writeheader()
        writer.writerows(rows)


def plot_sobol_indices(indices: dict, names: list[str], filename: str = "sgms_anchor_sobol.png") -> None:
    outputs = list(indices.keys())
    fig, axes = plt.subplots(len(outputs), 1, figsize=(10, 3.4 * len(outputs)), squeeze=False)

    for ax, output in zip(axes.ravel(), outputs):
        data = indices[output]
        x = np.arange(len(names))
        width = 0.38
        ax.bar(x - width / 2, data["S1"], width=width, label="S1", color="#79c0ff")
        ax.bar(x + width / 2, data["ST"], width=width, label="ST", color="#7ee787")
        ax.set_xticks(x, names)
        ax.set_ylim(bottom=min(0.0, np.min(data["S1"]) - 0.05), top=max(1.0, np.max(data["ST"]) + 0.05))
        ax.set_ylabel("Index")
        ax.set_title(f"Sobol Indices: {output}")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def print_sensitivity_summary(result: dict) -> None:
    print("=== ANCHOR SOBOL SENSITIVITY ===")
    names = result["problem"]["names"]
    for output, data in result["indices"].items():
        st = np.asarray(data["ST"])
        top_idx = int(np.nanargmax(st))
        print(f"{output}: dominant ST = {names[top_idx]} ({st[top_idx]:.4f})")


def main() -> None:
    result = run_sobol_sensitivity(
        N=512,
        outputs=("k_eff", "period_s", "static_offset_m", "packet_rate_hz"),
        calc_second_order=False,
        seed=11,
    )
    export_sobol_indices_csv(result["indices"], result["problem"]["names"], "sgms_anchor_sobol.csv")
    plot_sobol_indices(result["indices"], result["problem"]["names"])
    print_sensitivity_summary(result)


if __name__ == "__main__":
    main()
