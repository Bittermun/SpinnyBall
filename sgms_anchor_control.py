"""
State-space and LQR analysis for the reduced-order dynamic-anchor model.

This module intentionally builds on ``sgms_anchor_v1.py`` instead of defining a
second anchor model. The plant here is the small-signal linearization of the
existing continuum anchor dynamics:

    m_s * x_ddot + c_damp * x_dot + k_eff * x = u_force

where ``k_eff`` comes directly from the reduced-order anchor force law.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import control as ct
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sgms_anchor_v1 import DEFAULT_PARAMS, analytical_metrics


DEFAULT_Q = np.diag([100.0, 1.0])
DEFAULT_R = np.array([[0.01]])


def _copy_params(params: dict | None = None) -> dict:
    merged = DEFAULT_PARAMS.copy()
    if params is not None:
        merged.update(params)
    return merged


def build_state_space(params: dict | None = None) -> dict:
    params = _copy_params(params)
    metrics = analytical_metrics(params)

    A = np.array(
        [
            [0.0, 1.0],
            [-metrics["k_eff"] / params["ms"], -params["c_damp"] / params["ms"]],
        ]
    )
    B = np.array([[0.0], [1.0 / params["ms"]]])
    C = np.eye(2)
    D = np.zeros((2, 1))
    system = ct.ss(A, B, C, D)

    return {
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "system": system,
        "metrics": metrics,
        "params": params,
    }


def design_lqr(params: dict | None = None, Q: np.ndarray | None = None, R: np.ndarray | None = None) -> dict:
    model = build_state_space(params)
    Q = DEFAULT_Q.copy() if Q is None else np.asarray(Q, dtype=float)
    R = DEFAULT_R.copy() if R is None else np.asarray(R, dtype=float)

    K, S, E = ct.lqr(model["system"], Q, R)
    K = np.asarray(K, dtype=float)
    S = np.asarray(S, dtype=float)
    E = np.asarray(E, dtype=complex)

    A_cl = model["A"] - model["B"] @ K
    closed_loop = ct.ss(A_cl, model["B"], model["C"], model["D"])

    return {
        **model,
        "Q": Q,
        "R": R,
        "K": K,
        "riccati_S": S,
        "closed_loop_poles": E,
        "A_cl": A_cl,
        "closed_loop_system": closed_loop,
    }


def build_proportional_gain(params: dict | None = None, gain_scale: float = 1.0) -> np.ndarray:
    params = _copy_params(params)
    metrics = analytical_metrics(params)
    return np.array([[gain_scale * metrics["k_eff"], 0.0]])


def simulate_open_closed_loop(
    params: dict | None = None,
    t_eval: np.ndarray | None = None,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    disturbance_force: np.ndarray | None = None,
) -> dict:
    design = design_lqr(params, Q=Q, R=R)
    params = design["params"]

    if t_eval is None:
        t_eval = np.linspace(0.0, params["t_max"], 4000)
    else:
        t_eval = np.asarray(t_eval, dtype=float)

    if disturbance_force is None:
        disturbance_force = np.zeros_like(t_eval)
    else:
        disturbance_force = np.asarray(disturbance_force, dtype=float)
        if disturbance_force.shape != t_eval.shape:
            raise ValueError("disturbance_force must have the same shape as t_eval")

    x0 = np.array([params["x0"], params["v0"]], dtype=float)

    open_resp = ct.forced_response(design["system"], T=t_eval, U=disturbance_force, X0=x0, return_x=True)
    closed_resp = ct.forced_response(
        design["closed_loop_system"],
        T=t_eval,
        U=disturbance_force,
        X0=x0,
        return_x=True,
    )

    t_open, y_open, x_open = open_resp
    t_closed, y_closed, x_closed = closed_resp
    control_force = -(design["K"] @ x_closed).ravel()

    return {
        "t": np.asarray(t_open),
        "open_y": np.asarray(y_open),
        "closed_y": np.asarray(y_closed),
        "open_state": np.asarray(x_open),
        "closed_state": np.asarray(x_closed),
        "open_x": np.asarray(y_open)[0],
        "closed_x": np.asarray(y_closed)[0],
        "open_v": np.asarray(y_open)[1],
        "closed_v": np.asarray(y_closed)[1],
        "control_force": control_force,
        "disturbance_force": disturbance_force,
        "design": design,
    }


def simulate_controller(
    controller: str,
    params: dict | None = None,
    t_eval: np.ndarray | None = None,
    disturbance_force: np.ndarray | None = None,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    p_gain_scale: float = 1.0,
) -> dict:
    model = build_state_space(params)
    params = model["params"]

    if t_eval is None:
        t_eval = np.linspace(0.0, params["t_max"], 4000)
    else:
        t_eval = np.asarray(t_eval, dtype=float)

    if disturbance_force is None:
        disturbance_force = np.zeros_like(t_eval)
    else:
        disturbance_force = np.asarray(disturbance_force, dtype=float)

    x0 = np.array([params["x0"], params["v0"]], dtype=float)

    if controller == "open":
        K = np.array([[0.0, 0.0]])
        A_cl = model["A"]
        system = model["system"]
        poles = np.linalg.eigvals(A_cl)
    elif controller == "p":
        K = build_proportional_gain(params, gain_scale=p_gain_scale)
        A_cl = model["A"] - model["B"] @ K
        system = ct.ss(A_cl, model["B"], model["C"], model["D"])
        poles = np.linalg.eigvals(A_cl)
    elif controller == "lqr":
        design = design_lqr(params, Q=Q, R=R)
        K = design["K"]
        A_cl = design["A_cl"]
        system = design["closed_loop_system"]
        poles = design["closed_loop_poles"]
    else:
        raise ValueError(f"Unsupported controller: {controller}")

    response = ct.forced_response(system, T=t_eval, U=disturbance_force, X0=x0, return_x=True)
    t, y, x = response
    control_force = -(K @ np.asarray(x)).ravel()

    return {
        "controller": controller,
        "t": np.asarray(t),
        "y": np.asarray(y),
        "state": np.asarray(x),
        "x": np.asarray(y)[0],
        "v": np.asarray(y)[1],
        "control_force": control_force,
        "disturbance_force": disturbance_force,
        "K": K,
        "closed_loop_poles": np.asarray(poles),
        "metrics": model["metrics"],
        "params": params,
    }


def summarize_controller_response(result: dict) -> dict:
    return {
        "controller": result["controller"],
        "peak_abs_x_m": float(np.max(np.abs(result["x"]))),
        "rms_x_m": float(np.sqrt(np.mean(result["x"] ** 2))),
        "area_abs_x": float(np.trapezoid(np.abs(result["x"]), result["t"])),
        "peak_abs_u_n": float(np.max(np.abs(result["control_force"]))),
    }


def controller_trade_study(
    params: dict | None = None,
    controllers: tuple[str, ...] | list[str] = ("open", "p", "lqr"),
    t_eval: np.ndarray | None = None,
    disturbance_force: np.ndarray | None = None,
    p_gain_scale: float = 1.0,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
) -> dict:
    results = []
    rows = []
    for controller in controllers:
        result = simulate_controller(
            controller,
            params=params,
            t_eval=t_eval,
            disturbance_force=disturbance_force,
            Q=Q,
            R=R,
            p_gain_scale=p_gain_scale,
        )
        results.append(result)
        rows.append(summarize_controller_response(result))
    return {"results": results, "rows": rows}


def export_trade_study_csv(rows: list[dict], filename: str | Path) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows must not be empty")
    fieldnames = list(rows[0].keys())
    with filename.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_trade_study(study: dict, filename: str = "sgms_anchor_controller_trade.png") -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for result in study["results"]:
        axes[0].plot(result["t"], result["x"], label=result["controller"])
        axes[1].plot(result["t"], result["control_force"], label=result["controller"])
    axes[0].set_ylabel("x (m)")
    axes[0].set_title("Controller Trade Study")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Control force (N)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def run_robustness_scenarios(
    base_params: dict | None = None,
    scenarios: list[dict] | None = None,
    controller: str = "lqr",
    t_eval: np.ndarray | None = None,
    base_disturbance: np.ndarray | None = None,
    p_gain_scale: float = 1.0,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
) -> dict:
    base_params = _copy_params(base_params)
    scenarios = [{"name": "nominal", "params": {}}] if scenarios is None else scenarios
    results = []
    rows = []
    for scenario in scenarios:
        params = base_params.copy()
        params.update(scenario.get("params", {}))
        disturbance = base_disturbance
        if disturbance is None:
            disturbance = np.zeros_like(t_eval) if t_eval is not None else None
        result = simulate_controller(
            controller,
            params=params,
            t_eval=t_eval,
            disturbance_force=disturbance,
            Q=Q,
            R=R,
            p_gain_scale=p_gain_scale,
        )
        summary = summarize_controller_response(result)
        summary["scenario"] = scenario["name"]
        summary["controller"] = controller
        results.append({"scenario": scenario["name"], "result": result})
        rows.append(summary)
    return {"results": results, "rows": rows}


def export_robustness_csv(rows: list[dict], filename: str | Path) -> None:
    export_trade_study_csv(rows, filename)


def plot_robustness(robustness: dict, filename: str = "sgms_anchor_robustness.png") -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for item in robustness["results"]:
        result = item["result"]
        ax.plot(result["t"], result["x"], label=item["scenario"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("x (m)")
    ax.set_title("Robustness Scenarios")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_open_closed_loop(result: dict, filename: str = "sgms_anchor_control_response.png") -> None:
    design = result["design"]
    metrics = design["metrics"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(result["t"], result["open_x"], label="Open-loop displacement", linewidth=2)
    axes[0].plot(result["t"], result["closed_x"], label="Closed-loop displacement", linewidth=2)
    axes[0].set_ylabel("x (m)")
    axes[0].set_title(
        "Anchor LQR Control Comparison\n"
        f"k_eff = {metrics['k_eff']:.2f} N/m | "
        f"zeta = {metrics['zeta']:.3f} | "
        f"K = [{design['K'][0, 0]:.3f}, {design['K'][0, 1]:.3f}]"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(result["t"], result["control_force"], label="Control force", color="#f2cc60")
    if np.any(np.abs(result["disturbance_force"]) > 0.0):
        axes[1].plot(result["t"], result["disturbance_force"], label="Disturbance force", color="#ff7b72")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Force (N)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def print_design_summary(result: dict) -> None:
    design = result["design"]
    K = design["K"]
    poles = design["closed_loop_poles"]
    print("=== ANCHOR CONTROL ANALYSIS ===")
    print(f"k_eff:              {design['metrics']['k_eff']:.6f} N/m")
    print(f"Natural period:     {design['metrics']['period_s']:.3f} s")
    print(f"LQR gain K:         [{K[0, 0]:.6f}, {K[0, 1]:.6f}]")
    print("Closed-loop poles:  " + ", ".join(f"{pole.real:.6f}{pole.imag:+.6f}j" for pole in poles))
    print(f"Peak |x| open:      {np.max(np.abs(result['open_x'])):.6f} m")
    print(f"Peak |x| closed:    {np.max(np.abs(result['closed_x'])):.6f} m")
    print(f"Peak |u_ctrl|:      {np.max(np.abs(result['control_force'])):.6f} N")


def main() -> None:
    params = DEFAULT_PARAMS.copy()
    params.update(
        {
            "u": 10.0,
            "lam": 0.5,
            "g_gain": 0.05,
            "ms": 1000.0,
            "c_damp": 0.5,
            "x0": 0.3,
            "v0": 0.0,
            "t_max": 300.0,
        }
    )

    t_eval = np.linspace(0.0, params["t_max"], 3000)
    disturbance = np.zeros_like(t_eval)
    disturbance[(t_eval >= 60.0) & (t_eval <= 120.0)] = 0.02

    result = simulate_open_closed_loop(params, t_eval=t_eval, disturbance_force=disturbance)
    plot_open_closed_loop(result)
    print_design_summary(result)


if __name__ == "__main__":
    main()
