from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"

QSHARP_SOURCE = r"""
namespace XQ.Resource {
    operation SquaredDistanceOracleSkeleton(bits : Int, thresholdBits : Bool[]) : Unit {
        let n2 = 2 * bits;
        use x = Qubit[bits];
        use y = Qubit[bits];
        use x2 = Qubit[n2];
        use y2 = Qubit[n2];
        use s = Qubit[n2 + 1];
        use flag = Qubit();

        // Step 1/2: skeleton for x^2 and y^2 partial-product accumulation.
        for i in 0 .. bits - 1 {
            let idxSelf = (2 * i) % n2;
            CNOT(x[i], x2[idxSelf]);
            CNOT(y[i], y2[idxSelf]);
            if i < bits - 1 {
                for j in i + 1 .. bits - 1 {
                    let idx = (i + j) % n2;
                    Controlled X([x[i], x[j]], x2[idx]);
                    Controlled X([y[i], y[j]], y2[idx]);
                }
            }
        }

        // Step 3: sum skeleton (ripple-like XOR accumulation proxy).
        for k in 0 .. n2 - 1 {
            CNOT(x2[k], s[k]);
            CNOT(y2[k], s[k]);
        }

        // Step 4/5: compare skeleton against constant bits and mark a flag.
        for k in 0 .. Length(thresholdBits) - 1 {
            if thresholdBits[k] {
                CNOT(s[k], flag);
            } else {
                CNOT(s[k], flag);
                CNOT(s[k], flag);
            }
        }

        // Include one measurement so the estimator accepts the workload.
        let _ = M(flag);

        // Step 6: uncompute ancillas (reverse arithmetic skeleton).
        for k in 0 .. n2 - 1 {
            CNOT(y2[k], s[k]);
            CNOT(x2[k], s[k]);
        }

        for i in bits - 1 .. -1 .. 0 {
            if i < bits - 1 {
                for j in bits - 1 .. -1 .. i + 1 {
                    let idx = (i + j) % n2;
                    Controlled X([y[i], y[j]], y2[idx]);
                    Controlled X([x[i], x[j]], x2[idx]);
                }
            }
            let idxSelf = (2 * i) % n2;
            CNOT(y[i], y2[idxSelf]);
            CNOT(x[i], x2[idxSelf]);
        }

        Reset(flag);
        ResetAll(s);
        ResetAll(y2);
        ResetAll(x2);
        ResetAll(y);
        ResetAll(x);
    }
}
"""


def parse_int_list(text: str) -> list[int]:
    items = [p.strip() for p in text.split(",") if p.strip()]
    values = [int(x) for x in items]
    if not values:
        raise ValueError("list must contain at least one integer")
    return values


def parse_str_list(text: str) -> list[str]:
    items = [p.strip() for p in text.split(",") if p.strip()]
    if not items:
        raise ValueError("list must contain at least one value")
    return items


def parse_float_list(text: str) -> list[float]:
    items = [p.strip() for p in text.split(",") if p.strip()]
    values = [float(x) for x in items]
    if not values:
        raise ValueError("list must contain at least one float")
    return values


def _threshold_bits(bits: int) -> list[bool]:
    # Representative constant for d0^2 in a fixed-point-like integer encoding.
    n2p1 = 2 * bits + 1
    threshold_int = 1 << max(1, bits - 1)
    return [bool((threshold_int >> k) & 1) for k in range(n2p1)]


def _qsharp_bool_array_literal(values: list[bool]) -> str:
    return "[" + ", ".join("true" if v else "false" for v in values) + "]"


def _normalize_model_name(model: str) -> str:
    val = model.strip().lower()
    if not val:
        raise ValueError("empty model name")
    if val.startswith("qubit_"):
        return val
    return f"qubit_{val}"


def _estimate_one(
    qsharp_mod,
    estimator_mod,
    entry_expr: str,
    model: str,
    bits: int,
    error_budget: float,
) -> dict[str, float | int | str]:
    params = estimator_mod.EstimatorParams()
    params.error_budget = float(error_budget)
    params.qubit_params.name = model

    # Floquet QEC is the supported pre-defined path for Majorana models.
    if "_maj_" in model:
        params.qec_scheme.name = "floquet_code"

    result = qsharp_mod.estimate(entry_expr, params)

    physical = result["physicalCounts"]
    breakdown = physical["breakdown"]
    logical = result["logicalCounts"]
    qec_name = result["jobParams"]["qecScheme"]["name"]

    return {
        "bits": int(bits),
        "model": model,
        "error_budget": float(error_budget),
        "qec_scheme": str(qec_name),
        "physical_qubits": int(physical["physicalQubits"]),
        "runtime_ns": int(physical["runtime"]),
        "rqops": float(physical["rqops"]),
        "algorithmic_logical_qubits": int(breakdown["algorithmicLogicalQubits"]),
        "algorithmic_logical_depth": int(breakdown["algorithmicLogicalDepth"]),
        "logical_depth": int(breakdown["logicalDepth"]),
        "num_tstates": int(breakdown["numTstates"]),
        "logical_count_qubits": int(logical["numQubits"]),
        "logical_count_measurements": int(logical["measurementCount"]),
        "logical_count_ccz": int(logical["cczCount"]),
        "logical_count_t": int(logical["tCount"]),
    }


def _make_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.4))

    # X-axis groups are (bits, model) so each group gets one bar per error budget.
    combo = (
        df[["bits", "model"]].drop_duplicates().sort_values(["bits", "model"]).values
    )
    group_labels = [f"b={int(b)}\n{m}" for b, m in combo]
    budgets = sorted(df["error_budget"].unique())

    x = np.arange(len(group_labels), dtype=float)
    total_width = 0.85
    width = total_width / max(1, len(budgets))

    for j, eb in enumerate(budgets):
        y_vals = []
        for b, m in combo:
            row = df[
                (df["bits"] == int(b)) & (df["model"] == m) & (df["error_budget"] == eb)
            ]
            y_vals.append(float(row.iloc[0]["physical_qubits"]))

        offsets = x - total_width / 2 + width * (j + 0.5)
        bars = ax.bar(offsets, y_vals, width=width, label=f"error_budget={eb:g}")

        # Value labels (small font to keep figure readable).
        for rect, y in zip(bars, y_vals):
            ax.annotate(
                f"{int(y):,}",
                xy=(rect.get_x() + rect.get_width() / 2, y),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_yscale("log")
    ax.set_ylabel("Physical qubits")
    ax.set_title(
        "QDK Resource Estimator sweep: physical qubits by bits/model/error budget"
    )
    ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)


def _write_markdown(
    out_path: Path,
    bits_list: list[int],
    models: list[str],
    error_budgets: list[float],
    threshold_map: dict[int, list[bool]],
    df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# QDK Resource Estimate (Oracle Arithmetic Skeleton)")
    lines.append("")
    lines.append("## Sweep Parameters")
    lines.append(f"- bits: {bits_list}")
    lines.append(f"- models: {models}")
    lines.append(f"- error_budgets: {error_budgets}")
    lines.append("")

    threshold_summary = ", ".join(
        f"b={b}: set bits {[i for i, bit in enumerate(threshold_map[b]) if bit]}"
        for b in bits_list
    )
    lines.append("## Oracle Skeleton")
    lines.append(
        "- Q# workload approximates reversible arithmetic for `(x^2 + y^2) < d0^2`: "
        "partial-product style multiplies, sum skeleton, compare/mark skeleton, uncompute."
    )
    lines.append(f"- Threshold bit settings by bits: {threshold_summary}")
    lines.append("")

    lines.append("## Caveat")
    lines.append(
        "- **This artifact estimates oracle arithmetic skeleton only; it omits distribution state preparation (Gaussian/mixture) and full AE control overhead.**"
    )
    lines.append("")

    # Simple trend extraction for reviewer-safe takeaways.
    lines.append("## Takeaways")
    if len(bits_list) >= 2:
        b_lo = min(bits_list)
        b_hi = max(bits_list)
        q_lo = df[df["bits"] == b_lo]["physical_qubits"].median()
        q_hi = df[df["bits"] == b_hi]["physical_qubits"].median()
        lines.append(
            f"- Increasing precision from b={b_lo} to b={b_hi} raises median physical qubits by about {q_hi / q_lo:.2f}x in this sweep."
        )

    if len(error_budgets) >= 2:
        e_hi = max(error_budgets)
        e_lo = min(error_budgets)
        q_hi_err = df[df["error_budget"] == e_hi]["physical_qubits"].median()
        q_lo_err = df[df["error_budget"] == e_lo]["physical_qubits"].median()
        lines.append(
            f"- Tightening error budget from {e_hi:g} to {e_lo:g} increases median physical qubits by about {q_lo_err / q_hi_err:.2f}x."
        )

    by_model = (
        df.groupby("model", as_index=False)["physical_qubits"]
        .median()
        .sort_values("physical_qubits")
    )
    if len(by_model) >= 2:
        best = by_model.iloc[0]
        worst = by_model.iloc[-1]
        lines.append(
            f"- Across this sweep, lowest median qubit demand is `{best['model']}` ({int(best['physical_qubits']):,}), highest is `{worst['model']}` ({int(worst['physical_qubits']):,})."
        )

    lines.append("")
    lines.append("## Output Artifacts")
    lines.append(
        f"- Table: {REPO_ROOT / 'results' / 'tables' / 'qdk_resource_estimate_summary.csv'}"
    )
    lines.append(
        f"- Plot: {REPO_ROOT / 'results' / 'plots' / 'qdk_resource_estimate.png'}"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Microsoft QDK resource estimator for oracle arithmetic skeleton"
    )
    parser.add_argument(
        "--bits",
        type=parse_int_list,
        default=parse_int_list("32"),
        help="Comma-separated fixed-point bits per x/y, e.g. 16,24,32",
    )
    parser.add_argument(
        "--models",
        type=parse_str_list,
        default=["qubit_gate_ns_e3", "qubit_maj_ns_e4"],
        help="Comma-separated qubit model names",
    )
    parser.add_argument(
        "--error_budgets",
        "--error_budget",
        dest="error_budgets",
        type=parse_float_list,
        default=parse_float_list("1e-3"),
        help="Comma-separated error budgets in (0,1), e.g. 1e-2,1e-3,1e-4",
    )
    args = parser.parse_args()

    if any(b <= 0 for b in args.bits):
        raise ValueError("--bits values must be positive")
    if any(e <= 0 or e >= 1 for e in args.error_budgets):
        raise ValueError("--error_budgets values must be in (0, 1)")

    bits_list = sorted(set(args.bits))
    error_budgets = sorted(set(args.error_budgets), reverse=True)

    try:
        from qdk import qsharp
        from qsharp import estimator
    except Exception as exc:
        print(f"QDK import failed: {exc}")
        return 1

    allowed_models = {
        estimator.QubitParams.GATE_US_E3,
        estimator.QubitParams.GATE_US_E4,
        estimator.QubitParams.GATE_NS_E3,
        estimator.QubitParams.GATE_NS_E4,
        estimator.QubitParams.MAJ_NS_E4,
        estimator.QubitParams.MAJ_NS_E6,
    }

    models = [_normalize_model_name(m) for m in args.models]
    for model in models:
        if model not in allowed_models:
            raise ValueError(
                f"Unsupported model '{model}'. Allowed: {sorted(allowed_models)}"
            )

    qsharp.eval(QSHARP_SOURCE)

    rows: list[dict[str, float | int | str]] = []
    threshold_map: dict[int, list[bool]] = {}

    for bits in bits_list:
        threshold_bits = _threshold_bits(bits)
        threshold_map[bits] = threshold_bits
        threshold_literal = _qsharp_bool_array_literal(threshold_bits)
        entry_expr = (
            f"XQ.Resource.SquaredDistanceOracleSkeleton({bits}, {threshold_literal})"
        )

        for model in models:
            for eb in error_budgets:
                rows.append(
                    _estimate_one(
                        qsharp_mod=qsharp,
                        estimator_mod=estimator,
                        entry_expr=entry_expr,
                        model=model,
                        bits=bits,
                        error_budget=eb,
                    )
                )

    df = (
        pd.DataFrame(rows)
        .sort_values(["bits", "model", "error_budget"])
        .reset_index(drop=True)
    )

    table_path = TABLES_DIR / "qdk_resource_estimate_summary.csv"
    md_path = RESULTS_DIR / "qdk_resource_estimate.md"
    plot_path = PLOTS_DIR / "qdk_resource_estimate.png"

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_path, index=False)

    _make_plot(df, plot_path)
    _write_markdown(md_path, bits_list, models, error_budgets, threshold_map, df)

    print(f"Saved: {table_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {plot_path}")
    print(
        f"Ran {len(df)} estimate point(s): "
        f"bits={bits_list}, models={models}, error_budgets={error_budgets}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
