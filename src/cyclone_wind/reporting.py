"""Portable research tables and figures, without embedding raw storm observations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plots(predictions: pd.DataFrame, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    data = predictions.loc[predictions["is_champion"]].copy()
    for horizon, group in data.groupby("horizon_hours"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(group["target"], group["prediction"], s=8, alpha=0.3)
        maximum = max(group["target"].max(), group["prediction"].max())
        axes[0].plot([0, maximum], [0, maximum], "--", color="black", lw=1)
        axes[0].set(xlabel="Observed wind (m/s)", ylabel="Forecast wind (m/s)")
        axes[1].hist(group["prediction"] - group["target"], bins=30)
        axes[1].set(xlabel="Forecast minus observed (m/s)", ylabel="Count")
        fig.suptitle(f"{horizon}h held-out forecasts: development-selected champion")
        fig.tight_layout()
        fig.savefig(output / f"diagnostics_{horizon}h.png", dpi=140)
        plt.close(fig)


def write_summary(output: Path, metadata: dict, selection: pd.DataFrame, metrics: pd.DataFrame):
    lines = [
        "# Reproducible research run",
        "",
        f"Run: {metadata['run_id']}",
        f"Data SHA256: {metadata['data_audit']['input_sha256']}",
        "",
        "Historical reanalysis experiment. Not a live trading or weather-warning service.",
        "Model family and hyperparameters selected on development walk-forward folds only.",
        "Final holdout is never used for tuning or champion selection.",
        "",
        "## Development walk-forward selection",
        "",
        "| Horizon | Model | Validation MAE (m/s) | Champion |",
        "|---|---|---:|---|",
    ]
    for row in selection.to_dict("records"):
        lines.append(
            f"| {row['horizon_hours']}h | {row['model']} | {row['cv_mae_mps']:.4f} "
            f"| {row['is_champion']} |"
        )
    if not metrics.empty:
        lines += [
            "",
            "## Locked holdout",
            "",
            "| Horizon | Model | MAE | RMSE | R2 | Skill vs persistence |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in metrics.to_dict("records"):
            skill = row["mae_skill_vs_persistence"]
            lines.append(
                f"| {row['horizon_hours']}h | {row['model']} | {row['mae_mps']:.4f} "
                f"| {row['rmse_mps']:.4f} | {row['r2']:.4f} | {skill:.3f} |"
            )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- ERA5 fields are retrospective and not verified as available at forecast issuance.",
        "- Uncertainty intervals are calibrated on 2019-2021; temporal drift can reduce coverage.",
        "- Bootstrap intervals resample storms; there is no claim of IID observations.",
        "- Different model families can use different feature subsets, but share forecast origins "
        "and labels. LSTM uses the frozen sequence subset.",
        "- Old course scores used different target alignment and are not comparable.",
        "- No financial prices, technical indicators, trading backtest, profit or Sharpe ratio.",
        "- See metadata.json, selection.csv, cv_trials.csv and holdout_metrics.csv for evidence.",
    ]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
