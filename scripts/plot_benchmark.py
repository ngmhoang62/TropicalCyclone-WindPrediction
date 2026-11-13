"""Create a publishable aggregate comparison from completed benchmark tables."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", default="reports/benchmark")
args = parser.parse_args()
folder = Path(__file__).resolve().parents[1] / args.benchmark
metrics = pd.read_csv(folder / "holdout_metrics.csv")
manifest = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
horizons = sorted(metrics.horizon_hours.unique())
fig, axis = plt.subplots(figsize=(9, 5))
x = np.arange(len(horizons))
plotted_maximum = 0.0
for offset, label, color, mask in [
    (-0.18, "Persistence", "#78889b", metrics.model == "persistence"),
    (0.18, "Development-selected model", "#187f8c", metrics.is_champion),
]:
    group = metrics.loc[mask].set_index("horizon_hours").loc[horizons]
    values = group.mae_mps.to_numpy()
    ci = np.array([json.loads(v) for v in group.mae_storm_bootstrap_95ci])
    plotted_maximum = max(plotted_maximum, float(ci[:, 1].max()))
    errors = np.maximum(0, np.vstack([values - ci[:, 0], ci[:, 1] - values]))
    bars = axis.bar(
        x + offset, values, width=0.34, label=label, color=color, yerr=errors, capsize=4
    )
    for bar, value, upper in zip(bars, values, ci[:, 1]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            upper + 0.10,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
axis.set_xticks(x, [f"{h} hours" for h in horizons])
axis.set(ylabel="Mean absolute error (m/s), lower is better", xlabel="Forecast horizon")
axis.set_ylim(0, plotted_maximum * 1.22 + 0.2)
start = manifest["config"]["calibration_end"][:4]
end = manifest["data_audit"]["time_max"][:4]
axis.set_title(f"Wind forecasting on the {start}-{end} locked holdout", pad=16)
axis.legend(frameon=False)
axis.spines[["top", "right"]].set_visible(False)
axis.grid(axis="y", alpha=0.15)
axis.set_axisbelow(True)
fig.text(
    0.5,
    0.015,
    "Error bars: 95% MAE intervals from resampling entire storms.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.04, 1, 1))
(folder / "figures").mkdir(exist_ok=True)
path = folder / "figures" / "benchmark_mae.png"
fig.savefig(path, dpi=170)
plt.close(fig)
print(path)
