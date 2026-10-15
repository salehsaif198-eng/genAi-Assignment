import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = Path("metrics")
FIGURES = Path("figures")


def load(name):
    rows = []
    for line in (METRICS / name).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def column(rows, key):
    return [row[key] for row in rows]


FIGURES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------- Figure 1: SFT ----------------
sft = load("sft_metrics.jsonl")
steps = column(sft, "step")

fig, ax = plt.subplots(figsize=(5.4, 3.2))
ax.plot(steps, column(sft, "train_loss"), color="0.55", linestyle="--",
        marker="o", markersize=3, label="training loss")
ax.plot(steps, column(sft, "val_loss"), color="0.0", linestyle="-",
        marker="s", markersize=3, label="validation loss")

best = min(column(sft, "val_loss"))
best_step = steps[column(sft, "val_loss").index(best)]
ax.axvline(best_step, color="0.0", linewidth=0.8, linestyle=":")
ax.annotate("best checkpoint\nstep " + str(best_step) + ", " + str(round(best, 4)),
            xy=(best_step, best), xytext=(-95, 18),
            textcoords="offset points", fontsize=8,
            arrowprops=dict(arrowstyle="->", linewidth=0.7, color="0.0"))

ax.set_xlabel("optimizer step")
ax.set_ylabel("loss")
ax.set_title("SFT: Qwen2.5-0.5B, 1 epoch, 937 steps", fontsize=9.5)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / "sft_loss.png", dpi=200)
plt.close(fig)

# ---------------- Figure 2: both DPO runs ----------------
uf = load("dpo_uf_metrics.jsonl")
hh = load("dpo_hh_metrics.jsonl")

fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))

panels = [
    ("val_loss", "validation DPO loss", "loss"),
    ("reward_accuracy", "reward accuracy", "accuracy"),
    ("reward_margin", "reward margin", "margin"),
]

for ax, panel in zip(axes, panels):
    key = panel[0]
    title = panel[1]
    ylabel = panel[2]

    ax.plot(column(uf, "step"), column(uf, key), color="0.0",
            linestyle="-", marker="o", markersize=3, label="UltraFeedback")
    ax.plot(column(hh, "step"), column(hh, key), color="0.55",
            linestyle="--", marker="s", markersize=3, label="HH-RLHF")

    ax.set_xlabel("optimizer step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9.5)

axes[0].axhline(0.6931, color="0.3", linewidth=0.8, linestyle=":")
axes[0].annotate("log 2", xy=(axes[0].get_xlim()[1], 0.6931),
                 xytext=(-28, 4), textcoords="offset points", fontsize=8)
axes[1].axhline(0.5, color="0.3", linewidth=0.8, linestyle=":")
axes[1].set_ylim(0.4, 1.0)
axes[0].legend(frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig(FIGURES / "dpo_training.png", dpi=200)
plt.close(fig)

print("wrote", FIGURES / "sft_loss.png")
print("wrote", FIGURES / "dpo_training.png")