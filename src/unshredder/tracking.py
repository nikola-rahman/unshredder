"""Optional Weights & Biases logging: scalar metrics and config only.

No artifacts, model weights, code, git state, requirements or console output are
uploaded. Enabled when `wandb_project` is set in the config and wandb is logged in.
"""

from __future__ import annotations

from typing import Any


RECOVERY_BINS = [("0-20%", 0.0, 0.2), ("20-40%", 0.2, 0.4), ("40-60%", 0.4, 0.6),
                 ("60-80%", 0.6, 0.8), ("80-99%", 0.8, 1.0), ("100%", 1.0, 1.01)]


def recovery_bins(values: list[float]) -> dict[str, float]:
    """Share of documents per recovery band, as flat scalars a bar chart can compare across runs."""
    n = max(1, len(values))
    return {f"docs recovered {name}": sum(lo <= v < hi for v in values) / n for name, lo, hi in RECOVERY_BINS}


def recovery_chart(values: list[float], title: str):
    """Vertical histogram of per-document recovery: documents per band, labelled bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ink, muted, base, surface, blue = "#0b0b0b", "#898781", "#c3c2b7", "#fcfcfb", "#2a78d6"
    names = [name for name, _, _ in RECOVERY_BINS]
    counts = [sum(lo <= v < hi for v in values) for _, lo, hi in RECOVERY_BINS]
    n = max(1, len(values))
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.bar(range(len(names)), counts, width=0.9, color=blue, edgecolor=surface, linewidth=2)
    for i, c in enumerate(counts):
        if c:
            ax.text(i, c, f"{c}  ({c / n:.0%})", ha="center", va="bottom", fontsize=8, color=ink)
    ax.set_xticks(range(len(names)), names, fontsize=8, color=muted)
    ax.set_xlabel("share of the document recovered as one intact piece", fontsize=8, color=muted)
    ax.set_ylabel("documents", fontsize=8, color=muted)
    ax.tick_params(axis="y", labelsize=8, colors=muted)
    ax.set_ylim(0, max(counts) * 1.2 + 1)
    ax.grid(axis="y", color=base, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(base)
    mean, median = (float(np.mean(values)), float(np.median(values))) if values else (0.0, 0.0)
    ax.set_title(f"{title}\n{len(values)} documents · mean {mean:.0%} · median {median:.0%}",
                 fontsize=9, color=ink, loc="left")
    fig.tight_layout()
    return fig


class Tracker:
    def __init__(self, project: str | None, name: str | None, config: dict[str, Any], job_type: str,
                 group: str | None = None):
        self.run = None
        if not project:
            return
        import wandb

        self.run = wandb.init(
            project=project,
            name=name,
            job_type=job_type,
            group=group,
            config=config,
            settings=wandb.Settings(
                console="off",
                save_code=False,
                disable_code=True,
                disable_git=True,
                x_save_requirements=False,
            ),
        )

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.run:
            self.run.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, step=step)

    def log_distribution(self, name: str, per_item: dict[str, float], step: int | None = None) -> None:
        """Chart of per-document recovery (documents per recovery band) as an image."""
        if self.run:
            import matplotlib.pyplot as plt
            import wandb

            fig = recovery_chart(list(per_item.values()), self.run.name or "")
            self.run.log({name: wandb.Image(fig)}, step=step)
            plt.close(fig)

    def summary(self, metrics: dict[str, Any]) -> None:
        if self.run:
            self.run.summary.update({k: v for k, v in metrics.items() if isinstance(v, (int, float, str))})

    def finish(self) -> None:
        if self.run:
            self.run.finish()
