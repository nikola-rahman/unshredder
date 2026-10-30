"""Organise the W&B project for live watching: group existing runs and save a workspace
view with three sections (real-data accuracy, training loss, bag benchmark).

  uv run --with wandb-workspaces python scripts/wandb_dashboard.py <entity> <project> [view URL to update]
"""

import sys

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

entity, project = sys.argv[1], sys.argv[2]


RENAME = {
    "fold-holdD1-aug0-initv1": "D1 · no aug · from v1",
    "fold-holdD1-tfm-aug0-scratch": "D1 · transformer · no aug · scratch",
    "fold-holdD2-aug0": "D2 · no aug",
    "fold-holdD2-aug05": "D2 · aug 0.5",
    "fold-holdD3-aug05": "D3 · aug 0.5",
}
DELETE = {"fold-holdD1-aug0-scratch", "fold-holdD3-aug0"}  # answered: no augmentation fails


def group_for(name: str) -> str | None:
    if name[:2] in ("D1", "D2", "D3"):
        return "train-hold" + name[:2]
    if name.startswith("bag"):
        return "bags"
    return None


api = wandb.Api()
for run in api.runs(f"{entity}/{project}"):
    if run.name in DELETE and run.state != "running":
        run.delete()
        print("deleted", run.name)
        continue
    if run.name in RENAME:
        run.name = RENAME[run.name]
    g = group_for(run.name)
    if g:
        run.group = g
    run.update()

SECTIONS = [
    ws.Section(name="1. Real held-out shreds: true neighbour ranked #1 among all ~1,100 candidates (higher wins)",
               is_open=True, panels=[
        wr.LinePlot(title="val/top1_pooled = hits / true seams (non-matches not counted)", x="Step",
                    y=["val/top1_pooled"]),
    ]),
    ws.Section(name="2. Documents recovered: share of each document that comes out as one intact piece",
               is_open=True, panels=[
        wr.LinePlot(title="page recovery, averaged over the 45 validation documents", x="Step",
                    y=["val/page_recovery"]),
        wr.MediaBrowser(title="distribution per run (latest validation)", media_keys=["val/recovery_distribution"],
                        mode="gallery", gallery_axis="run", num_columns=2),
    ]),
    ws.Section(name="3. Training", is_open=True, panels=[
        wr.LinePlot(title="training loss", x="Step", y=["train/loss"]),
        wr.LinePlot(title="training batch top1 (true neighbour ranked #1 within the training batch)", x="Step",
                    y=["train/batch_top1"]),
        wr.LinePlot(title="val/top1: true neighbour ranked #1 within its own document", x="Step", y=["val/top1"]),
    ]),
    ws.Section(name="4. Bag benchmark (the use case)", is_open=True, panels=[
        wr.BarPlot(title="recall at 99% precision (headline)", metrics=["recall_at_p99"]),
        wr.BarPlot(title="pages fully recovered", metrics=["perfect_doc_rate"]),
    ]),
]

if len(sys.argv) > 3:  # update an existing view in place (keeps its URL)
    workspace = ws.Workspace.from_url(sys.argv[3])
    workspace.sections = SECTIONS
else:
    workspace = ws.Workspace(name="Live: unshredder", entity=entity, project=project, sections=SECTIONS)
workspace.save()
print("workspace:", workspace.url.replace("forge.coreweave.com/", "forge.coreweave.com/wandb/"))
