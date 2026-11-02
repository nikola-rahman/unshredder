"""Pipeline configuration: dataclasses, YAML loading and dotted-key overrides."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GenerateConfig:
    # Synthetic documents are generated unless input_dir points at real page images.
    input_dir: str | None = None
    num_docs: int = 3
    # Defaults: A4 at 300 dpi, like the real scanned benchmarks (configs/small.yaml for quick runs).
    width: int = 2480
    height: int = 3508
    size_jitter: int = 0  # +/- px applied independently to each document's width/height
    font_size: tuple[int, int] = (36, 60)  # ~9-14 pt at 300 dpi
    line_spacing: tuple[float, float] = (1.25, 1.8)
    margin: tuple[int, int] = (150, 300)
    paragraphs: tuple[int, int] = (6, 14)
    figure_prob: float = 0.3  # chance of drawing a box/table figure per document
    fonts: list[str] = field(default_factory=list)  # extra .ttf paths; system fonts used otherwise


@dataclass
class ShredConfig:
    num_strips: int = 30  # strips per document (ignored if strip_width > 0)
    strip_width: int = 0  # fixed strip width in px, like a real shredder
    width_jitter: int = 0  # +/- px random variation of each cut position
    kerf: int = 0  # px destroyed at every cut
    noise: float = 0.0  # gaussian noise std, intensities in [0, 1]
    blur: float = 0.0  # gaussian blur sigma applied per strip
    brightness_jitter: float = 0.0  # per-strip exposure variation (scanner drift)
    flip_prob: float = 0.0  # probability a strip is scanned upside down (rotated 180)
    vertical_jitter: int = 0  # +/- px vertical misalignment of each strip (crop offset)
    edge_damage: int = 0  # px at each strip edge replaced by random noise (torn/frayed cut)


@dataclass
class ReconstructConfig:
    # int, "auto" (split where links look implausible) or "oracle" (true count; `run` only)
    num_docs: int | str = "auto"
    compat: str = "classic"  # "classic" (pixel edges), "seamnet" (learned) or "cnn" (Paixão 2020)
    solver: str = "tsp"  # "tsp" (global, LKH) or "greedy" (Kruskal-style linking)
    split_threshold: float | None = None  # "auto" split score cutoff; None = per-compat default
    allow_flips: bool = False  # consider each strip in both orientations (greedy solver only)
    refine: bool = True  # local search on each chain (greedy solver only)
    link_threshold: float = 0.5  # greedy + learned model: only accept links at least this confident
    blank_threshold: float = 0.001  # fraction of dark pixels below which an edge/strip is blank
    normalize: bool = True  # per-strip background normalisation (undo brightness jitter)
    # classic compatibility
    metric: str = "prediction"  # "prediction" (gradient-extrapolated) or "l1"
    p: float = 0.5  # exponent of the per-pixel edge difference
    profile_weight: float = 0.0  # extra weight of row-ink-profile distance on inked seams
    # learned compatibility (seamnet / cnn)
    model_path: str = "models/seamnet.pt"
    max_shift: int = 12  # search vertical offsets in [-max_shift, max_shift] px
    shift_step: int = 1
    max_rows: int = 3000  # only the central rows of each seam are scored
    inference_batch: int = 2048  # seam images per forward pass
    device: str = "auto"


@dataclass
class TrainConfig:
    """Self-supervised training of the cnn compatibility (documents from `generate`)."""
    docs_dir: str | None = None  # intact page images to train on; synthetic pages otherwise
    num_docs: int = 400  # synthetic training pages (generated with seed + 10_000)
    strips_per_doc: int = 30
    samples_per_doc: int = 1000  # positive patches per document (+ as many negatives)
    min_ink: float = 0.2  # discard patches with less ink than this
    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-4  # 1e-3 collapses SqueezeNet to a constant output
    val_frac: float = 0.1
    pretrained: bool = True  # start from ImageNet weights
    model_path: str = "models/compat.pt"
    device: str = "auto"


@dataclass
class SeamNetConfig:
    """Contrastive edge bi-encoder (seamnet.py). Training data: synthetic pages, real
    intact pages (page_dirs), PDFs from collected shards (shard_dir) and real shredded
    documents (shred_dirs), mixed by weight."""
    model_path: str = "models/seamnet.pt"
    init_from: str | None = None  # checkpoint to fine-tune from
    band: int = 48  # px of each edge the encoder sees
    dim: int = 128  # embedding size per row
    row_context: str = "conv"  # "conv" (1D convs) or "transformer" (windowed attention over rows)
    pretrained: bool = True  # ImageNet ConvNeXt-Tiny weights
    max_shift: int = 12  # px of vertical misalignment searched (training and scoring)
    steps: int = 20000
    pages_per_batch: int = 4
    crop_rows: int = 1024  # training window height
    strips_per_page: tuple[int, int] = (20, 40)
    page_width: int = 2560  # page windows are padded/cropped to this width for GPU batching
    augment: float = 1.0  # GPU shredding/scan augmentation strength (0 = clean cuts)
    lr: float = 2e-4
    weight_decay: float = 0.05
    warmup: int = 500
    synth_weight: float = 1.0
    page_weight: float = 1.0
    shred_weight: float = 0.0
    shard_weight: float = 0.0  # PDFs streamed from collected tar shards (scripts/data/collect_finepdfs.py)
    shard_dir: str = ""  # local shard cache filled by scripts/data/sync_shards.py; *.tar files are used as they appear
    page_dirs: list[str] = field(default_factory=list)
    shred_dirs: list[str] = field(default_factory=list)
    val_dirs: list[str] = field(default_factory=list)  # real shredded sets for validation (academic: D1-D3 splits)
    val_dirs_practical: list[str] = field(default_factory=list)  # held-out Serbian set (make_srp_val.py), logged as val_srp/
    val_docs: int = 20  # documents per validation set
    workers: int = 8
    compile: bool = True  # torch.compile the encoder trunk during training (CUDA)
    cudnn_benchmark: bool = False  # cuDNN autotuning; off because batch shapes vary every step (see seamnet.train)
    checkpoint: bool = False  # recompute trunk activations in backward (same gradients, less GPU memory)
    log_every: int = 50
    eval_every: int = 1000
    val_review_every: int = 0  # also run the (slow, CPU) reviewer simulation every N steps and at the end; 0 = never
    device: str = "auto"


@dataclass
class Config:
    seed: int = 0
    output_dir: str = "runs/default"
    wandb_project: str | None = None  # log metrics (only) to this W&B project
    run_name: str | None = None
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    shred: ShredConfig = field(default_factory=ShredConfig)
    reconstruct: ReconstructConfig = field(default_factory=ReconstructConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seamnet: SeamNetConfig = field(default_factory=SeamNetConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_SECTIONS = {
    "generate": GenerateConfig,
    "shred": ShredConfig,
    "reconstruct": ReconstructConfig,
    "train": TrainConfig,
    "seamnet": SeamNetConfig,
}


def _coerce(cls, key: str, value: Any) -> Any:
    names = {f.name for f in dataclasses.fields(cls)}
    if key not in names:
        raise KeyError(f"unknown config key {cls.__name__}.{key}; valid: {sorted(names)}")
    default = getattr(cls(), key)
    if isinstance(default, tuple) and isinstance(value, (list, tuple)):
        return tuple(value)
    # YAML 1.1 reads "3e-4" as a string; numeric settings must stay numeric.
    if isinstance(default, (int, float)) and not isinstance(default, bool) and isinstance(value, str):
        try:
            return type(default)(float(value)) if isinstance(default, float) else int(float(value))
        except ValueError:
            pass
    return value


def from_dict(data: dict[str, Any]) -> Config:
    cfg = Config()
    for key, value in (data or {}).items():
        if key in _SECTIONS:
            section = getattr(cfg, key)
            for k, v in (value or {}).items():
                setattr(section, k, _coerce(type(section), k, v))
        else:
            setattr(cfg, key, _coerce(Config, key, value))
    return cfg


def load(path: str | Path | None, overrides: list[str] | None = None) -> Config:
    data = yaml.safe_load(Path(path).read_text()) if path else {}
    cfg = from_dict(data)
    for item in overrides or []:
        apply_override(cfg, item)
    return cfg


def apply_override(cfg: Config, item: str) -> None:
    """Apply `section.key=value` (value parsed as YAML, so 3, 0.5, [1, 2], true all work)."""
    if "=" not in item:
        raise ValueError(f"override must look like key=value, got {item!r}")
    key, raw = item.split("=", 1)
    set_value(cfg, key.strip(), yaml.safe_load(raw))


def set_value(cfg: Config, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    if len(parts) == 1:
        setattr(cfg, parts[0], _coerce(Config, parts[0], value))
    elif len(parts) == 2 and parts[0] in _SECTIONS:
        section = getattr(cfg, parts[0])
        setattr(section, parts[1], _coerce(type(section), parts[1], value))
    else:
        raise KeyError(f"bad config key {dotted!r}")


def save(cfg: Config, path: str | Path) -> None:
    def plain(obj):
        if isinstance(obj, dict):
            return {k: plain(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [plain(v) for v in obj]
        return obj

    Path(path).write_text(yaml.safe_dump(plain(cfg.to_dict()), sort_keys=False))
