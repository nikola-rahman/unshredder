"""Command line interface: `unshredder {run,shred,reconstruct,evaluate,sweep,train-seamnet,benchmark,...}`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from . import config as config_mod
from .evaluate import evaluate
from .pipeline import benchmark, eval_bag, make_strips, review_sim, run, sweep
from .reconstruct import reconstruct, save_reconstruction
from .shred import load_strips


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", help="YAML config file (see configs/)")
    p.add_argument(
        "-s", "--set", action="append", default=[], metavar="KEY=VALUE",
        help="override a config value, e.g. -s shred.num_strips=40 -s generate.num_docs=5",
    )
    p.add_argument("-o", "--output-dir", help="shortcut for -s output_dir=...")


def _load(args) -> config_mod.Config:
    cfg = config_mod.load(args.config, args.set)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    return cfg


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="unshredder", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="generate -> shred -> reconstruct -> evaluate")
    _add_config_args(p)

    p = sub.add_parser("shred", help="generate (or load) documents and write a shuffled strip pile")
    _add_config_args(p)

    p = sub.add_parser("reconstruct", help="reassemble a folder of strip images")
    p.add_argument("strips_dir")
    _add_config_args(p)

    p = sub.add_parser("evaluate", help="score a reconstruction.json against ground_truth.json")
    p.add_argument("reconstruction")
    p.add_argument("ground_truth")

    p = sub.add_parser("sweep", help="run a grid of configurations and tabulate metrics")
    _add_config_args(p)
    p.add_argument(
        "-g", "--grid", action="append", default=[], metavar="KEY=[V1,V2,...]",
        help="parameter to sweep, e.g. -g shred.num_strips=[10,20,40]",
    )
    p.add_argument("-r", "--repeats", type=int, default=1, help="seeds per grid point")

    p = sub.add_parser("train", help="train the Paixão 2020 cnn baseline")
    _add_config_args(p)

    p = sub.add_parser("train-seamnet", help="train the SeamNet compatibility model")
    _add_config_args(p)

    p = sub.add_parser("render-pages", help="pre-render synthetic training pages to PNG")
    p.add_argument("out", help="output folder")
    p.add_argument("-n", type=int, default=20000)
    p.add_argument("--workers", type=int, default=8)
    _add_config_args(p)

    p = sub.add_parser("render-pdfs", help="render real PDFs from PDFA WebDataset shards to 300 dpi PNGs")
    p.add_argument("shards", help="folder of pdfa-eng-train-*.tar shards")
    p.add_argument("out", help="output folder")
    p.add_argument("--pages-per-doc", type=int, default=2)
    p.add_argument("--workers", type=int, default=32)

    p = sub.add_parser("make-bag", help="simulate a bag: many pages shredded into one shuffled pile")
    p.add_argument("out", help="output folder")
    p.add_argument("--pages", type=int, default=300)
    p.add_argument("--page-dirs", nargs="+", required=True, help="folders of page images to shred")
    p.add_argument("--loss", type=float, default=0.03, help="fraction of strips lost")
    p.add_argument("--strength", type=float, default=1.0, help="shredding/scan damage strength")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("eval-bag", help="reconstruct a simulated bag and report pile-scale metrics")
    p.add_argument("bag", help="folder made by make-bag")
    _add_config_args(p)

    p = sub.add_parser("serve", help="inference API next to the model (run on the GPU server)")
    p.add_argument("--model", required=True, help="SeamNet checkpoint, e.g. models/run2.pt")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--jobs", default="/root/data/app_jobs", help="where job inputs and results are kept")

    p = sub.add_parser("app", help="local web app that forwards uploads to the inference API")
    p.add_argument("--remote", default="http://127.0.0.1:8001", help="inference API URL (via SSH tunnel)")
    p.add_argument("--port", type=int, default=8080)

    p = sub.add_parser("review-sim", help="simulated reviewer: decisions needed to get every page 100% right")
    p.add_argument("source", help='"val" (validation documents pooled) or a bag folder')
    p.add_argument("--k", type=int, default=5, help="suggestions shown per question")
    p.add_argument("--auto", type=float, default=0.99, help="auto-commit confidence threshold")
    p.add_argument("--error-rate", type=float, default=0.0, help="share of reviewer answers that are wrong")
    _add_config_args(p)

    p = sub.add_parser("benchmark", help="reconstruct a real dataset (S-MARQUES/S-ISRI-OCR/S-CDIP)")
    p.add_argument("data", help="dataset folder, e.g. datasets/D1")
    p.add_argument("--mode", choices=["mixed", "single"], default="mixed")
    p.add_argument("--docs", help='document slice, e.g. "0:20" (validation) or "20:" (test)')
    _add_config_args(p)

    args = parser.parse_args(argv)

    if args.cmd == "run":
        run(_load(args))
    elif args.cmd == "shred":
        out = make_strips(_load(args))
        print(f"wrote strips to {out / 'strips'} and ground truth to {out / 'ground_truth.json'}")
    elif args.cmd == "reconstruct":
        cfg = _load(args)
        names, images = load_strips(Path(args.strips_dir))
        chains, blanks, conf = reconstruct(images, cfg.reconstruct)
        out = Path(cfg.output_dir) / "reconstructed"
        save_reconstruction(chains, blanks, names, images, out, conf, render_images=len(chains) <= 2000)
        print(f"{len(names)} strips -> {len(chains)} documents in {out}")
    elif args.cmd == "evaluate":
        recon = json.loads(Path(args.reconstruction).read_text())
        truth = json.loads(Path(args.ground_truth).read_text())
        print(json.dumps(evaluate(recon, truth), indent=1))
    elif args.cmd == "sweep":
        cfg = _load(args)
        grid = {}
        for item in args.grid:
            key, raw = item.split("=", 1)
            values = yaml.safe_load(raw)
            grid[key.strip()] = values if isinstance(values, list) else [values]
        if not grid:
            parser.error("sweep needs at least one --grid")
        sweep(cfg, grid, args.repeats)
    elif args.cmd == "train":
        from .deep import train  # keeps torch out of the other commands

        train(_load(args))
    elif args.cmd == "train-seamnet":
        from .seamnet import train as train_seamnet

        train_seamnet(_load(args))
    elif args.cmd == "render-pages":
        from .generate import render_pages

        cfg = _load(args)
        render_pages(cfg.generate, args.out, args.n, args.workers, cfg.seed + 50_000)
    elif args.cmd == "render-pdfs":
        from .corpus import render_shards

        render_shards(args.shards, args.out, args.pages_per_doc, args.workers)
    elif args.cmd == "make-bag":
        from .bags import make_bag

        make_bag(args.page_dirs, args.pages, args.out, args.seed, loss=args.loss, strength=args.strength)
    elif args.cmd == "eval-bag":
        eval_bag(_load(args), args.bag)
    elif args.cmd == "serve":
        from .server import serve

        serve(args.model, args.host, args.port, args.jobs)
    elif args.cmd == "app":
        from .webapp import run as run_app

        print(f"open http://127.0.0.1:{args.port}  (forwarding to {args.remote})")
        run_app(args.remote, args.port)
    elif args.cmd == "review-sim":
        review_sim(_load(args), args.source, args.k, args.auto, args.error_rate)
    elif args.cmd == "benchmark":
        benchmark(_load(args), args.data, args.mode, args.docs)
