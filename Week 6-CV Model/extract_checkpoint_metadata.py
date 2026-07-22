"""
Week 6 — Extract per-epoch training history from a YOLOv8 checkpoint (.pt)
WITHOUT needing torch or ultralytics installed.

Why this exists: `best.pt` already contains everything Ultralytics logged
during training — full per-epoch loss/metric history — inside its pickled
metadata. Normally you'd need `torch.load()` (and therefore a torch
install) just to read those numbers back out.

This version works by statically disassembling the pickle bytecode with
`pickletools.dis()` and scanning the resulting text for known metric keys
followed by their float values — it never actually constructs the pickled
objects (which would require resolving torch tensor classes we don't have
installed, and can be unstable on some machines/pickle protocol versions).
That makes it safe to run in a plain Python environment with only the
standard library.

Usage:
    python extract_checkpoint_metadata.py --weights best.pt --out_dir results
"""

import argparse
import io
import json
import pickletools
import re
import zipfile
from pathlib import Path

METRIC_KEYS = [
    "train/box_loss", "train/cls_loss", "train/dfl_loss",
    "val/box_loss", "val/cls_loss", "val/dfl_loss",
    "metrics/precision(B)", "metrics/recall(B)",
    "metrics/mAP50(B)", "metrics/mAP50-95(B)",
]


def _find_key_index(lines, key):
    needle = f"BINUNICODE '{key}'"
    for i, l in enumerate(lines):
        if needle in l:
            return i
    return None


def _collect_floats_after(lines, start_idx, max_lines=3000):
    floats = []
    for l in lines[start_idx + 1:start_idx + max_lines]:
        m = re.search(r"BINFLOAT\s+([\-0-9.]+)", l)
        if m:
            floats.append(float(m.group(1)))
        elif floats and "BINUNICODE" in l:
            # hit the next dict key — this list is done
            break
    return floats


def extract_training_history(weights_path: str) -> dict:
    z = zipfile.ZipFile(weights_path)
    pkl_name = next(n for n in z.namelist() if n.endswith("data.pkl"))
    data = z.read(pkl_name)

    buf = io.StringIO()
    pickletools.dis(data, out=buf)
    lines = buf.getvalue().split("\n")

    tr_idx = _find_key_index(lines, "train_results")
    if tr_idx is None:
        raise ValueError(
            "Could not find 'train_results' in this checkpoint. It may not "
            "have been saved with training history (e.g. a bare inference-only "
            "export), or the pickle format changed."
        )

    history = {}
    for key in METRIC_KEYS:
        for i in range(tr_idx, min(tr_idx + 30000, len(lines))):
            if f"BINUNICODE '{key}'" in lines[i]:
                vals = _collect_floats_after(lines, i)
                if vals:
                    history[key] = vals
                break

    if not history:
        raise ValueError("Found 'train_results' but couldn't extract any metric arrays from it.")

    # The literal 'epoch' field in some Ultralytics versions stores elapsed
    # time rather than the epoch number when read this way — a plain
    # sequential index (1, 2, 3, ...) is what you want for the x-axis anyway.
    n = len(next(iter(history.values())))
    history["epoch"] = list(range(1, n + 1))
    return history


def extract_run_args(weights_path: str) -> dict:
    """Best-effort extraction of a few top-level scalar/string fields
    (model name, dataset path, epochs requested, etc). Falls back to an
    empty dict for any field it can't find — these are convenience
    metadata only, not required for the metric history above."""
    z = zipfile.ZipFile(weights_path)
    pkl_name = next(n for n in z.namelist() if n.endswith("data.pkl"))
    data = z.read(pkl_name)
    buf = io.StringIO()
    pickletools.dis(data, out=buf)
    lines = buf.getvalue().split("\n")

    args = {}
    for key in ["model", "data", "epochs", "imgsz", "batch", "patience", "optimizer", "lr0", "seed"]:
        idx = _find_key_index(lines, key)
        if idx is None:
            continue
        for l in lines[idx + 1:idx + 4]:
            m = re.search(r"(?:SHORT_BINUNICODE|BINUNICODE)\s+'([^']*)'", l)
            if m:
                args[key] = m.group(1)
                break
            m2 = re.search(r"BININT1?\s+(\d+)", l)
            if m2:
                args[key] = int(m2.group(1))
                break
            m3 = re.search(r"BINFLOAT\s+([\-0-9.]+)", l)
            if m3:
                args[key] = float(m3.group(1))
                break
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    history = extract_training_history(args.weights)
    run_args = extract_run_args(args.weights)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Checkpoint: {args.weights}")
    print(f"Epochs run: {len(history['epoch'])}")
    if run_args:
        print(f"Run args (best-effort): {run_args}")
    print("=" * 60)
    print("Metrics at final logged epoch:")
    for k in ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
        if k in history:
            print(f"  {k}: {history[k][-1]}")

    best_idx = max(range(len(history["epoch"])), key=lambda i: history["metrics/mAP50-95(B)"][i])
    print(f"\nBest epoch by mAP50-95: epoch {history['epoch'][best_idx]}")
    for k in ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
        if k in history:
            print(f"  {k}: {history[k][best_idx]}")
    print("=" * 60)

    with open(out_dir / "full_training_history.json", "w") as f:
        json.dump(history, f)
    print(f"\nFull per-epoch history written to {out_dir / 'full_training_history.json'}")

    summary = {
        "run_args": run_args,
        "epochs_run": len(history["epoch"]),
        "best_epoch": history["epoch"][best_idx],
        "best_epoch_metrics": {k: history[k][best_idx] for k in
                                ["metrics/precision(B)", "metrics/recall(B)",
                                 "metrics/mAP50(B)", "metrics/mAP50-95(B)"] if k in history},
        "final_epoch_metrics": {k: history[k][-1] for k in
                                 ["metrics/precision(B)", "metrics/recall(B)",
                                  "metrics/mAP50(B)", "metrics/mAP50-95(B)"] if k in history},
    }
    with open(out_dir / "checkpoint_metadata_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {out_dir / 'checkpoint_metadata_summary.json'}")

    print(
        "\nNOTE: these are the metrics Ultralytics logged on the val split "
        "during training. For the final submission you should still run "
        "evaluate.py against the held-out test split for official per-class "
        "numbers, the confusion matrix, and sample prediction screenshots — "
        "this script only recovers what's already inside the checkpoint, it "
        "doesn't run new inference."
    )


if __name__ == "__main__":
    main()
