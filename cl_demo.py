#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cl_demo.py — On-device continual learning demo (CPU-only)

This script checks whether a tiny robot could fine-tune a vision model locally
(privacy-friendly) under tight time/memory budgets.

Flow:
  1) Generate a clean synthetic dataset of colored shapes (lab-like data).
  2) Load ~20 real photos of the same shapes (field-like data).
  3) Load a tiny pretrained model (MobileNetV2), replace the last classifier
     layer, and fine-tune only that head on the real photos (CPU only).
  4) Measure per-epoch wall time, RAM (both current RSS and absolute PEAK),
     and checkpoint write times (ext4 vs tmpfs).
  5) Save a CSV and simple plots.

Success bar checked at the end:
  - Total fine-tune time < 60 s
  - Peak RAM (absolute RSS) < 400 MB
"""

# ---------- Imports & CPU pinning (do this before importing torch) ----------
import os                               # OS utilities (env vars, paths, PIDs)
os.environ.setdefault("OMP_NUM_THREADS", "1")        # keep BLAS single-thread
os.environ.setdefault("MKL_NUM_THREADS", "1")        # "
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # "

import time                             # precise timing
import csv                              # write results.csv
import argparse                         # CLI flags like --workers/--batch
import math                             # safe division / NaN guards
from pathlib import Path                # cross-platform file paths
import sys                              # to detect platform for ru_maxrss units
import resource                         # ru_maxrss (absolute peak RSS)
import psutil                           # current RSS (point-in-time)

import torch                            # core tensor/training lib (CPU)
torch.set_num_threads(1)                # PyTorch intra-op threads = 1
torch.set_num_interop_threads(1)        # PyTorch inter-op threads = 1
torch.backends.mkldnn.enabled = False   # avoid MKL-DNN path on CPU (stability)
from torch import nn                    # layers/losses
from torch.utils.data import Dataset, DataLoader  # minimal data plumbing

from torchvision import models, transforms         # MobileNet + transforms
from PIL import Image, ImageDraw, ImageOps        # image I/O, drawing, EXIF rotate

# Optional warm-start on synth data (kept False for speed/memory)
DO_WARMSTART = False

# ---------- CLI arguments ----------
ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=0,
                help="DataLoader workers: 0 (simplest) or 4 (parallel I/O)")
ap.add_argument("--epochs_real", type=int, default=4,
                help="Number of epochs on the real photos")
ap.add_argument("--batch", type=int, default=4,
                help="Mini-batch size for training/eval (CPU-friendly small)")
ap.add_argument("--synth", type=int, default=100,
                help="Number of synthetic images to generate")
args = ap.parse_args()

# ---------- Paths & device ----------
DEVICE  = torch.device("cpu")           # force CPU (fits low-end hardware)
PROJECT = Path(".")                     # project root (current folder)
SYN     = PROJECT / "data_synth"        # synthetic images live here
REAL    = PROJECT / "data_real"         # put your ~20 photos here
PLOTS   = PROJECT / "plots"             # output charts go here
SYN.mkdir(exist_ok=True)                # create folders if missing
PLOTS.mkdir(exist_ok=True)
assert REAL.exists(), "Put ~20 real photos in ./data_real named orange_square_*.jpg etc."

# ---------- Class labels ----------
# Four classes = {orange, blue} × {square, circle}
classes   = ["orange_square", "blue_square", "orange_circle", "blue_circle"]
label_map = {c: i for i, c in enumerate(classes)}  # "orange_square" -> 0, etc.

# ---------- Image preprocessing ----------
# Resize small (faster/cheaper), then normalize like ImageNet.
IMG_SIZE = 128                           # good balance: fast + accurate on CPU

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

tf_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),               # shrink early
    transforms.ColorJitter(brightness=0.25, contrast=0.15),# tiny photometric wiggle
    transforms.RandomRotation(8, fill=(210, 210, 210)),    # paper-like border
    transforms.ToTensor(),                                 # [0,1] -> tensor
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

tf_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---------- Dataset that reads a flat folder of images ----------
class LabeledFolder(Dataset):
    """
    Expects filenames like: orange_square_001.jpg
    Label is parsed from the first two underscore-separated tokens.
    """
    def __init__(self, path: Path, transform):
        self.transform = transform
        self.paths, self.labels = [], []
        for p in sorted(path.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            key = "_".join(p.stem.split("_")[:2])          # e.g., "orange_square"
            if key in label_map:
                self.paths.append(p)
                self.labels.append(label_map[key])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # EXIF-aware rotate, force RGB, apply transform, return (img, label)
        img = ImageOps.exif_transpose(Image.open(self.paths[i])).convert("RGB")
        return self.transform(img), torch.tensor(self.labels[i])

# ---------- Synthetic "lab" images (clean colored shapes) ----------
def gen_synth(n: int = 100):
    """Create n synthetic images split evenly across the 4 classes."""
    per = max(1, n // len(classes))                         # roughly equal per class
    for cls in classes:
        color, shape = cls.split("_")                       # e.g., ("orange", "square")
        for k in range(per):
            img = Image.new("RGB", (224, 224), "black")     # black background
            d = ImageDraw.Draw(img)
            c = (240, 140, 0) if color == "orange" else (0, 110, 255)
            if shape == "square":
                d.rectangle([60, 60, 164, 164], fill=c)     # filled square
            else:
                d.ellipse([60, 60, 164, 164], fill=c)       # filled circle
            img.save(SYN / f"{cls}_{k}.png")

# ---------- Memory helpers ----------
def _ru_to_mb(v: int) -> float:
    """Convert ru_maxrss units to MB (KB on Linux, bytes on macOS)."""
    return (v / (1024 * 1024)) if sys.platform == "darwin" else (v / 1024.0)

def rss_now_mb() -> float:
    """Point-in-time Resident Set Size (what's in RAM *right now*)."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def abs_peak_mb() -> float:
    """Absolute process peak RSS so far (monotonic, since process start)."""
    return _ru_to_mb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def delta_peak_mb(baseline_ru: int) -> float:
    """
    Extra peak above a saved baseline ru_maxrss. Useful to report
    'how much higher did our peak go during training' for the paper.
    """
    return max(0.0, abs_peak_mb() - _ru_to_mb(baseline_ru))

# ---------- Simple accuracy ----------
@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader) -> float:
    """Compute classification accuracy over a loader (no gradients)."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        logits = model(x.to(DEVICE))          # forward pass (CPU)
        pred = logits.argmax(1).cpu()         # predicted class index
        correct += (pred == y).sum().item()
        total   += y.numel()
    return (correct / total) if total > 0 else math.nan

# ---------- Checkpoint timing: ext4 vs tmpfs ----------
def timed_checkpoint(model: nn.Module, path: Path):
    """
    Save the model twice and time each:
      1) ext4 (normal disk under the project folder)
      2) tmpfs (/dev/shm — RAM disk)
    Use atomic rename to avoid partial files.
    """
    # --- ext4 write ---
    t0 = time.perf_counter()
    torch.save(model.state_dict(), path.with_suffix(".pt.tmp"))
    os.replace(path.with_suffix(".pt.tmp"), path.with_suffix(".pt"))
    ext4_ms = (time.perf_counter() - t0) * 1000.0

    # --- tmpfs write (/dev/shm is memory-backed) ---
    t1 = time.perf_counter()
    tmpfs_tmp = Path("/dev/shm/model_tmp.pt.tmp")
    tmpfs_dst = Path("/dev/shm/model_tmp.pt")
    torch.save(model.state_dict(), tmpfs_tmp)
    os.replace(tmpfs_tmp, tmpfs_dst)
    tmpfs_ms = (time.perf_counter() - t1) * 1000.0

    # tidy RAM file (best effort)
    try:
        tmpfs_dst.unlink()
    except Exception:
        pass

    return ext4_ms, tmpfs_ms

# ---------- Main experiment ----------
def main():
    # 1) If synthetic data is missing, create it once.
    if not any(SYN.iterdir()):
        gen_synth(args.synth)

    # 2) Build datasets/loaders.
    ds_syn        = LabeledFolder(SYN,  tf_eval)   # synth: eval tf is fine for warm-start
    ds_real_train = LabeledFolder(REAL, tf_train)  # real: train tf with light augments
    ds_real_eval  = LabeledFolder(REAL, tf_eval)   # real: eval tf

    dl_syn  = DataLoader(ds_syn,        batch_size=args.batch, shuffle=True,  num_workers=args.workers)
    dl_real = DataLoader(ds_real_train, batch_size=args.batch, shuffle=True,  num_workers=args.workers)
    dl_eval = DataLoader(ds_real_eval,  batch_size=args.batch, shuffle=False, num_workers=0)

    # (Optional) show class balance — handy sanity check in logs
    from collections import Counter
    counts = Counter([label_map["_".join(p.stem.split("_")[:2])]
                      for p in REAL.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    print("real class counts:", {classes[k]: v for k, v in counts.items()})

    # 3) Load MobileNetV2 pretrained on ImageNet; swap classifier.
    model = models.mobilenet_v2(weights="IMAGENET1K_V2")
    for p in model.features.parameters():
        p.requires_grad = False                     # freeze all feature extractor
    model.classifier[1] = nn.Linear(1280, len(classes))  # 1000 -> 4 classes
    model.to(DEVICE)

    # 4) Optimizer + loss (head-only training is faster + tiny RAM).
    opt     = torch.optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # 5) Optional "lab warm-start" — off by default (kept for completeness).
    if DO_WARMSTART:
        model.train()
        for x, y in dl_syn:
            opt.zero_grad()
            logits = model(x.to(DEVICE))
            loss = loss_fn(logits, y.to(DEVICE))
            loss.backward()
            opt.step()

    # 6) Baseline accuracy on real photos *before* tuning.
    pre_acc = accuracy(model, dl_eval)

    # Save baseline ru_maxrss so we can also report delta-peak during training.
    baseline_ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # 7) Fine-tune loop with logging.
    # CSV columns include both now-RSS, absolute PEAK, and delta-PEAK for reporting.
    rows = [["epoch", "train_acc", "val_acc", "sec",
             "now_MB", "peak_MB", "delta_MB", "ext4_ms", "tmpfs_ms"]]

    for epoch in range(1, args.epochs_real + 1):
        # --- train one epoch ---
        t0 = time.time()
        model.train()
        for x, y in dl_real:
            opt.zero_grad()
            logits = model(x.to(DEVICE))
            loss = loss_fn(logits, y.to(DEVICE))
            loss.backward()
            opt.step()
        sec = time.time() - t0

        # --- memory + accuracy + checkpoint timings ---
        nowMB    = rss_now_mb()                 # point-in-time RSS (MB)
        peakMB   = abs_peak_mb()                # absolute process peak so far (MB)
        deltaMB  = delta_peak_mb(baseline_ru)   # extra peak above baseline (MB)

        tr_acc = accuracy(model, dl_eval)       # on tiny training set, acts like proxy-val
        va_acc = tr_acc

        ext4_ms, tmpfs_ms = timed_checkpoint(model, PROJECT / "model_final")

        # --- log to console & CSV ---
        print(f"epoch {epoch}: acc={tr_acc:.2f}  time={sec:.2f}s  "
              f"now={nowMB:.1f}MB  peak={peakMB:.1f}MB  delta={deltaMB:.1f}MB  "
              f"ext4={ext4_ms:.1f}ms  tmpfs={tmpfs_ms:.1f}ms")

        rows.append([epoch, f"{tr_acc:.4f}", f"{va_acc:.4f}",
                     f"{sec:.3f}", f"{nowMB:.1f}", f"{peakMB:.1f}", f"{deltaMB:.1f}",
                     f"{ext4_ms:.2f}", f"{tmpfs_ms:.2f}"])

    # 8) Write results.csv
    with open(PROJECT / "results.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # 9) Make two simple plots: accuracy vs epoch and PEAK RSS vs epoch.
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.read_csv(PROJECT / "results.csv")

        plt.figure()
        plt.plot(df["epoch"], df["train_acc"], marker="o")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title("Accuracy vs epoch")
        (PLOTS / "accuracy_vs_epoch.png").parent.mkdir(exist_ok=True)
        plt.savefig(PLOTS / "accuracy_vs_epoch.png", dpi=160)

        plt.figure()
        # Plot the *absolute* peak RSS to align with success criterion
        plt.plot(df["epoch"], df["peak_MB"], marker="o")
        plt.xlabel("epoch")
        plt.ylabel("peak RSS (MB)")
        plt.title("Peak RSS vs epoch")
        plt.savefig(PLOTS / "mem_vs_epoch.png", dpi=160)

        print("Wrote plots to", PLOTS)
    except Exception as e:
        print("Plotting skipped:", e)

    # 10) Summarize vs success criteria.
    final_acc  = float(rows[-1][1])   # last epoch’s accuracy (string in CSV row)
    total_time = sum(float(r[3]) for r in rows[1:])
    # Use the absolute *peak_MB* column for the memory criterion
    peak_overall = max(float(r[5]) for r in rows[1:])

    ok_time = total_time < 60.0
    ok_mem  = peak_overall < 400.0

    print(f"\nPre-tune acc on real: {pre_acc:.2f}")
    print(f"Final acc on real:    {final_acc:.2f}")
    print(f"Total fine-tune time: {total_time:.1f}s  (target < 60s)  -> {'OK' if ok_time else 'OVER'}")
    print(f"Peak RSS during tune: {peak_overall:.0f}MB (target < 400MB) -> {'OK' if ok_mem else 'OVER'}")
    print("Check results.csv and plots/ for details.")

# ---------- Entrypoint ----------
if __name__ == "__main__":
    main()