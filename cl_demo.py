#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cl_demo.py — On‑device continual learning demo (CPU‑only)

This script shows, step by step, how a small robot (or any camera‑equipped
sensor) can locally fine‑tune a vision model, and simultaneously log
operating‑system‑relevant metrics such as runtime, RAM (resident set size)
usage, and filesystem‑write latency.

High‑level flow:
  1) Make a tiny synthetic dataset of four classes (orange/blue × square/circle).
  2) Load 20 real photos (same four classes) from ./data_real.
  3) Load a pretrained MobileNet‑V2, replace the final classifier layer, and
     freeze the feature extractor to keep RAM low.
  4) (Optional) Do one warm‑start pass on the synthetic data.
  5) Fine‑tune on real photos for a few epochs and log metrics each epoch.
  6) Save a CSV of results and 4 PNG plots (accuracy + three memory views).

Why three memory views?
  • peak_abs_MB  : absolute peak RSS the OS has seen so far for this process
                   (backed by getrusage(…ru_maxrss)).
  • delta_MB     : how much that peak increased relative to the baseline
                   taken right before fine‑tuning (peak_abs − baseline).
  • now_MB       : point‑in‑time RSS, i.e., “what am I using right now?”
                   (via psutil).

Success criteria (SLOs) we check at the end:
  • Total fine‑tune time < 60 seconds
  • Absolute peak RSS  < 400 MB

Run example (recommended for CPU‑only VMs):
  MALLOC_ARENA_MAX=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
  python -u cl_demo.py --epochs_real 4 --batch 4 --workers 0 --img 128
"""

# ---------- Standard library imports ----------
import os                   # OS env vars and filesystem operations
import sys                  # platform check (ru_maxrss units differ)
import time                 # wall‑clock timing
import csv                  # write results.csv
import math                 # NaN‑safe divisions
import argparse             # command‑line flags
from pathlib import Path    # OS‑independent paths

# ---------- Third‑party imports ----------
import psutil               # resident set (RSS) for "now_MB"
import resource             # ru_maxrss for absolute peak RSS
import torch                # PyTorch core
from torch import nn        # neural network layers / loss
from torch.utils.data import Dataset, DataLoader  # data plumbing
from torchvision import models, transforms        # pretrained models + transforms
from PIL import Image, ImageDraw, ImageOps        # image I/O + EXIF‑aware rotation

# ---------- Conservative defaults for CPU‑only box ----------
# Limit glibc's per‑thread arena growth (helps keep RSS lower on Linux).
os.environ.setdefault("MALLOC_ARENA_MAX", "1")
# Keep BLAS libraries from oversubscribing CPU cores.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# Also tell PyTorch to avoid too much threading in this small workload.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
# In this project we avoid MKLDNN to keep behavior simple across hosts.
torch.backends.mkldnn.enabled = False

# ---------- Global toggles ----------
DO_WARMSTART = False  # one optional pass over synthetic images before fine‑tune

# ---------- Helper: convert ru_maxrss to MB (Linux vs macOS units differ) ----------
def _ru_maxrss_to_mb(v):
    """Convert resource.getrusage(...).ru_maxrss to MB across platforms.
    On Linux this value is in kilobytes; on macOS it's in bytes.
    """
    return (v / (1024.0 * 1024.0)) if sys.platform == "darwin" else (v / 1024.0)

def rss_now_mb():
    """Point‑in‑time resident set size (MB); 'what am I using right now?'
    Uses psutil to query this process' RSS immediately.
    """
    return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)

def peak_abs_mb():
    """Absolute peak RSS (MB) the kernel has recorded for this process so far."""
    return _ru_maxrss_to_mb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def peak_delta_mb(baseline_ru_maxrss):
    """Increase in peak RSS (MB) relative to a baseline ru_maxrss snapshot.
    We take the baseline immediately before starting fine‑tuning so this
    quantity reflects training‑only memory growth.
    """
    return max(0.0, _ru_maxrss_to_mb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                    - _ru_maxrss_to_mb(baseline_ru_maxrss))

# ---------- CLI arguments ----------
ap = argparse.ArgumentParser(description="On‑device continual learning demo (CPU‑only)")
ap.add_argument("--workers",     type=int, default=0,   help="DataLoader worker threads (0 or small integer)")
ap.add_argument("--epochs_real", type=int, default=3,   help="Epochs (passes) over your real photos")
ap.add_argument("--batch",       type=int, default=16,  help="Batch size")
ap.add_argument("--synth",       type=int, default=100, help="How many synthetic images to generate")
ap.add_argument("--img",         type=int, default=128, help="Square image size fed to the model (pixels)")
args = ap.parse_args()  # parse argv into a simple Namespace

# ---------- Paths & device ----------
DEVICE  = torch.device("cpu")           # we constrain this project to CPU‑only
PROJECT = Path(".")                     # project root (current folder)
SYN     = PROJECT / "data_synth"        # synthetic images live here
REAL    = PROJECT / "data_real"         # put your ~20 labeled real photos here
PLOTS   = PROJECT / "plots"             # PNG charts will be written here
SYN.mkdir(exist_ok=True)                # create folders if they don't exist
PLOTS.mkdir(exist_ok=True)

# ---------- Class labels ----------
# Four classes: two colors (orange/blue) × two shapes (square/circle).
CLASSES   = ["orange_square", "blue_square", "orange_circle", "blue_circle"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}  # map class string -> integer id

# ---------- Image preprocessing ----------
# Normalize images the same way the pretrained model expects (ImageNet stats).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE      = int(args.img)  # size can be tuned via --img

# Training‑time transforms (mild augmentation keeps the model robust)
tf_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),                     # resize to model input
    transforms.ColorJitter(brightness=0.25, contrast=0.15),      # mild photometric jitter
    transforms.RandomRotation(8, fill=(210, 210, 210)),          # gentle rotation; avoid black borders
    transforms.ToTensor(),                                       # HWC [0..255] -> CHW [0..1]
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # ImageNet normalization
])

# Evaluation transforms (no augmentation)
tf_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---------- Dataset reading a flat folder of images ----------
class LabeledFolder(Dataset):
    """Reads ./data_real/*.{jpg,png} where file names start with a class key,
    e.g., 'orange_square_001.jpg'. The first two underscore‑separated tokens
    form the class name ('orange_square').
    """
    def __init__(self, path: Path, transform):
        self.transform = transform
        self.paths, self.labels = [], []
        for p in sorted(path.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            key = "_".join(p.stem.split("_")[:2])   # 'orange_square_12' -> 'orange_square'
            if key in LABEL_MAP:
                self.paths.append(p)
                self.labels.append(LABEL_MAP[key])

    def __len__(self):  # number of images
        return len(self.paths)

    def __getitem__(self, i):  # return (Tensor image, Tensor label)
        img = ImageOps.exif_transpose(Image.open(self.paths[i])).convert("RGB")
        return self.transform(img), torch.tensor(self.labels[i])

# ---------- Generate synthetic "lab" images (clean colored shapes) ----------
def gen_synth(n: int = 100):
    """Create n synthetic images split roughly evenly across the four classes.
    We draw a single colored shape on a black background to make the task
    simple and controlled.
    """
    per = max(1, n // len(CLASSES))  # images per class (approx equal)
    for cls in CLASSES:
        color, shape = cls.split("_")
        for k in range(per):
            img = Image.new("RGB", (224, 224), "black")  # start with black
            d = ImageDraw.Draw(img)
            if color == "orange":
                c = (240, 140, 0)
            else:  # blue
                c = (0, 110, 255)
            if shape == "square":
                d.rectangle([60, 60, 164, 164], fill=c)
            else:  # circle
                d.ellipse([60, 60, 164, 164], fill=c)
            img.save(SYN / f"{cls}_{k}.png")

# ---------- Accuracy helper ----------
def accuracy(model, loader):
    """Compute simple classification accuracy over a loader."""
    model.eval()                                   # disable dropout/batchnorm updates
    correct, total = 0, 0
    with torch.no_grad():                          # no gradients for evaluation
        for x, y in loader:                        # loop over minibatches
            logits = model(x.to(DEVICE))          # forward pass on CPU
            pred   = logits.argmax(1).cpu()       # predicted class index
            correct += (pred == y).sum().item()   # add correct count
            total   += y.numel()                  # add batch size
    return (correct / total) if total > 0 else math.nan

# ---------- Checkpoint helper (ext4 vs tmpfs timing) ----------
def timed_checkpoint(model, path: Path):
    """Save the model twice and time it:
       1) ext4 (project folder on disk)
       2) tmpfs (/dev/shm — RAM disk)
       We use an atomic rename to avoid partial files.
    """
    # ext4 write
    t0 = time.perf_counter()
    torch.save(model.state_dict(), path.with_suffix(".pt.tmp"))
    os.replace(path.with_suffix(".pt.tmp"), path.with_suffix(".pt"))
    ext4_ms = (time.perf_counter() - t0) * 1000.0

    # tmpfs write
    t1       = time.perf_counter()
    tmp_tmp  = Path("/dev/shm/model_tmp.pt.tmp")
    tmp_dst  = Path("/dev/shm/model_tmp.pt")
    torch.save(model.state_dict(), tmp_tmp)
    os.replace(tmp_tmp, tmp_dst)
    tmpfs_ms = (time.perf_counter() - t1) * 1000.0
    try:
        tmp_dst.unlink()
    except Exception:
        pass
    return ext4_ms, tmpfs_ms

# ---------- Main experiment ----------
def main():
    # Fail fast if the user forgot to place real data.
    assert REAL.exists(), "Put ~20 labeled photos in ./data_real named orange_square_*.jpg etc."

    # Create synthetic data once if the folder is empty.
    if not any(SYN.iterdir()):
        gen_synth(args.synth)

    # Build datasets and data loaders.
    ds_syn        = LabeledFolder(SYN,  tf_eval)   # synthetic uses eval transforms
    ds_real_train = LabeledFolder(REAL, tf_train)  # augment during fine‑tune
    ds_real_eval  = LabeledFolder(REAL, tf_eval)   # clean evaluation

    dl_syn  = DataLoader(ds_syn,        batch_size=args.batch, shuffle=True,  num_workers=args.workers)
    dl_real = DataLoader(ds_real_train, batch_size=args.batch, shuffle=True,  num_workers=args.workers)
    dl_eval = DataLoader(ds_real_eval,  batch_size=args.batch, shuffle=False, num_workers=0)

    # Print a quick class‑count sanity check.
    from collections import Counter
    print("real class counts:", Counter([LABEL_MAP["_".join(p.stem.split("_")[:2])]
                                        for p in REAL.iterdir()
                                        if p.suffix.lower() in (".jpg", ".jpeg", ".png")]))
    # Load MobileNet‑V2 pretrained on ImageNet.
    model = models.mobilenet_v2(weights="IMAGENET1K_V2")

    # Freeze the feature extractor (all layers before the classifier) to reduce
    # the number of trainable parameters, gradients, and optimizer state -> lower RAM.
    for p in model.features.parameters():
        p.requires_grad = False

    # Replace the 1000‑class head with a 4‑class linear layer.
    model.classifier[1] = nn.Linear(1280, len(CLASSES))
    model.to(DEVICE)

    # Optimizer / loss. Only the classifier's parameters will update.
    opt     = torch.optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Optional warm‑start pass over synthetic data.
    if DO_WARMSTART:
        model.train()
        for x, y in dl_syn:
            opt.zero_grad()
            logits = model(x.to(DEVICE))
            loss   = loss_fn(logits, y.to(DEVICE))
            loss.backward()
            opt.step()

    # Baseline accuracy before fine‑tune.
    pre_acc = accuracy(model, dl_eval)

    # Take the baseline ru_maxrss right before we start fine‑tuning.
    baseline_ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Prepare CSV rows (header + data per epoch).
    rows = [["epoch", "train_acc", "sec", "mem_peak_abs_MB", "mem_delta_MB", "mem_now_MB",
             "ext4_ms", "tmpfs_ms"]]

    # Fine‑tuning loop (few epochs over real photos).
    for epoch in range(1, args.epochs_real + 1):
        t0 = time.time()          # start wall‑clock timer
        model.train()             # enable training mode
        for x, y in dl_real:      # one pass over the real photos
            opt.zero_grad()       # clear grads
            logits = model(x.to(DEVICE))
            loss   = loss_fn(logits, y.to(DEVICE))
            loss.backward()       # compute grads
            opt.step()            # update classifier weights
        sec = time.time() - t0    # elapsed time for this epoch

        # Memory views
        nowMB   = rss_now_mb()            # "now" RSS after this epoch
        peakMB  = peak_abs_mb()           # absolute peak so far
        deltaMB = peak_delta_mb(baseline_ru)  # increase over the baseline

        # Quick eval on the tiny set (acts as both train/val proxy).
        tr_acc = accuracy(model, dl_eval)

        # Timed checkpoint writes (ext4 vs tmpfs).
        ext4_ms, tmpfs_ms = timed_checkpoint(model, PROJECT / "model_final")

        # One‑line epoch log for the terminal.
        print(f"epoch {epoch}: acc={tr_acc:.2f} time={sec:.2f}s "
              f"peak={peakMB:.1f}MB delta={deltaMB:.1f}MB now={nowMB:.1f}MB "
              f"ext4={ext4_ms:.1f}ms tmpfs={tmpfs_ms:.1f}ms")

        # Append row for the CSV.
        rows.append([epoch, f"{tr_acc:.4f}", f"{sec:.3f}", f"{peakMB:.1f}", f"{deltaMB:.1f}",
                     f"{nowMB:.1f}", f"{ext4_ms:.2f}", f"{tmpfs_ms:.2f}"])

    # Write results.csv for reproducibility and plotting.
    with open(PROJECT / "results.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # Plot: accuracy + three memory views.
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.read_csv(PROJECT / "results.csv")

        # Accuracy vs epoch
        plt.figure()
        plt.plot(df["epoch"], df["train_acc"], marker="o")
        plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Accuracy vs epoch")
        plt.savefig(PLOTS / "accuracy_vs_epoch.png", dpi=160)

        # Memory: delta vs epoch
        plt.figure()
        plt.plot(df["epoch"], df["mem_delta_MB"], marker="o")
        plt.xlabel("epoch"); plt.ylabel("delta RSS (MB)"); plt.title("Peak Δ vs epoch (above baseline)")
        plt.savefig(PLOTS / "mem_delta_vs_epoch.png", dpi=160)

        # Memory: now vs epoch
        plt.figure()
        plt.plot(df["epoch"], df["mem_now_MB"], marker="o")
        plt.xlabel("epoch"); plt.ylabel("RSS now (MB)"); plt.title("RSS now vs epoch")
        plt.savefig(PLOTS / "mem_now_vs_epoch.png", dpi=160)

        # Memory: absolute peak vs epoch
        plt.figure()
        plt.plot(df["epoch"], df["mem_peak_abs_MB"], marker="o")
        plt.xlabel("epoch"); plt.ylabel("peak RSS (MB)"); plt.title("Peak RSS vs epoch (ru_maxrss)")
        plt.savefig(PLOTS / "mem_peak_vs_epoch.png", dpi=160)

        print("Wrote plots to", PLOTS)
    except Exception as e:
        print("Plotting skipped:", e)

    # Summarize against success criteria.
    final_acc  = float(rows[-1][1])                     # last epoch accuracy
    total_time = sum(float(r[2]) for r in rows[1:])     # sum of per‑epoch seconds
    overall_pk = max(float(r[3]) for r in rows[1:])     # maximum absolute peak across epochs
    ok_time    = total_time < 60.0
    ok_mem     = overall_pk < 400.0

    # Friendly summary for the report.
    print(f"\nPre‑tune acc on real: {pre_acc:.2f}")                  # baseline accuracy
    print(f"Final acc on real:    {final_acc:.2f}")                  # last‑epoch accuracy
    print(f"Total fine‑tune time: {total_time:.1f}s  (target < 60s)  -> {'OK' if ok_time else 'OVER'}")
    print(f"Peak RSS (absolute):  {overall_pk:.1f}MB (target < 400MB) -> {'OK' if ok_mem else 'OVER'}")
    print("See results.csv and plots/ for accuracy + three memory views.")

# ---------- Standard Python entry point ----------
if __name__ == "__main__":
    main()