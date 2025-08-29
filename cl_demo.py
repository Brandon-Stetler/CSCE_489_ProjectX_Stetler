#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cl_demo.py — On-device continual learning demo (CPU-only)

This script shows whether a small robot could update ("fine-tune") its vision
model on its own hardware without sending images to the cloud.

Flow:
  1) Generate a clean synthetic dataset of colored shapes (lab-like data).
  2) Load your 20 real photos of the same shapes (field-like data).
  3) Load a small pretrained image model (MobileNet-V2), replace the last layer,
     and fine-tune only that layer on the 20 real photos.
  4) Measure wall-clock time per epoch, peak RAM, and checkpoint write times.
  5) Save results to a CSV and simple plots.

Success bar (checked at the end):
  - Total fine-tune < 60 s
  - Peak RAM < 400 MB
"""

# ---------- Imports ----------
import os                   # OS utilities (paths, process info)
import time                 # timing (wall-clock seconds)
import csv                  # write results.csv
import argparse             # command-line options like --workers
import math                 # safe division / NaN handling
from pathlib import Path    # filesystem paths that work on any OS

import psutil               # read current process memory (peak RSS)
import torch
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.mkldnn.enabled = False
from torch import nn        # neural-network layers/losses
from torch.utils.data import Dataset, DataLoader  # data plumbing
from torchvision import models, transforms        # pretrained models + image transforms
from PIL import Image, ImageDraw, ImageOps        # image I/O, drawing, EXIF-aware rotation

DO_WARMSTART = False  # add near top

# ---------- CLI arguments ----------
ap = argparse.ArgumentParser()                                # create parser
ap.add_argument("--workers", type=int, default=0,             # DataLoader worker threads
                help="DataLoader workers: 0 (simplest) or 4 (parallel I/O)")
ap.add_argument("--epochs_real", type=int, default=3,         # how many passes over real photos
                help="Number of epochs on the 20 real photos")
ap.add_argument("--batch", type=int, default=16,              # minibatch size
                help="Batch size for training and evaluation")
ap.add_argument("--synth", type=int, default=100,             # how many synthetic images to make
                help="Number of synthetic images to generate")
args = ap.parse_args()                                        # parse argv into 'args'

# ---------- Paths & device ----------
DEVICE = torch.device("cpu")                                  # force CPU (fits a low-end robot)
PROJECT = Path(".")                                           # project root (current folder)
SYN = PROJECT / "data_synth"                                  # synthetic images go here
REAL = PROJECT / "data_real"                                  # your 20 photos go here
PLOTS = PROJECT / "plots"                                     # output charts go here
SYN.mkdir(exist_ok=True)                                      # make folders if missing
PLOTS.mkdir(exist_ok=True)
assert REAL.exists(), "Put ~20 real photos in ./data_real named orange_square_*.jpg etc."

# ---------- Class labels ----------
# Four classes: two colors (orange/blue) × two shapes (square/circle).
classes = ["orange_square", "blue_square", "orange_circle", "blue_circle"]
label_map = {c: i for i, c in enumerate(classes)}             # map class name -> integer id

# ---------- Image preprocessing ----------
# Normalize images the same way the pretrained model expects (ImageNet stats).
IMG_SIZE = 144  

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

tf_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.25, contrast=0.15),   # no saturation
    transforms.RandomRotation(8, fill=(210, 210, 210)),       # close to paper, not black
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

tf_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---------- Simple dataset that reads a flat folder of images ----------
class LabeledFolder(Dataset):
    def __init__(self, path: Path, transform):
        self.transform = transform
        self.paths, self.labels = [], []
        for p in sorted(path.iterdir()):
            if p.suffix.lower() not in [".jpg",".jpeg",".png"]:
                continue
            key = "_".join(p.stem.split("_")[:2])
            if key in label_map:
                self.paths.append(p)
                self.labels.append(label_map[key])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = ImageOps.exif_transpose(Image.open(self.paths[i])).convert("RGB")
        return self.transform(img), torch.tensor(self.labels[i])
# ---------- Generate synthetic "lab" images (clean colored shapes) ----------
def gen_synth(n: int = 100):
    """Create n synthetic images split evenly across the 4 classes."""
    per = max(1, n // len(classes))                           # images per class (roughly equal)
    for cls in classes:                                       # loop over each class name
        color, shape = cls.split("_")                         # separate 'orange' and 'square'
        for k in range(per):                                  # produce 'per' images for this class
            img = Image.new("RGB", (224, 224), "black")       # start with black background
            d = ImageDraw.Draw(img)                           # drawing context
            if color == "orange":
                c = (240, 140, 0)                             # RGB for orange (roughly)
            else:  # blue
                c = (0, 110, 255)                             # RGB for blue
            if shape == "square":
                d.rectangle([60, 60, 164, 164], fill=c)       # draw a filled square
            else:
                d.ellipse([60, 60, 164, 164], fill=c)         # draw a filled circle
            img.save(SYN / f"{cls}_{k}.png")                  # write to data_synth/

# ---------- Utility: current process memory in MB ----------
#def peak_rss_mb():
#    """Return resident set size (RAM in use) for this process, in MB."""
#   return psutil.Process(os.getpid()).memory_info().rss // (1024 * 1024)
import sys
import resource
def peak_rss_mb():
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024*1024) if sys.platform == "darwin" else peak / 1024.0
    except Exception:
        return psutil.Process(os.getpid()).memory_info().rss / (1024*1024)

# ---------- Utility: accuracy on a DataLoader ----------
def accuracy(model, loader):
    """Compute simple classification accuracy over a loader."""
    model.eval()                                              # eval mode = no dropout/batchnorm updates
    correct = 0                                               # count of correct predictions
    total = 0                                                 # total examples seen
    with torch.no_grad():                                     # disable gradients for speed/memory
        for x, y in loader:                                   # loop over batches
            logits = model(x.to(DEVICE))                      # forward pass on CPU
            pred = logits.argmax(1).cpu()                     # predicted class indices
            correct += (pred == y).sum().item()               # add correct predictions
            total += y.numel()                                # add batch size
    return (correct / total) if total > 0 else math.nan       # guard against divide-by-zero

# ---------- Utility: checkpoint to ext4 vs tmpfs and time each ----------
def timed_checkpoint(model, path: Path):
    """Save the model twice and time it:
       1) ext4 (normal disk in your project folder)
       2) tmpfs (/dev/shm — RAM disk)
       Use atomic rename to avoid partial files.
    """
    # --- ext4 write ---
    t0 = time.perf_counter()                                  # high-resolution start time
    torch.save(model.state_dict(), path.with_suffix(".pt.tmp")) # write to temp file
    os.replace(path.with_suffix(".pt.tmp"),                   # atomic rename = crash-safe
               path.with_suffix(".pt"))
    ext4_ms = (time.perf_counter() - t0) * 1000.0             # elapsed time in milliseconds

    # --- tmpfs write (/dev/shm is memory-backed) ---
    t1 = time.perf_counter()                                  # new timer
    tmpfs_tmp = Path("/dev/shm/model_tmp.pt.tmp")             # temp file in RAM
    tmpfs_dst = Path("/dev/shm/model_tmp.pt")                 # final file in RAM
    torch.save(model.state_dict(), tmpfs_tmp)                 # write model state
    os.replace(tmpfs_tmp, tmpfs_dst)                          # atomic rename in RAM
    tmpfs_ms = (time.perf_counter() - t1) * 1000.0            # elapsed ms for tmpfs
    try:
        tmpfs_dst.unlink()                                    # clean up RAM file
    except:
        pass
    return ext4_ms, tmpfs_ms                                  # return both timings

# ---------- Main experiment ----------
def main():
    # 1) Ensure synthetic data exists; if not, create it once.
    if not any(SYN.iterdir()):                                # folder empty?
        gen_synth(args.synth)                                 # generate synthetic shapes

    # 2) Build datasets and loaders (synthetic once, real for adaptation).
    ds_syn        = LabeledFolder(SYN, tf_eval)
    ds_real_train = LabeledFolder(REAL, tf_train)
    ds_real_eval  = LabeledFolder(REAL, tf_eval)

    dl_syn  = DataLoader(ds_syn,  batch_size=args.batch, shuffle=True,  num_workers=args.workers)
    dl_real = DataLoader(ds_real_train, batch_size=args.batch, shuffle=True,  num_workers=args.workers)
    dl_eval = DataLoader(ds_real_eval, batch_size=args.batch, shuffle=False, num_workers=0)

    from collections import Counter
    print("real class counts:", Counter([label_map["_".join(p.stem.split("_")[:2])] 
                                        for p in REAL.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png"]]))

    # 3) Load MobileNet-V2 pretrained on ImageNet and replace the last layer.
    model = models.mobilenet_v2(weights="IMAGENET1K_V2")      # download/use pretrained weights
    for p in model.features.parameters():                     # freeze all earlier layers
        p.requires_grad = False                               # only train the classifier head
    # ...except the last feature block (give the model a little capacity to adapt)
    #for p in model.features[-1].parameters():
    #    p.requires_grad = True

    model.classifier[1] = nn.Linear(1280, len(classes))       # swap 1000-class head -> 4 classes
    model.to(DEVICE)                                          # move to CPU device

    # 4) Set up optimizer and loss (only the new head’s parameters will update).
    # BEFORE
    # opt = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    opt = torch.optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    #opt = torch.optim.Adam(list(model.classifier.parameters()) + list(model.features[-1].parameters()),
    #lr=5e-4, weight_decay=1e-4)  # Adam = fast convergence
    loss_fn = nn.CrossEntropyLoss()                           # standard multi-class loss

    # 5) Quick "lab calibration" pass on synthetic data (optional warm-start).
    if DO_WARMSTART:
        model.train()                                             # training mode
        for x, y in dl_syn:                                       # iterate synthetic batches once
            opt.zero_grad()                                       # clear previous gradients
            logits = model(x.to(DEVICE))                          # forward pass
            loss = loss_fn(logits, y.to(DEVICE))                  # compute loss
            loss.backward()                                       # backprop to compute gradients
            opt.step()                                            # update the head’s weights

    # 6) Measure accuracy on real photos *before* fine-tuning (baseline).
    pre_acc = accuracy(model, dl_eval)

    # 7) Fine-tune on the 20 real photos for a few epochs with logging.
    rows = [["epoch", "train_acc", "val_acc", "sec", "peak_MB", "ext4_ms", "tmpfs_ms"]] # CSV header
    for epoch in range(1, args.epochs_real + 1):              # e.g., 1..3
        t0 = time.time()                                      # start epoch timer
        model.train()                                         # set train mode
        for x, y in dl_real:                                  # iterate real data
            opt.zero_grad()                                   # reset gradients
            logits = model(x.to(DEVICE))                      # forward
            loss = loss_fn(logits, y.to(DEVICE))              # compute loss
            loss.backward()                                   # backward pass
            opt.step()                                        # update classifier weights
        sec = time.time() - t0                                # seconds for this epoch

        peakMB = peak_rss_mb()                                # measure RAM usage now
        tr_acc = accuracy(model, dl_eval)                     # accuracy on (tiny) training set
        va_acc = tr_acc                                       # no separate val set → reuse as proxy

        ext4_ms, tmpfs_ms = timed_checkpoint(model,           # time disk vs RAM checkpoint
                                             PROJECT / "model_final")

        # Print a one-line log for this epoch (easy to read in terminal).
        print(f"epoch {epoch}: acc={tr_acc:.2f} time={sec:.2f}s "
              f"peak={peakMB}MB ext4={ext4_ms:.1f}ms tmpfs={tmpfs_ms:.1f}ms")

        # Append a row to write later into results.csv.
        rows.append([epoch, f"{tr_acc:.4f}", f"{va_acc:.4f}",
                     f"{sec:.3f}", peakMB, f"{ext4_ms:.2f}", f"{tmpfs_ms:.2f}"])

    # 8) Save the CSV so results are reproducible and easy to plot later.
    with open(PROJECT / "results.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # 9) Make two simple plots: accuracy vs epoch and peak RAM vs epoch.
    try:
        import pandas as pd                                  # only used for plotting convenience
        import matplotlib.pyplot as plt                      # plotting library

        df = pd.read_csv(PROJECT / "results.csv")           # read the CSV we just wrote

        plt.figure()
        plt.plot(df["epoch"], df["train_acc"], marker="o")  # accuracy curve
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title("Accuracy vs epoch")
        (PLOTS / "accuracy_vs_epoch.png").parent.mkdir(exist_ok=True)
        plt.savefig(PLOTS / "accuracy_vs_epoch.png", dpi=160)

        plt.figure()
        plt.plot(df["epoch"], df["peak_MB"], marker="o")    # memory curve
        plt.xlabel("epoch")
        plt.ylabel("peak RSS (MB)")
        plt.title("Peak RSS vs epoch")
        plt.savefig(PLOTS / "mem_vs_epoch.png", dpi=160)

        print("Wrote plots to", PLOTS)                      # confirm where images went
    except Exception as e:
        # If matplotlib/pandas aren’t installed, skip plotting but keep CSV.
        print("Plotting skipped:", e)

    # 10) Summarize against success criteria (time + memory thresholds).
    final_acc = float(rows[-1][1])                           # last epoch’s accuracy
    total_time = sum(float(r[3]) for r in rows[1:])          # sum of per-epoch seconds
    ok_time = total_time < 60.0                              # target: < 60 seconds total
    ok_mem  = max(int(r[4]) for r in rows[1:]) < 400         # target: < 400 MB peak

    # Print a short, decision-oriented summary for your report.
    print(f"\nPre-tune acc on real: {pre_acc:.2f}")
    print(f"Final acc on real:    {final_acc:.2f}")
    print(f"Total fine-tune time: {total_time:.1f}s  (target < 60s)  -> {'OK' if ok_time else 'OVER'}")
    print(f"Peak RSS during tune: {max(int(r[4]) for r in rows[1:])}MB (target < 400MB) -> {'OK' if ok_mem else 'OVER'}")
    print("Check results.csv and plots/ for details.")

# Standard Python entry point so the script only runs when executed directly.
if __name__ == "__main__":
    main()
