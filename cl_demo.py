#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cl_demo.py — On-device continual learning demo (CPU-only)

Goal: fine-tune only a small classifier head on top of a frozen MobileNetV2
      using ~20 real photos, while staying under:
        • total fine-tune time  < 60 s
        • absolute peak RSS    < 400 MB

Key implementation details that reduce RAM:
  - All feature extraction (model.features) is done under torch.no_grad()
  - model.features kept in eval() during training to freeze batchnorm
  - Only the classifier head has gradients & gets updated
"""

# ---------- imports & low-thread env ----------
import os, sys, time, csv, math, argparse
from pathlib import Path

# keep BLAS libs from over-threading on CPU
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import psutil, resource
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageOps

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.mkldnn.enabled = False  # safer on minimal CPUs

# ---------- CLI ----------
ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 simplest)")
ap.add_argument("--epochs_real", type=int, default=4, help="Epochs over the real photos")
ap.add_argument("--batch", type=int, default=2, help="Batch size (2 is safe for 128px)")
ap.add_argument("--synth", type=int, default=100, help="# synthetic images to generate (once)")
ap.add_argument("--img", type=int, default=128, help="Square resize (e.g., 128, 144, 160)")
args = ap.parse_args()

# ---------- paths ----------
DEVICE = torch.device("cpu")
PROJECT = Path(".")
SYN = PROJECT / "data_synth"
REAL = PROJECT / "data_real"
PLOTS = PROJECT / "plots"
SYN.mkdir(exist_ok=True); PLOTS.mkdir(exist_ok=True)
assert REAL.exists(), "Put ~20 real photos in ./data_real named like orange_square_*.jpg"

# ---------- labels ----------
classes = ["orange_square", "blue_square", "orange_circle", "blue_circle"]
label_map = {c: i for i, c in enumerate(classes)}

# ---------- transforms ----------
IMG_SIZE = args.img
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

tf_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.25, contrast=0.15),
    transforms.RandomRotation(8, fill=(210, 210, 210)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
tf_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---------- dataset ----------
class LabeledFolder(Dataset):
    def __init__(self, path: Path, transform):
        self.transform = transform
        self.paths, self.labels = [], []
        for p in sorted(path.iterdir()):
            if p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            key = "_".join(p.stem.split("_")[:2])
            if key in label_map:
                self.paths.append(p)
                self.labels.append(label_map[key])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = ImageOps.exif_transpose(Image.open(self.paths[i])).convert("RGB")
        return self.transform(img), torch.tensor(self.labels[i])

# ---------- synth generation ----------
def gen_synth(n: int = 100):
    per = max(1, n // len(classes))
    for cls in classes:
        color, shape = cls.split("_")
        for k in range(per):
            img = Image.new("RGB", (224, 224), "black")
            d = ImageDraw.Draw(img)
            c = (240, 140, 0) if color == "orange" else (0, 110, 255)
            if shape == "square":
                d.rectangle([60, 60, 164, 164], fill=c)
            else:
                d.ellipse([60, 60, 164, 164], fill=c)
            img.save(SYN / f"{cls}_{k}.png")

# ---------- memory helpers ----------
def _ru_to_mb(v):
    # ru_maxrss is KB on Linux, bytes on macOS
    return (v / (1024 * 1024)) if sys.platform == "darwin" else (v / 1024.0)

def rss_now_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def peak_abs_mb():
    return _ru_to_mb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

# set after imports (contains model & dataloaders once built -> "baseline before training")
BASELINE_ABS_MB = peak_abs_mb()

# ---------- accuracy ----------
def accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(DEVICE))
            pred = logits.argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.numel()
    return (correct / total) if total > 0 else float("nan")

# ---------- checkpoint timing ----------
def timed_checkpoint(model, path: Path):
    t0 = time.perf_counter()
    torch.save(model.state_dict(), path.with_suffix(".pt.tmp"))
    os.replace(path.with_suffix(".pt.tmp"), path.with_suffix(".pt"))
    ext4_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    tmpfs_tmp = Path("/dev/shm/model_tmp.pt.tmp")
    tmpfs_dst = Path("/dev/shm/model_tmp.pt")
    try:
        torch.save(model.state_dict(), tmpfs_tmp)
        os.replace(tmpfs_tmp, tmpfs_dst)
        tmpfs_ms = (time.perf_counter() - t1) * 1000.0
    finally:
        try: tmpfs_dst.unlink()
        except: pass
    return ext4_ms, tmpfs_ms

# ---------- main ----------
def main():
    # generate synth once
    if not any(SYN.iterdir()):
        gen_synth(args.synth)

    # data
    ds_syn        = LabeledFolder(SYN, tf_eval)
    ds_real_train = LabeledFolder(REAL, tf_train)
    ds_real_eval  = LabeledFolder(REAL, tf_eval)

    dl_syn  = DataLoader(ds_syn,  batch_size=args.batch, shuffle=True,  num_workers=args.workers)
    dl_real = DataLoader(ds_real_train, batch_size=args.batch, shuffle=True,  num_workers=args.workers)
    dl_eval = DataLoader(ds_real_eval,  batch_size=args.batch, shuffle=False, num_workers=0)

    # show class balance
    from collections import Counter
    print("real class counts:", Counter([label_map["_".join(p.stem.split("_")[:2])]
                                        for p in REAL.iterdir()
                                        if p.suffix.lower() in [".jpg",".jpeg",".png"]]))

    # model: MobileNetV2 pretrained
    try:
        from torchvision.models import MobileNet_V2_Weights
        weights = MobileNet_V2_Weights.IMAGENET1K_V2
        model = models.mobilenet_v2(weights=weights)
    except Exception:
        model = models.mobilenet_v2(weights="IMAGENET1K_V2")

    # freeze feature extractor
    for p in model.features.parameters():
        p.requires_grad = False

    # replace head
    model.classifier[1] = nn.Linear(1280, len(classes))
    model.to(DEVICE)

    # optimizer & loss (head only)
    opt = torch.optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # optional tiny warm-start on synth (still memory-safe: features under no_grad)
    DO_WARMSTART = False
    if DO_WARMSTART and len(dl_syn) > 0:
        model.train()
        model.features.eval()
        for x, y in dl_syn:
            opt.zero_grad()
            with torch.no_grad():
                f = model.features(x.to(DEVICE))
                f = f.mean(dim=(2, 3))           # global average pool to 1280
            logits = model.classifier(f)         # grads only for classifier
            loss = loss_fn(logits, y.to(DEVICE))
            loss.backward()
            opt.step()

    # pre-tune accuracy
    pre_acc = accuracy(model, dl_eval)

    # train on real data with frozen features under no_grad
    rows = [["epoch", "train_acc", "sec", "peak_abs_MB", "delta_MB", "rss_now_MB", "ext4_ms", "tmpfs_ms"]]
    for epoch in range(1, args.epochs_real + 1):
        t0 = time.time()
        model.train()
        model.features.eval()                    # keep BN frozen in features
        for x, y in dl_real:
            opt.zero_grad()

            # ---- big memory win: do not build graph for feature extractor ----
            with torch.no_grad():
                f = model.features(x.to(DEVICE)) # [N, 1280, h, w]
                f = f.mean(dim=(2, 3))           # global average pool -> [N, 1280]
            # -----------------------------------------------------------------

            logits = model.classifier(f)         # classifier has grads
            loss = loss_fn(logits, y.to(DEVICE))
            loss.backward()
            opt.step()

        sec = time.time() - t0

        # memory views
        peak_abs = peak_abs_mb()
        delta = max(0.0, peak_abs - BASELINE_ABS_MB)
        now = rss_now_mb()

        tr_acc = accuracy(model, dl_eval)

        ext4_ms, tmpfs_ms = timed_checkpoint(model, PROJECT / "model_final")

        print(f"epoch {epoch}: acc={tr_acc:.2f} time={sec:.2f}s "
              f"peak={peak_abs:.1f}MB delta={delta:.1f}MB now={now:.1f}MB "
              f"ext4={ext4_ms:.1f}ms tmpfs={tmpfs_ms:.1f}ms")

        rows.append([epoch,
                     f"{tr_acc:.4f}",
                     f"{sec:.3f}",
                     f"{peak_abs:.1f}",
                     f"{delta:.1f}",
                     f"{now:.1f}",
                     f"{ext4_ms:.1f}",
                     f"{tmpfs_ms:.1f}"])

    # write CSV
    with open(PROJECT / "results.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # plots
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        df = pd.read_csv(PROJECT / "results.csv")

        plt.figure()
        plt.plot(df["epoch"], df["train_acc"], marker="o")
        plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Accuracy vs epoch")
        plt.savefig(PLOTS / "accuracy_vs_epoch.png", dpi=160)

        plt.figure()
        plt.plot(df["epoch"], df["peak_abs_MB"], marker="o")
        plt.xlabel("epoch"); plt.ylabel("peak RSS (MB)"); plt.title("Peak RSS vs epoch (ru_maxrss)")
        plt.savefig(PLOTS / "mem_peak_vs_epoch.png", dpi=160)

        plt.figure()
        plt.plot(df["epoch"], df["rss_now_MB"], marker="o")
        plt.xlabel("epoch"); plt.ylabel("RSS now (MB)"); plt.title("RSS now vs epoch")
        plt.savefig(PLOTS / "mem_now_vs_epoch.png", dpi=160)

        plt.figure()
        plt.plot(df["epoch"], df["delta_MB"], marker="o")
        plt.xlabel("epoch"); plt.ylabel("Δ peak above baseline (MB)")
        plt.title("Peak Δ vs epoch (above baseline)")
        plt.savefig(PLOTS / "mem_delta_vs_epoch.png", dpi=160)

        print("Wrote plots to", PLOTS)
    except Exception as e:
        print("Plotting skipped:", e)

    # summary vs targets
    final_acc = float(rows[-1][1])
    total_time = sum(float(r[2]) for r in rows[1:])
    max_peak = max(float(r[3]) for r in rows[1:])
    ok_time = total_time < 60.0
    ok_mem  = max_peak < 400.0

    print(f"\nPre-tune acc on real: {pre_acc:.2f}")
    print(f"Final acc on real:    {final_acc:.2f}")
    print(f"Total fine-tune time: {total_time:.1f}s  (target < 60s)  -> {'OK' if ok_time else 'OVER'}")
    print(f"Peak RSS during tune: {max_peak:.1f}MB (target < 400MB) -> {'OK' if ok_mem else 'OVER'}")
    print("See results.csv and plots/ for accuracy + three memory views.")

# entry
if __name__ == "__main__":
    main()