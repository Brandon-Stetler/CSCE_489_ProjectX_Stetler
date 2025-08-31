# On‑Device Continual Learning

> **Goal:** Show that a small robot (no cloud) can fine‑tune a lightweight vision model on its own CPU while measuring **OS‑relevant metrics**: wall‑clock time, **resident memory** (RSS), and **filesystem latency** (ext4 vs tmpfs).

## Repository layout
```
.
├─ cl_demo.py           # main script (fully commented, creates CSV + plots)
├─ requirements.txt     # minimal Python dependencies (CPU-only)
├─ run.sh               # one‑liner to reproduce the “final” run
├─ README.md            # this file
├─ data_real/           # put ~20 labeled photos here (see below)
├─ data_synth/          # will be auto‑filled with synthetic images
└─ plots/               # PNG charts are saved here
```

### Your data: `data_real/`
Place ~20 photos with names that **start with a class key**:
- `orange_square_*.jpg`
- `blue_square_*.jpg`
- `orange_circle_*.jpg`
- `blue_circle_*.jpg`

Example filenames:
```
orange_square_01.jpg
orange_square_02.jpg
blue_square_01.jpg
orange_circle_01.jpg
...
```

The script maps the first two underscore‑separated tokens to a label (e.g., `orange_square_foo.jpg` → class `orange_square`).

## Quick start

1) **Create/activate a venv (recommended).**
```bash
python3 -m venv .venv && source .venv/bin/activate
```

2) **Install dependencies.**
```bash
pip install -r requirements.txt
```

3) **Add your 20 photos** to `./data_real/` (naming as above).

4) **Run the demo** with sane CPU‑only defaults:
```bash
MALLOC_ARENA_MAX=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u cl_demo.py --epochs_real 4 --batch 4 --workers 0 --img 128 | tee run.log
```

5) **Inspect outputs**:
- `results.csv` (per‑epoch metrics)
- `plots/accuracy_vs_epoch.png`
- `plots/mem_peak_vs_epoch.png` (absolute peak RSS, **ru_maxrss**)
- `plots/mem_now_vs_epoch.png`   (point‑in‑time RSS, **psutil**)
- `plots/mem_delta_vs_epoch.png` (peak minus baseline, **training‑only delta**)
- Terminal log: `epoch N: acc=… time=…s peak=…MB delta=…MB now=…MB ext4=…ms tmpfs=…ms`

## What the script measures (OS‑relevant)

- **Wall‑clock time** per epoch and total: simple `time.time()` around the training loop.

- **Memory, three ways**
  - **Absolute peak (`mem_peak_abs_MB`)** — what the kernel reports as the **maximum resident set size so far** for the process, via `resource.getrusage(...).ru_maxrss`. Units differ: KB on Linux, bytes on macOS (the code normalizes to MB).
  - **Delta (`mem_delta_MB`)** — how much that peak increased **over a baseline** captured just before fine‑tuning. This isolates **training‑only growth** (primarily gradients and Adam optimizer state).
  - **Now (`mem_now_MB`)** — instantaneous RSS via `psutil.Process(...).memory_info().rss`.

- **Filesystem latency** for model checkpoints
  - Writes the model to **ext4** (project directory) and **tmpfs** (`/dev/shm`) using an **atomic rename** (`os.replace`) to avoid partial files after a crash. Reports per‑write milliseconds.

## Why these defaults?

- `--img 128` keeps tensors small for a CPU‑only VM while preserving clear signal.
- `--workers 0` avoids thread oversubscription inside a VM; with a fast host SSD you can try `--workers 4`.
- `MALLOC_ARENA_MAX=1` prevents glibc from creating many memory arenas per thread, which reduces the process’s resident memory on Linux.

## Mapping to Operating Systems concepts (for the report)

- **Process/threads & scheduler** — the `--workers` knob adjusts DataLoader worker threads; you can compare wall time and CPU utilization with `top`, `pidstat`, or `perf` to discuss Linux’s **CFS** behavior.
- **Virtual memory management** — `ru_maxrss` vs `psutil` RSS contrasts **peak vs instantaneous** residency; you can reason about **paging risk** and why optimizer state inflates memory.
- **Filesystem & I/O** — tmpfs vs ext4 timings show **memory‑backed** write vs **durable** write; `os.replace` demonstrates **atomic rename**, a common journal‑friendly pattern.
- **Security/privacy** — all tuning happens **locally** (no sockets opened), reducing attack surface and avoiding data egress.

## Reproducing the “final” numbers from class
Use the `run.sh` script provided. On our test VM this configuration typically yields:
- Accuracy rises into the **0.85–0.90** range after 4 epochs
- Total fine‑tune time **< 60 s**
- Absolute **peak RSS ~520–560 MB** (over the 400 MB target on this stack)
- Training‑only **delta ~140–170 MB**
- tmpfs writes usually faster than ext4 after first‑epoch warm‑up

Your exact numbers will depend on CPU, Python, glibc, and VM settings.

## Troubleshooting

- **Module not found** — ensure the venv is activated and `pip install -r requirements.txt` succeeded.
- **Peak RSS higher than expected** — prefix the run with `MALLOC_ARENA_MAX=1` (already in `run.sh`) and keep `--workers 0–1` inside VMs.
- **Plots missing** — `matplotlib` or `pandas` may be missing; install from `requirements.txt`.

---

© 2025 Brandon Stetler.
