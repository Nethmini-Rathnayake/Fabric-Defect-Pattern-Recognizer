"""
Measure end-to-end latency breakdown and FPS on the validation set.
Outputs: reports/benchmark.txt  +  reports/benchmark_latency.png
"""

import sys, time, platform
from pathlib import Path

import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.detection.model import FabricDefectModel, DEFECT_CLASSES

MODEL_PATH  = "models/efficientnet/best.pt"
VAL_DIR     = Path("data/images/val")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)
IMAGE_SIZE  = 64
N_WARMUP    = 20

PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── load model ────────────────────────────────────────────────────────────────
state = torch.load(MODEL_PATH, map_location="cpu")
model = FabricDefectModel(num_classes=len(DEFECT_CLASSES), pretrained=False)
model.load_state_dict(state["model_state_dict"])
model.eval()

# ── collect image paths ───────────────────────────────────────────────────────
img_paths = []
for cls_dir in sorted(VAL_DIR.iterdir()):
    if cls_dir.is_dir():
        img_paths.extend(sorted(cls_dir.glob("*.[jpJP][pnPN][gG]*"))[:50])

print(f"Benchmarking on {len(img_paths)} images  (warm-up: {N_WARMUP})")

t_load_ms, t_pre_ms, t_inf_ms, t_post_ms, t_overlay_ms = [], [], [], [], []

for i, path in enumerate(img_paths):

    # 1. Load
    t0 = time.perf_counter()
    frame = cv2.imread(str(path))
    frame = cv2.resize(frame, (640, 480))
    t1 = time.perf_counter()

    # 2. Preprocess
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = PREPROCESS(rgb).unsqueeze(0)
    t2 = time.perf_counter()

    # 3. Inference
    with torch.no_grad():
        logits = model(tensor)
    t3 = time.perf_counter()

    # 4. Postprocess (softmax + label)
    probs = torch.softmax(logits, dim=1)[0]
    conf, idx = probs.max(0)
    label = DEFECT_CLASSES[idx.item()]
    t4 = time.perf_counter()

    # 5. Overlay
    color = (34, 139, 34)
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (8, 8), (w - 8, h - 8), color, 3)
    cv2.rectangle(out, (8, 8), (w - 8, 48), color, -1)
    cv2.putText(out, f"{label}  {conf:.0%}", (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    t5 = time.perf_counter()

    if i >= N_WARMUP:
        t_load_ms.append((t1 - t0) * 1000)
        t_pre_ms.append((t2 - t1) * 1000)
        t_inf_ms.append((t3 - t2) * 1000)
        t_post_ms.append((t4 - t3) * 1000)
        t_overlay_ms.append((t5 - t4) * 1000)

# ── compute stats ─────────────────────────────────────────────────────────────
def stats(arr):
    a = np.array(arr)
    return dict(mean=a.mean(), std=a.std(), p50=np.percentile(a, 50),
                p95=np.percentile(a, 95), p99=np.percentile(a, 99))

stages = {
    "Load + resize (640×480)": t_load_ms,
    "Preprocess (norm+tensor)": t_pre_ms,
    "Inference (MobileNetV3-S)": t_inf_ms,
    "Postprocess (softmax)":    t_post_ms,
    "Overlay (cv2 draw)":       t_overlay_ms,
}

total = [sum(x) for x in zip(t_load_ms, t_pre_ms, t_inf_ms, t_post_ms, t_overlay_ms)]
end2end = stats(total)
fps = 1000.0 / end2end["mean"]

# ── hardware info ─────────────────────────────────────────────────────────────
import subprocess
try:
    cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
except Exception:
    cpu = platform.processor()
try:
    ram_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip())
    ram = f"{ram_bytes / 1e9:.0f} GB"
except Exception:
    ram = "unknown"

hw_info = (
    f"Hardware : {cpu}\n"
    f"RAM      : {ram}\n"
    f"OS       : {platform.system()} {platform.release()}\n"
    f"PyTorch  : {torch.__version__}\n"
    f"Input res: 640×480 → resized to {IMAGE_SIZE}×{IMAGE_SIZE}\n"
    f"Device   : CPU (no GPU / MPS)\n"
    f"N samples: {len(total)}"
)

# ── text report ───────────────────────────────────────────────────────────────
lines = [
    "=" * 60,
    "  FABRIC DEFECT PIPELINE — LATENCY BENCHMARK",
    "=" * 60,
    "",
    hw_info,
    "",
    f"{'Stage':<30} {'Mean':>7} {'Std':>7} {'p50':>7} {'p95':>7} {'p99':>7}",
    "-" * 60,
]
for name, arr in stages.items():
    s = stats(arr)
    lines.append(f"{name:<30} {s['mean']:>6.2f}ms {s['std']:>6.2f}ms "
                 f"{s['p50']:>6.2f}ms {s['p95']:>6.2f}ms {s['p99']:>6.2f}ms")
lines += [
    "-" * 60,
    f"{'End-to-end':<30} {end2end['mean']:>6.2f}ms {end2end['std']:>6.2f}ms "
    f"{end2end['p50']:>6.2f}ms {end2end['p95']:>6.2f}ms {end2end['p99']:>6.2f}ms",
    "",
    f"  Effective FPS (mean latency): {fps:.1f}",
    f"  Target: >20 FPS  →  {'PASS ✓' if fps >= 20 else 'FAIL ✗'}",
    "=" * 60,
]
report_text = "\n".join(lines)
print(report_text)

txt_path = REPORTS_DIR / "benchmark.txt"
txt_path.write_text(report_text)
print(f"\nSaved → {txt_path}")

# ── latency chart ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Pipeline Latency Breakdown — Apple M1 CPU  |  {fps:.1f} FPS",
             fontsize=13, fontweight="bold")

# left: stacked bar of mean stage times
stage_means = [stats(arr)["mean"] for arr in stages.values()]
stage_names = [n.split("(")[0].strip() for n in stages.keys()]
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#937860"]
bottoms, prev = np.zeros(1), 0.0
ax = axes[0]
for i, (name, mean, color) in enumerate(zip(stage_names, stage_means, colors)):
    ax.bar(["End-to-end"], [mean], bottom=[prev], color=color, label=f"{name} ({mean:.1f} ms)")
    prev += mean
ax.axhline(1000 / 20, color="red", linestyle="--", linewidth=1.2, label="20 FPS threshold (50 ms)")
ax.set_ylabel("Latency (ms)")
ax.set_title("Mean Latency per Stage")
ax.legend(fontsize=8, loc="upper right")
ax.set_ylim(0, max(prev * 1.4, 60))

# right: inference latency distribution
ax2 = axes[1]
ax2.hist(t_inf_ms, bins=30, color="#C44E52", edgecolor="white", alpha=0.85)
ax2.axvline(np.mean(t_inf_ms), color="black",  linestyle="--", label=f"Mean {np.mean(t_inf_ms):.1f} ms")
ax2.axvline(np.percentile(t_inf_ms, 95), color="orange", linestyle="--",
            label=f"p95  {np.percentile(t_inf_ms, 95):.1f} ms")
ax2.set_xlabel("Inference latency (ms)")
ax2.set_ylabel("Count")
ax2.set_title("Inference Latency Distribution")
ax2.legend(fontsize=9)

plt.tight_layout()
chart_path = REPORTS_DIR / "benchmark_latency.png"
fig.savefig(chart_path, dpi=150, bbox_inches="tight")
print(f"Saved → {chart_path}")
