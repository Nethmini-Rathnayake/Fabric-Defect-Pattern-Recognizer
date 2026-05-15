"""Generate a full evaluation report: dataset stats, confusion matrix, F1, example predictions."""

import sys, time
from pathlib import Path

import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.detection.model import FabricDefectModel, DEFECT_CLASSES
from src.detection.dataset import FabricDefectDataset, get_val_transforms

MODEL_PATH   = "models/efficientnet/best.pt"
TRAIN_DIR    = "data/images/train"
VAL_DIR      = "data/images/val"
IMAGE_SIZE   = 64
BATCH_SIZE   = 32
REPORTS_DIR  = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def load_display_image(path: str, size: int = 96) -> np.ndarray:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img

def count_per_class(root: str) -> dict:
    root = Path(root)
    return {d.name: len(list(d.glob("*.[jpJP][pnPN][gG]*"))) for d in sorted(root.iterdir()) if d.is_dir()}

# ── load model ────────────────────────────────────────────────────────────────

state = torch.load(MODEL_PATH, map_location="cpu")
model = FabricDefectModel(num_classes=len(DEFECT_CLASSES), pretrained=False)
model.load_state_dict(state["model_state_dict"])
model.eval()
print(f"Checkpoint: epoch {state['epoch']}, val_acc {state['val_acc']:.4f}")

# ── inference — keep track of paths too ──────────────────────────────────────

val_ds = FabricDefectDataset(VAL_DIR, get_val_transforms(IMAGE_SIZE))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
class_names = val_ds.classes

all_preds, all_labels, all_paths = [], [], []
t0 = time.time()
with torch.no_grad():
    idx = 0
    for images, labels in val_loader:
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)
        preds  = logits.argmax(1).cpu().numpy()
        for i, (p, l) in enumerate(zip(preds, labels.numpy())):
            all_preds.append(p)
            all_labels.append(l)
            all_paths.append(val_ds.samples[idx + i][0])
        idx += len(labels)
elapsed = time.time() - t0
fps = len(val_ds) / elapsed
print(f"Inference: {fps:.1f} FPS")

# ── metrics ───────────────────────────────────────────────────────────────────

report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
cm     = confusion_matrix(all_labels, all_preds)

print("\n--- Classification Report ---")
print(pd.DataFrame(report).T.round(3).to_string())
print(f"\nOverall Accuracy: {report['accuracy']:.4f}")

# ── collect example images ────────────────────────────────────────────────────
# For each class: up to 3 correct predictions, up to 3 wrong predictions

correct_imgs = {c: [] for c in class_names}
wrong_imgs   = {c: [] for c in class_names}   # keyed by true class

for path, true, pred in zip(all_paths, all_labels, all_preds):
    true_name = class_names[true]
    pred_name = class_names[pred]
    if true == pred:
        if len(correct_imgs[true_name]) < 3:
            correct_imgs[true_name].append(path)
    else:
        if len(wrong_imgs[true_name]) < 3:
            wrong_imgs[true_name].append((path, pred_name))

# ── build figure ─────────────────────────────────────────────────────────────

plt.rcParams.update({"font.size": 9})
fig = plt.figure(figsize=(20, 26), facecolor="#f8f9fa")
fig.suptitle(
    "Fabric Defect Classifier — Evaluation Report\n"
    f"MobileNetV3-Small · 4 classes · Val accuracy {report['accuracy']:.1%} · {fps:.0f} FPS",
    fontsize=15, fontweight="bold", y=0.995
)

outer = gridspec.GridSpec(4, 1, figure=fig, hspace=0.45,
                          top=0.97, bottom=0.02, left=0.06, right=0.97)

# ── Row 0: dataset overview ───────────────────────────────────────────────────

ax_data = fig.add_subplot(outer[0])
train_counts = count_per_class(TRAIN_DIR)
val_counts   = count_per_class(VAL_DIR)
x = np.arange(len(class_names))
w = 0.35
bars1 = ax_data.bar(x - w/2, [train_counts.get(c, 0) for c in class_names], w,
                    label="Train", color="#4C72B0")
bars2 = ax_data.bar(x + w/2, [val_counts.get(c, 0)   for c in class_names], w,
                    label="Val",   color="#DD8452")
for bar in list(bars1) + list(bars2):
    ax_data.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)
ax_data.set_xticks(x)
ax_data.set_xticklabels([c.replace("_", "\n") for c in class_names])
ax_data.set_ylabel("Images"); ax_data.legend()
ax_data.set_title(
    f"Dataset Distribution  |  Train total: {sum(train_counts.values())}  "
    f"Val total: {sum(val_counts.values())}  "
    f"|  Most common class is {max(train_counts, key=train_counts.get)} "
    f"({max(train_counts.values())/sum(train_counts.values()):.0%} of train)",
    fontsize=9
)

# ── Row 1: confusion matrix + F1/precision/recall ─────────────────────────────

row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.35)

# confusion matrix
ax_cm = fig.add_subplot(row1[0])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[c.replace("_", "\n") for c in class_names],
            yticklabels=[c.replace("_", "\n") for c in class_names],
            ax=ax_cm, linewidths=0.5)
ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("True")
ax_cm.set_title("Confusion Matrix")

# per-class metrics
ax_met = fig.add_subplot(row1[1])
metrics = {
    "Precision": [report[c]["precision"] for c in class_names],
    "Recall":    [report[c]["recall"]    for c in class_names],
    "F1":        [report[c]["f1-score"]  for c in class_names],
}
df_met = pd.DataFrame(metrics, index=[c.replace("_", "\n") for c in class_names])
x2 = np.arange(len(class_names))
w2 = 0.25
for i, (label, vals) in enumerate(metrics.items()):
    ax_met.bar(x2 + i*w2 - w2, vals, w2, label=label,
               color=["#4C72B0", "#55A868", "#C44E52"][i])
ax_met.axhline(0.92, color="black", linestyle="--", linewidth=0.8, label="92% target")
ax_met.set_xticks(x2); ax_met.set_xticklabels([c.replace("_", "\n") for c in class_names])
ax_met.set_ylim(0, 1.05); ax_met.set_ylabel("Score")
ax_met.set_title("Per-Class Precision / Recall / F1"); ax_met.legend(fontsize=8)
for i, (label, vals) in enumerate(metrics.items()):
    for j, v in enumerate(vals):
        ax_met.text(x2[j] + i*w2 - w2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7)

# ── Row 2: correct predictions ────────────────────────────────────────────────

row2 = gridspec.GridSpecFromSubplotSpec(
    1, len(class_names), subplot_spec=outer[2], wspace=0.08)
inner_rows2 = [gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=row2[i], hspace=0.05)
               for i in range(len(class_names))]

for col, cls in enumerate(class_names):
    for row_idx in range(3):
        ax = fig.add_subplot(inner_rows2[col][row_idx])
        ax.axis("off")
        imgs = correct_imgs[cls]
        if row_idx < len(imgs):
            ax.imshow(load_display_image(imgs[row_idx]))
            if row_idx == 0:
                ax.set_title(f"✓ {cls.replace('_', ' ')}", fontsize=8,
                             color="green", fontweight="bold", pad=3)
        else:
            ax.set_facecolor("#eeeeee")

# section label
fig.text(0.5, outer[2].get_position(fig).y1 + 0.005,
         "Correct Predictions", ha="center", fontsize=10, fontweight="bold", color="green")

# ── Row 3: wrong predictions ──────────────────────────────────────────────────

row3 = gridspec.GridSpecFromSubplotSpec(
    1, len(class_names), subplot_spec=outer[3], wspace=0.08)
inner_rows3 = [gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=row3[i], hspace=0.05)
               for i in range(len(class_names))]

for col, cls in enumerate(class_names):
    for row_idx in range(3):
        ax = fig.add_subplot(inner_rows3[col][row_idx])
        ax.axis("off")
        errs = wrong_imgs[cls]
        if row_idx < len(errs):
            path, pred_name = errs[row_idx]
            ax.imshow(load_display_image(path))
            if row_idx == 0:
                ax.set_title(f"✗ true: {cls.replace('_',' ')}\n→ pred: {pred_name.replace('_',' ')}",
                             fontsize=7, color="red", fontweight="bold", pad=3)
        else:
            ax.set_facecolor("#eeeeee")

fig.text(0.5, outer[3].get_position(fig).y1 + 0.005,
         "Wrong Predictions  (true → predicted)", ha="center",
         fontsize=10, fontweight="bold", color="red")

# ── save ──────────────────────────────────────────────────────────────────────

out = REPORTS_DIR / "model_report.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved full report → {out}")
