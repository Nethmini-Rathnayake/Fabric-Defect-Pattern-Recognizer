"""
Build a demo MP4 from val images: annotated with class, confidence, FPS.
Saves: reports/demo.mp4
"""

import sys, random, time
from pathlib import Path

import cv2
import torch
import numpy as np
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.detection.model import FabricDefectModel, DEFECT_CLASSES

MODEL_PATH  = "models/efficientnet/best.pt"
VAL_DIR     = Path("data/images/val")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)
IMAGE_SIZE  = 64
OUT_W, OUT_H = 960, 540
FPS_OUT      = 15
FRAMES_PER_IMAGE = 8   # each image shown for N frames so the video is watchable

COLORS = {
    "normal":           (34,  139,  34),
    "stain":            (255, 140,   0),
    "tear":             (200,  30,  30),
    "weave_distortion": (30,  100, 200),
}

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

# ── collect images: 20 per class, shuffled so classes interleave ──────────────
all_paths = []
for cls_dir in sorted(VAL_DIR.iterdir()):
    if cls_dir.is_dir():
        paths = sorted(cls_dir.glob("*.[jpJP][pnPN][gG]*"))
        all_paths.append(random.sample(paths, min(20, len(paths))))

# interleave classes
interleaved = []
for row in zip(*all_paths):
    interleaved.extend(row)
interleaved += [p for col in all_paths for p in col[len(interleaved)//len(all_paths):]]

print(f"Rendering {len(interleaved)} images × {FRAMES_PER_IMAGE} frames = "
      f"{len(interleaved)*FRAMES_PER_IMAGE} frames  @ {FPS_OUT} fps")

# ── video writer ──────────────────────────────────────────────────────────────
out_path = REPORTS_DIR / "demo.mp4"
fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
writer   = cv2.VideoWriter(str(out_path), fourcc, FPS_OUT, (OUT_W, OUT_H))

@torch.no_grad()
def predict(frame_rgb: np.ndarray):
    t0     = time.perf_counter()
    tensor = PREPROCESS(frame_rgb).unsqueeze(0)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1)[0]
    conf, idx = probs.max(0)
    ms = (time.perf_counter() - t0) * 1000
    return DEFECT_CLASSES[idx.item()], conf.item(), ms

def build_frame(img_bgr: np.ndarray, true_cls: str, pred_cls: str,
                conf: float, inf_ms: float, frame_fps: float) -> np.ndarray:
    canvas = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    # large image panel (left)
    disp = cv2.resize(img_bgr, (OUT_H, OUT_H))
    canvas[:, :OUT_H] = disp

    # prediction color + border
    color  = COLORS.get(pred_cls, (180, 180, 180))
    correct = (pred_cls == true_cls)
    border_color = (0, 200, 0) if correct else (0, 0, 220)
    cv2.rectangle(canvas, (0, 0), (OUT_H - 1, OUT_H - 1), border_color, 6)

    # right panel text
    rx = OUT_H + 24
    def put(text, y, scale=0.75, clr=(230, 230, 230), bold=False):
        thickness = 2 if bold else 1
        cv2.putText(canvas, text, (rx, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, clr, thickness, cv2.LINE_AA)

    put("FABRIC DEFECT DETECTOR", 50, 0.85, (255, 255, 255), bold=True)
    cv2.line(canvas, (rx, 60), (OUT_W - 20, 60), (80, 80, 80), 1)

    put("Predicted:", 110, 0.65, (160, 160, 160))
    put(pred_cls.replace("_", " ").upper(), 148, 1.0, color, bold=True)
    put(f"Confidence:  {conf:.1%}", 195, 0.72)

    put("True label:", 255, 0.65, (160, 160, 160))
    put(true_cls.replace("_", " ").upper(), 293, 0.85,
        (0, 200, 0) if correct else (0, 80, 220), bold=True)

    verdict = "CORRECT" if correct else "WRONG"
    v_color = (0, 200, 0) if correct else (0, 60, 200)
    put(verdict, 340, 0.9, v_color, bold=True)

    cv2.line(canvas, (rx, 365), (OUT_W - 20, 365), (80, 80, 80), 1)

    put("Hardware:  Apple M1 CPU", 400, 0.6)
    put(f"Input res: 640×480 → {IMAGE_SIZE}×{IMAGE_SIZE}", 428, 0.6)
    put(f"Inf latency: {inf_ms:.1f} ms", 456, 0.6)
    put(f"FPS: {frame_fps:.1f}", 484, 0.72, (255, 220, 80), bold=True)

    return canvas

# ── render ────────────────────────────────────────────────────────────────────
fps_window, t0 = [], time.perf_counter()

for n, path in enumerate(interleaved):
    true_cls = path.parent.name
    img_bgr  = cv2.imread(str(path))
    img_bgr  = cv2.resize(img_bgr, (640, 480))
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    pred_cls, conf, inf_ms = predict(img_rgb)

    elapsed = time.perf_counter() - t0
    fps_window.append(inf_ms)
    if len(fps_window) > 30:
        fps_window.pop(0)
    live_fps = 1000.0 / (sum(fps_window) / len(fps_window))

    frame = build_frame(img_bgr, true_cls, pred_cls, conf, inf_ms, live_fps)

    for _ in range(FRAMES_PER_IMAGE):
        writer.write(frame)

    if (n + 1) % 20 == 0:
        print(f"  {n+1}/{len(interleaved)}  live FPS {live_fps:.1f}")

writer.release()
total_s = len(interleaved) * FRAMES_PER_IMAGE / FPS_OUT
print(f"\nSaved {out_path}  ({total_s:.0f}s @ {FPS_OUT} fps)")
