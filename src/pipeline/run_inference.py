"""Real-time fabric defect detection pipeline using OpenCV + MobileNetV3-Small."""

import argparse
import time
from pathlib import Path

import cv2
import torch
import numpy as np
from torchvision import transforms
from dotenv import load_dotenv
import os

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.detection.model import FabricDefectModel, DEFECT_CLASSES

load_dotenv()

CONF_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.70))
IMAGE_SIZE = 64

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


def load_pipeline_model(model_path: str):
    state = torch.load(model_path, map_location="cpu")
    model = FabricDefectModel(num_classes=len(DEFECT_CLASSES), pretrained=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_frame(model, frame: np.ndarray):
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = PREPROCESS(rgb).unsqueeze(0)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1)[0]
    conf, idx = probs.max(0)
    return DEFECT_CLASSES[idx.item()], conf.item()


def annotate_frame(frame: np.ndarray, label: str, confidence: float, fps: float) -> np.ndarray:
    out   = frame.copy()
    color = COLORS.get(label, (0, 255, 0))
    h, w  = out.shape[:2]
    cv2.rectangle(out, (8, 8), (w - 8, h - 8), color, 3)
    cv2.rectangle(out, (8, 8), (w - 8, 48), color, -1)
    disp = f"{label.replace('_', ' ').title()}  {confidence:.0%}"
    cv2.putText(out, disp, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(out, f"FPS {fps:.1f}", (16, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    return out


def run(source, model_path: str):
    model = load_pipeline_model(model_path)
    cap   = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    fps_counter, t_start = 0, time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        label, conf = predict_frame(model, frame)
        fps = fps_counter / max(time.time() - t_start, 1e-6)
        if conf >= CONF_THRESHOLD:
            frame = annotate_frame(frame, label, conf, fps)
        fps_counter += 1
        cv2.imshow("Fabric Defect Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Average FPS: {fps_counter / (time.time() - t_start):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=os.getenv("VIDEO_SOURCE", "0"))
    parser.add_argument("--model",  default=os.getenv("MODEL_PATH", "models/efficientnet/best.pt"))
    args = parser.parse_args()
    run(args.source, args.model)
