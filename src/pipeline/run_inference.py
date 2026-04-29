"""Real-time fabric defect detection pipeline using OpenCV + EfficientNet."""

import argparse
import time
from pathlib import Path

import cv2
import torch
import numpy as np
from torchvision import transforms
from dotenv import load_dotenv
import os

from src.detection.model import load_model, DEFECT_CLASSES

load_dotenv()

CONF_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.90))

PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

COLORS = {
    "oil_stain": (0, 0, 200),
    "dye_stain": (200, 0, 200),
    "hole_snag": (0, 0, 0),
    "drop_stitch": (0, 100, 255),
    "weave_distortion": (255, 165, 0),
    "slub_nep": (0, 200, 200),
    "shade_variation": (150, 0, 150),
    "shrinkage": (100, 100, 0),
}


@torch.no_grad()
def predict_frame(model, frame: np.ndarray, device: str):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = PREPROCESS(rgb).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    conf, idx = probs.max(0)
    return DEFECT_CLASSES[idx.item()], conf.item()


def annotate_frame(frame, defect_class: str, confidence: float) -> np.ndarray:
    color = COLORS.get(defect_class, (0, 255, 0))
    label = f"{defect_class.replace('_', ' ').title()} {confidence:.1%}"
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (10, 10), (w - 10, h - 10), color, 3)
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return frame


def run(source, model_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device)

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    fps_counter, t_start = 0, time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        defect_class, confidence = predict_frame(model, frame, device)

        if confidence >= CONF_THRESHOLD:
            frame = annotate_frame(frame, defect_class, confidence)

        fps_counter += 1
        elapsed = time.time() - t_start
        fps = fps_counter / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Fabric Defect Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Average FPS: {fps_counter / (time.time() - t_start):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=os.getenv("VIDEO_SOURCE", "0"), help="Video source: 0 for webcam or path to video file")
    parser.add_argument("--model", default=os.getenv("MODEL_PATH", "models/efficientnet/best.pt"))
    args = parser.parse_args()
    run(args.source, args.model)
