"""
End-to-end RAG demo: model prediction → retrieval → explanation.
Run from project root: PYTHONPATH=. python src/explainability/demo_rag.py
"""

import sys, time
from pathlib import Path

import torch
from torchvision import transforms
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.detection.model import FabricDefectModel, DEFECT_CLASSES
from src.explainability.rag_pipeline import FabricDefectExplainer

MODEL_PATH = "models/efficientnet/best.pt"
VAL_DIR    = Path("data/images/val")
IMAGE_SIZE = 64

PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DIVIDER = "─" * 70

def load_model():
    state = torch.load(MODEL_PATH, map_location="cpu")
    model = FabricDefectModel(num_classes=len(DEFECT_CLASSES), pretrained=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model

@torch.no_grad()
def run_model(model, img_path: str):
    frame = cv2.imread(img_path)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t     = PREPROCESS(rgb).unsqueeze(0)
    probs = torch.softmax(model(t), dim=1)[0]
    conf, idx = probs.max(0)
    return DEFECT_CLASSES[idx.item()], conf.item(), probs.tolist()

def pick_sample(cls: str) -> str:
    """Pick one example image from the val set for the given class."""
    cls_dir = VAL_DIR / cls
    imgs    = sorted(cls_dir.glob("*.[jpJP][pnPN][gG]*"))
    if not imgs:
        raise FileNotFoundError(f"No images found in {cls_dir}")
    return str(imgs[0])


def main():
    print(f"\n{'='*70}")
    print("  FABRIC DEFECT RAG PIPELINE — END-TO-END DEMO")
    print(f"{'='*70}\n")

    print("Loading model …")
    model = load_model()
    print("Loading RAG explainer …")
    explainer = FabricDefectExplainer()

    # Run one example per defect class
    for cls in DEFECT_CLASSES:
        img_path = pick_sample(cls)

        t0 = time.perf_counter()
        pred_class, confidence, all_probs = run_model(model, img_path)
        model_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        result = explainer.explain(pred_class, confidence)
        rag_ms = (time.perf_counter() - t1) * 1000

        print(f"\n{DIVIDER}")
        print(f"  TRUE CLASS : {cls.upper()}")
        print(f"  IMAGE      : {Path(img_path).name}")
        print(DIVIDER)

        # Model output
        print(f"\n  MODEL PREDICTION")
        print(f"  {'Class':<22} {'Confidence':>10}")
        print(f"  {'-'*34}")
        for i, (name, prob) in enumerate(zip(DEFECT_CLASSES, all_probs)):
            marker = " ◀" if i == DEFECT_CLASSES.index(pred_class) else ""
            print(f"  {name:<22} {prob:>9.1%}{marker}")
        print(f"\n  Model inference: {model_ms:.1f} ms")

        # Retrieved chunks
        print(f"\n  RETRIEVED KNOWLEDGE CHUNKS  (top {len(result['retrieved_chunks'])})")
        print(f"  {'-'*66}")
        for j, chunk in enumerate(result["retrieved_chunks"], 1):
            preview = chunk["text"].replace("\n", " ")[:120]
            print(f"  [{j}] source={chunk['source']:<20} score={chunk['score']:.3f}")
            print(f"       \"{preview}...\"")

        # Explanation
        print(f"\n  EXPLANATION  ({result['source']})")
        print(f"  {'-'*66}")
        for line in result["explanation"].split("\n"):
            print(f"  {line}")
        print(f"\n  RAG latency: {rag_ms:.0f} ms")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
