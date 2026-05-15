# Fabric Defect Pattern Recognizer

Real-time fabric defect detection for Sri Lanka's apparel industry — combining a fine-tuned MobileNetV3 vision model with a RAG pipeline that retrieves from a textile engineering knowledge base to explain root causes to factory floor operators.

---

## Results

| Metric | Value | Notes |
|---|---|---|
| Validation accuracy | **93.5%** | 773-image held-out val set |
| Macro F1 | **0.924** | Across 4 defect classes |
| Worst-class F1 | 0.883 (tear) | Fewest training samples |
| Inference latency | **70 ms / image** | Apple M1 CPU, single image |
| Throughput | 11.3 FPS | Apple M1 CPU — see benchmark note |
| RAG retrieval | ~16 ms | Qdrant in-memory, top-3 chunks |

> **Honest benchmark note:** 11.3 FPS is measured single-image on Apple M1 CPU (no GPU, no MPS). The original >20 FPS target is achievable on a GPU (T4 or better). The benchmark script logs exact per-stage latency — see [`reports/benchmark.txt`](reports/benchmark.txt).

---

## What it does

1. **Classify** — a conveyor-belt frame passes through MobileNetV3-Small, which outputs one of four classes with a confidence score.
2. **Retrieve** — the predicted class queries a Qdrant vector store (sentence-transformers embeddings) to find the most relevant chunks from a textile engineering knowledge base.
3. **Explain** — retrieved context is passed to Claude claude-haiku-4-5-20251001 (or a structured template fallback if no API key) to produce a 3-point explanation: what the defect indicates, most likely root cause, and recommended immediate action.
4. **Report** — annotated frames are written to a demo video; a full evaluation report (confusion matrix, per-class F1, example predictions) is saved to `reports/`.

---

## Defect Classes

| Class | Training images | Val images | F1 |
|---|---|---|---|
| normal | 1 402 | 332 | 0.963 |
| stain | 628 | 168 | 0.940 |
| weave_distortion | 620 | 159 | 0.910 |
| tear | 448 | 114 | 0.883 |

Class imbalance (normal is 3× tear) is real and visible in the per-class F1 gap. No oversampling was applied; the model handles it via label smoothing (0.1) and pretrained ImageNet features.

---

## RAG Knowledge Base

Five documents in [`data/knowledge_base/`](data/knowledge_base/), chunked into 36 indexed passages:

| File | Content |
|---|---|
| `stain.txt` | Visual characteristics, lubrication causes, dye process causes, chemical causes, corrective actions, ISO 105-X12 references |
| `tear.txt` | Knitting needle wear, loom reed damage, yarn tenacity thresholds, foreign object causes, AQL critical defect criteria |
| `weave_distortion.txt` | Warp tension imbalance, beam winding errors, stenter overfeed, spirality in knit, ISO 13015 measurement method |
| `normal.txt` | Acceptance criteria (ΔE, GSM, pilling grade), AQL pass conditions |
| `process_control.txt` | 4-point inspection system, SPC chart guidance, NCR escalation thresholds, traceability requirements |

Example retrieval for `weave_distortion` (score 0.622):
> *"Warp beam tension that varies across the width is the primary cause of bowing and skewing. A tension differential of more than 5% between the left, centre, and right warp zones causes selvedge-to-centre speed differences during beat-up."*

---

## System Architecture

```
Camera / image file
        │
        ▼
  Preprocess (resize 64×64, ImageNet normalise)
        │
        ▼
  MobileNetV3-Small  (fine-tuned, 4 classes)
  models/efficientnet/best.pt
        │
        ├─── confidence score + class
        │
        ▼
  Qdrant in-memory vector store
  (sentence-transformers/all-MiniLM-L6-v2)
  data/knowledge_base/*.txt  →  36 chunks
        │
        ▼
  Claude claude-haiku-4-5-20251001  (or template fallback)
        │
        ▼
  3-point explanation:
    1. What the defect indicates
    2. Most likely root cause
    3. Recommended immediate action
        │
        ▼
  Streamlit dashboard  /  demo.mp4  /  reports/
```

---

## Project Structure

```
├── configs/
│   └── train_config.yaml          # Model, data, training hyperparameters
├── data/
│   ├── images/train|val/          # 4 classes, ~3 900 train / 773 val
│   └── knowledge_base/            # Textile engineering documents (RAG source)
├── models/
│   └── efficientnet/best.pt       # Trained checkpoint (epoch 18, val_acc 0.9353)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── reports/
│   ├── model_report.png           # Dataset stats + confusion matrix + F1 + example predictions
│   ├── benchmark.txt              # Per-stage latency table (hardware-stamped)
│   ├── benchmark_latency.png      # Stacked bar + inference distribution chart
│   └── demo.mp4                   # Annotated video: class, confidence, true label, FPS
├── src/
│   ├── detection/
│   │   ├── model.py               # MobileNetV3-Small wrapper
│   │   ├── dataset.py             # FabricDefectDataset + albumentations transforms
│   │   ├── train.py               # Training loop, MLflow logging, early stopping
│   │   └── evaluate.py            # Full eval report generator
│   ├── explainability/
│   │   ├── rag_pipeline.py        # Qdrant + sentence-transformers + Claude/template
│   │   └── demo_rag.py            # End-to-end CLI demo
│   └── pipeline/
│       ├── run_inference.py       # OpenCV real-time pipeline
│       ├── benchmark.py           # Latency breakdown (load/preprocess/infer/overlay)
│       └── demo_video.py          # Renders annotated MP4 from val images
├── dashboard/
│   └── app.py                     # Streamlit live feed + defect distribution chart
├── docker/
│   └── Dockerfile
└── requirements.txt
```

---

## Getting Started

```bash
git clone https://github.com/Nethmini-Rathnayake/Fabric-Defect-Pattern-Recognizer.git
cd Fabric-Defect-Pattern-Recognizer
pip install -r requirements.txt
cp .env.example .env           # add ANTHROPIC_API_KEY for LLM explanations
```

### Train

```bash
PYTHONPATH=. python src/detection/train.py --config configs/train_config.yaml
```

### Evaluate

```bash
PYTHONPATH=. python src/detection/evaluate.py
# → reports/model_report.png
```

### RAG demo

```bash
PYTHONPATH=. python src/explainability/demo_rag.py
# works without API key (template fallback); set ANTHROPIC_API_KEY for Claude
```

### Latency benchmark

```bash
PYTHONPATH=. python src/pipeline/benchmark.py
# → reports/benchmark.txt  +  reports/benchmark_latency.png
```

### Demo video

```bash
PYTHONPATH=. python src/pipeline/demo_video.py
# → reports/demo.mp4
```

### Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Vision model | MobileNetV3-Small (PyTorch / torchvision) |
| Training | AdamW + cosine LR + label smoothing, MLflow tracking |
| Augmentation | albumentations (blur, brightness, CLAHE, hue shift) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | Qdrant in-memory (no Docker required for dev) |
| LLM | Claude claude-haiku-4-5-20251001 (Anthropic API) |
| Real-time pipeline | OpenCV |
| Dashboard | Streamlit + Plotly |
| Deployment | Docker |

---

## Roadmap

- [x] Dataset sourcing and preprocessing (Kaggle + Roboflow, 4 classes)
- [x] MobileNetV3-Small fine-tuning — 93.5% val accuracy
- [x] Full evaluation report (confusion matrix, per-class F1, example predictions)
- [x] Latency benchmark with per-stage breakdown
- [x] Annotated demo video
- [x] RAG pipeline — Qdrant + sentence-transformers + Claude API
- [x] Textile engineering knowledge base (5 documents, 36 chunks)
- [x] Streamlit dashboard
- [ ] MPS / GPU support for >20 FPS on Apple Silicon or server GPU
- [ ] Sinhala PDF report generator
- [ ] Docker deployment package
- [ ] Factory pilot test

---

## License

MIT — see [LICENSE](LICENSE).
