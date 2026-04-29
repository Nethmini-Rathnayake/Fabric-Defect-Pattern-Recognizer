# 🧵 Fabric Defect Pattern Recognizer

> Real-time AI-powered fabric defect detection for Sri Lanka's apparel industry, combining EfficientNet computer vision with LLM-generated root-cause explanations.

---

## 📌 Overview

The **Fabric Defect Pattern Recognizer** is an end-to-end machine learning system that detects textile defects on conveyor belts in real time. It combines a CNN-based vision model (EfficientNet) with a Retrieval-Augmented Generation (RAG) pipeline to not only classify defects but explain their likely causes — empowering factory operators to act fast and reduce waste.

Designed specifically for Sri Lanka's Western Province apparel exporters (e.g., Brandix, MAS Holdings), the system targets the unique challenges of tropical humidity, knitted fabric dominance, and rare defect imbalances that generic models fail to handle.

---

## 🎯 Key Features

- **Real-time detection** via OpenCV at >20 FPS on conveyor belt footage
- **Multi-class defect classification** — stains, holes, weave distortions, yarn slubs, shade variation, and more
- **LLM root-cause explanations** — e.g., *"irregular weave pattern suggests loom tension misalignment"*
- **Sinhala PDF reports** for local operator accessibility
- **Streamlit dashboard** with live video feed, bounding boxes, and confidence scores
- **Edge-deployable** with Docker — suitable for SME factory floors
- **Augmented training** for tropical conditions (humidity, lighting variation)
- **>92% accuracy** target on Sri Lankan fabric datasets

---

## 🏭 Problem Statement

Manual fabric inspection in Sri Lankan apparel factories suffers from:

| Challenge | Impact |
|-----------|--------|
| Shift fatigue | 20–30% human error rate |
| High conveyor speeds (30–50 m/min) | Missed micro-defects |
| Low-contrast flaws (<1mm) | Invisible to the naked eye |
| Variable humidity and lighting | Inconsistent inspection conditions |
| Rare defect imbalance | F1 < 0.80 on existing models |

This project directly addresses these gaps with a fatigue-free, explainable AI system.

---

## 🧩 System Architecture

```
Camera Feed (OpenCV)
        │
        ▼
 Frame Preprocessor
  (resize, normalize, augment)
        │
        ▼
EfficientNet-B4 (Fine-tuned)
  Defect Classification + Bounding Box
        │
        ├──── Defect Detected ────▶ RAG Pipeline (LangChain + Qdrant + Llama-3)
        │                                   │
        │                           Root-Cause Explanation
        │                                   │
        ▼                                   ▼
  Streamlit Dashboard ◀──── Alert + Confidence Score + Fix Suggestion
        │
        ▼
  CSV Batch Metrics / Sinhala PDF Report
```

---

## 🗂️ Project Structure

```
fabric-defect-recognizer/
├── data/
│   ├── raw/                    # Original dataset images
│   └── augmented/              # Tropical-augmented training data
├── models/
│   └── efficientnet/           # Fine-tuned model weights
├── src/
│   ├── detection/              # EfficientNet inference + OpenCV pipeline
│   ├── explainability/         # RAG root-cause generation (LangChain + Qdrant)
│   └── pipeline/               # Real-time conveyor belt processing
├── reports/
│   └── sinhala_pdf/            # Sinhala PDF report generator
├── dashboard/                  # Streamlit web app
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── docker/
│   └── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🧪 Defect Classes

| Defect Type | Description | Common Cause |
|-------------|-------------|--------------|
| Oil Stain | Dark patches on fabric surface | Machine lubrication leakage |
| Dye Stain / Color Bleed | Irregular color patches | Poor dyeing process control |
| Hole / Snag | Physical fabric damage | Yarn breaks, knitting needle errors |
| Drop Stitch / Run | Missing loops in knit structure | Knitting machine malfunction |
| Weave Distortion | Skewing, bowing, pattern irregularity | Loom tension misalignment |
| Slub / Nep | Thick yarn irregularities | Spinning process flaws |
| Shade Variation | Uneven color across fabric width | Inconsistent dye bath conditions |
| Shrinkage | Dimensional instability | Improper heat/moisture treatment |

---

## 📦 Datasets

| Dataset | Source | Size | Notes |
|---------|--------|------|-------|
| Kaggle Sri Lankan Fabric Defects | Kaggle (Moratuwa) | ~2k images | Stains, holes — local domain |
| Jasper Sri Lankan Textile Dataset | Jasper Research | — | Sri Lanka-specific |
| Lusitano Textile Dataset | Public | ~32k images | Multi-defect, large scale |
| Roboflow Fabric Defects | Roboflow Universe | Multi-class | Annotated bounding boxes |
| IndustrialTextileDataset | GitHub | — | Industrial inspection |

**Augmentation strategy:** Humidity simulation, varied lighting, CLAHE contrast enhancement, rotation, flipping — tailored for Western Province factory conditions.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Computer Vision | EfficientNet-B4 (PyTorch), OpenCV |
| Object Detection | YOLOv8 / custom bounding box head |
| LLM / RAG | Llama-3, LangChain, Qdrant |
| Dashboard | Streamlit |
| Report Generation | ReportLab (Sinhala PDF) |
| Deployment | Docker, edge-compatible |
| Experiment Tracking | MLflow / Weights & Biases |
| Dataset Management | Roboflow |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for training)
- Docker (for deployment)

### Installation

```bash
git clone https://github.com/your-username/fabric-defect-recognizer.git
cd fabric-defect-recognizer

pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys and paths
```

### Run the Dashboard

```bash
streamlit run dashboard/app.py
```

### Run Inference on a Video

```bash
python src/pipeline/run_inference.py --source data/sample_video.mp4 --model models/efficientnet/best.pt
```

### Train the Model

```bash
python src/detection/train.py --config configs/train_config.yaml
```

---

## 📊 Target Performance Metrics

| Metric | Target |
|--------|--------|
| Overall Accuracy | > 92% |
| F1 Score (rare defects) | > 0.80 |
| Inference Speed | > 20 FPS |
| Confidence Threshold | > 90% |

---

## 📄 Output Examples

- **Annotated frames** — bounding boxes with defect class + confidence score
- **LLM explanation** — *"92% confidence: oil stain detected. Likely cause: machine lubrication leakage near feed roller. Recommended action: clean with solvent X and inspect roller seal."*
- **Real-time alerts** — reject flags sent to production line controller
- **Batch CSV report** — F1, precision, recall, FPS per session
- **Sinhala PDF summary** — printable shift report for local operators

---

## 🌏 Local Context

This project is tailored for **Negombo and Western Province apparel factories** — a hub of Sri Lanka's garment export industry. Key adaptations:

- Training data sourced from or augmented to match **local knitted fabric types**
- **Tropical humidity** factored into preprocessing and augmentation
- Reports generated in **Sinhala** for operator accessibility
- Edge deployment designed for **SME budgets** without cloud dependency

---

## 🗺️ Roadmap

- [x] Project structure and dataset sourcing
- [ ] EfficientNet fine-tuning on local datasets
- [ ] RAG pipeline integration (Llama-3 + Qdrant)
- [ ] OpenCV real-time conveyor pipeline
- [ ] Streamlit dashboard with live feed
- [ ] Sinhala PDF report generator
- [ ] Docker deployment package
- [ ] Edge optimization (quantization, TensorRT)
- [ ] Factory pilot test (SME partner)

---


## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
