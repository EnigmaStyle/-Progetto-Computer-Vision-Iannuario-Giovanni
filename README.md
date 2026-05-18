# ♻️ Waste Classifier

Automated waste image classification using a full Computer Vision pipeline.  
**Course:** Introduction to Computer Vision — Epicode Institute of Technology  
**Author:** Giovanni Iannuario

---

## Overview

Classifies waste images into 6 material categories: **cardboard, glass, metal, paper, plastic, trash**.

The project implements a complete CV pipeline including:
- Classical approach: HOG feature extraction + SVM classifier (baseline)
- Deep learning approach: EfficientNet-B0 fine-tuned with a custom classification head
- Grad-CAM visual explanations
- Gradio web application for interactive inference

---

## Pipeline Architecture

```
Input Image
    │
    ▼
Preprocessing (resize, normalize, augment)
    │
    ├──► HOG Features ──► SVM Classifier ──► Prediction (baseline)
    │
    └──► EfficientNet-B0 backbone ──► Custom Head ──► Prediction + Grad-CAM
                                                           │
                                                      Gradio Web App
```

---

## Setup

```bash
# Clone the repository
git clone https://github.com/EnigmaStyle/Progetto-Computer-Visioni-Iannuario-Giovanni

cd waste-classifier

# Install dependencies (Python 3.10+)
pip install -r requirements.txt

# Download dataset from Kaggle: garythung/trashnet
# Extract to: data/raw/dataset-resized/
# Expected structure:
#   data/raw/dataset-resized/cardboard/
#   data/raw/dataset-resized/glass/
#   ...
```

---

## Usage

### Train the classical model (HOG + SVM)
```bash
python train_classical.py
# With hyperparameter search:
python train_classical.py --grid_search
```

### Train the deep model (EfficientNet-B0)
```bash
python train_deep.py
# Custom epochs:
python train_deep.py --epochs_frozen 5 --epochs_unfreeze 25 --patience 7
```

### Launch the web app
```bash
python app.py
# Open: http://localhost:7860
```

> **Nota:** al primo avvio l'app scarica automaticamente i modelli pre-addestrati da Google Drive
> (`efficientnet_best.pth` ~17 MB, `svm_hog.joblib` ~350 MB). Assicurati di avere una connessione attiva.
> I file vengono salvati in `models/` e non vengono riscaricati nelle esecuzioni successive.

### Run tests
```bash
pytest tests/ -v
```

---

## Results

| Model           | Test Accuracy | Macro F1 |
|-----------------|--------------|----------|
| HOG + SVM       | 63.7%        | 0.620    |
| EfficientNet-B0 | **93.2%**    | **0.924**|

**EfficientNet-B0 per-class F1:** cardboard 0.96 · glass 0.96 · paper 0.95 · metal 0.92 · trash 0.88 · plastic 0.89

Training plots and confusion matrices are saved to `results/` after training.

---

## Project Structure

```
waste-classifier/
├── src/
│   ├── utils.py            # Constants, seed, device helpers
│   ├── preprocessing.py    # DataLoaders, augmentation transforms
│   ├── features.py         # HOG feature extraction
│   ├── classical_model.py  # SVM pipeline with save/load
│   ├── deep_model.py       # EfficientNet-B0 with custom head
│   ├── gradcam.py          # Grad-CAM implementation
│   ├── evaluate.py         # Metrics computation and plotting
│   └── model_downloader.py # Auto-download pre-trained models from Google Drive
├── tests/                  # pytest test suite
├── docs/
│   └── technical_report.pdf # Technical analysis report
├── app.py                  # Gradio web application
├── train_classical.py      # CLI: train HOG+SVM
├── train_deep.py           # CLI: fine-tune EfficientNet
├── requirements.txt
└── README.md
```

---

## Ethical Considerations

- **Dataset bias:** TrashNet images are captured in controlled conditions; performance may degrade in real-world, dirty, or occluded settings.
- **Class imbalance:** The `trash` catch-all class may absorb edge cases and inflate error rates for recyclables.
- **Environmental cost:** Mislabelling recyclable items as `trash` has direct environmental consequences — recall for recyclable classes should be prioritized in deployment.
- **Privacy:** The dataset contains no personal data; all images are of objects only.

---

## Dataset

[TrashNet](https://github.com/garythung/trashnet) by Gary Thung and Mindy Yang  
Available on Kaggle: `garythung/trashnet`
