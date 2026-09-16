# 🔍 Cheque Fraud Detection using Ink Variation Analysis

An AI-powered system that detects fraudulent modifications in bank cheque images by analyzing pen ink variations using deep learning and classical machine learning approaches.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📖 Overview

When fraudsters alter bank cheques by adding or modifying text, they typically use different pens. Even if the inks appear visually similar, subtle color and texture differences can be detected computationally. This project implements a complete pipeline for such detection using:

- **Siamese Network** — Main model using contrastive loss for ink similarity learning
- **CNN Baseline** — Standard classifier for comparison
- **Classical ML** — SVM/Random Forest with GLCM, LBP, and LAB features

## 🏗️ Architecture

```
Cheque Image → Preprocessing → Patch Extraction → Model Analysis → Fraud Verdict
                   ↓                  ↓                ↓
              LAB/Grayscale    Sliding Window    Similarity Map
              Ink Masking      Contour-based     Heatmap Overlay
                                                 Grad-CAM
```

## 📁 Project Structure

```
├── config.py                    # Central configuration
├── app.py                       # Streamlit web application
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── src/
│   ├── data_loader.py           # Dataset loading & synthetic generation
│   ├── preprocessing.py         # Image preprocessing (resize, LAB, augmentation)
│   ├── patch_generator.py       # Patch extraction & pair generation
│   ├── train.py                 # Training script for all models
│   ├── evaluate.py              # Evaluation metrics & comparison
│   ├── fraud_pipeline.py        # End-to-end fraud detection
│   ├── gradcam.py               # Grad-CAM visualization
│   ├── utils.py                 # Shared utilities
│   └── models/
│       ├── siamese_model.py     # Siamese Network + Contrastive Loss
│       ├── cnn_model.py         # CNN Baseline Classifier
│       └── classical_model.py   # GLCM + LBP + SVM/RF
│
├── data/
│   ├── raw/                     # Place Kaggle dataset here
│   ├── processed/               # Preprocessed data
│   └── synthetic/               # Auto-generated test data
│
├── outputs/
│   ├── models/                  # Saved model weights
│   ├── figures/                 # Training curves, confusion matrices
│   └── sample_results/          # Sample detection outputs
│
├── report/
│   └── report.md                # Full project report
│
└── tests/
    └── test_pipeline.py         # Unit tests
```

## 🚀 Setup & Installation

### 1. Prerequisites

- Python 3.8 or newer
- pip package manager

### 2. Create Virtual Environment

```bash
cd "Check Fraud Detection"
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset (Optional)

Download the IDRBT Cheque Dataset from:
https://www.kaggle.com/datasets/jdranpariya/cheque-data/data

Extract and place the contents in `data/raw/`.

> **Note**: The system will automatically generate synthetic data for testing if no real dataset is found.

## 🎯 Usage

### Train All Models

```bash
python src/train.py
```

This trains:
1. Siamese Network (~200K params, contrastive loss)
2. CNN Baseline (~100K params, cross-entropy)
3. Classical SVM & Random Forest

Training takes ~2-5 minutes on CPU.

### Evaluate Models

```bash
python src/evaluate.py
```

Generates:
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices
- ROC curves
- Model comparison charts

### Run Streamlit App

```bash
streamlit run app.py
```

Features:
- Upload cheque images
- Select detection model
- View fraud/genuine verdict with confidence
- Highlighted suspicious regions
- Similarity heatmap
- Grad-CAM visualization

### Run Tests

```bash
python -m pytest tests/test_pipeline.py -v
```

## 📊 Models

| Model | Type | Parameters | Description |
|-------|------|-----------|-------------|
| Siamese Network | Deep Learning | ~200K | Twin CNN branches with shared weights, contrastive loss |
| CNN Baseline | Deep Learning | ~100K | Standard classifier for ink type prediction |
| SVM | Classical ML | N/A | RBF kernel SVM with GLCM+LBP+LAB features |
| Random Forest | Classical ML | N/A | 100-tree ensemble with same features |

## 🔬 Methodology

1. **Patch Extraction**: 64×64 patches via sliding window (stride=32)
2. **Pair Generation**: Same-ink pairs (label=1) and different-ink pairs (label=0)
3. **Color Analysis**: LAB color space separates luminance from chromaticity
4. **Similarity Learning**: Contrastive loss pulls same-ink embeddings together
5. **Fraud Decision**: If >20% of cross-region patch pairs have low similarity → FRAUD

## 📄 Citation

```
P. Dansena, S. Bag, and R. Pal,
"Differentiating Pen Inks in Hand-written Bank Cheques Using Multi-Layer Perceptron",
Proc. of 7th International Conference on Pattern Recognition and Machine Intelligence (ICPRMI),
Kolkata, India, December 2017.
```

## 📝 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- IDRBT for the cheque dataset
- Dansena, Bag & Pal for the foundational research
- PyTorch, Scikit-learn, Streamlit communities
