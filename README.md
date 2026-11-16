# 🔍 Cheque Fraud Detection using Ink Variation Analysis

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.46+-green?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success)

An AI-powered cheque fraud detection system that identifies bank cheques altered with a **second pen** (payee or amount changed after filling) by comparing the ink of handwritten fields — even when both pens are the **same colour**. Printed parts of the cheque (logo, labels, lines, security pattern, coloured background) are automatically removed before any analysis.

---

## 📊 Results (IDRBT Dataset — 5-Fold Cross-Validation Grouped by Cheque)

| Model | Cheque ROC-AUC | Accuracy | Precision | Recall | Ink-pair AUC |
|---|---|---|---|---|---|
| **Classical ML** — optical-density ink features + Random Forest | **0.966** | **92.9%** | 97.1% | 88.4% | 0.939 |
| CNN Pen Classifier (14 pens) | 0.912 | 83.9% | 91.3% | 75.0% | 0.892 |
| Siamese CNN | 0.880 | 83.5% | 91.2% | 74.1% | 0.878 |

> Every held-out cheque is scored twice on its amount fields: as scanned (pen 1 + pen 2 → tampered) and with pen 2's additions removed (pen 1 only → genuine). No cheque is ever in both training and test folds. Exact numbers are in `outputs/models/ink_metrics.json` and plots in `outputs/figures/`.

---

## ✨ Features

- **🔮 Multi-Model Detection:** Three complementary approaches — Classical ML (Random Forest on optical-density features), CNN Pen Classifier, and Siamese CNN — each with distinct performance trade-offs.
- **🖊️ Ink Analysis:** Measures optical density `OD = −log(I / I_paper)` per channel, cancelling paper tint; features include OD chromaticity, Beer–Lambert slopes, density percentiles, stroke width, edge sharpness, and texture.
- **🏦 Bank Template Matching:** Automatically aligns scans to blank cheque templates (ECC) and subtracts the printed background, isolating only handwritten ink.
- **📊 Interactive Streamlit App:** Upload a cheque image and get a real-time fraud verdict with Grad-CAM visualisations highlighting the suspicious ink regions.
- **🧪 Reproducible Pipeline:** One command re-runs the full 5-fold cross-validation from raw TIFF scans to final metrics.

---

## 🏗️ Pipeline

```
scan ─► match bank layout ─► align to blank template ─► subtract print ─► handwriting only
                                                                             │
verdict ◄─ coherent group of different ink? ◄─ P(different pen) for every pair ◄─ ink units
```

1. **Blank templates** — all IDRBT scans of a bank are aligned (ECC) and a high per-pixel percentile removes the handwriting, leaving the printed cheque.
2. **Handwriting isolation** — a pixel is ink only if it is darker than *any* printed pixel of the template within ±2 px. Logos, text, lines, security patterns and background colour can never pass. Only the filled fields are kept: payee, amount in words, amount in figures (date/signature optional — they are often pre-written by the account holder).
3. **Ink units & features** — handwriting is cut into ~letter-sized units. Ink colour is measured as optical density per channel; features are OD chromaticity (dye absorption ratios), Beer–Lambert slopes, density percentiles, stroke width, edge sharpness and texture.
4. **Pen comparison** — a model gives P(different pen) for every unit pair of the cheque.
5. **Decision** — a cheque is **FRAUD** when a coherent minority of units disagrees with the rest; the threshold is calibrated by cross-validation.

### 🏷️ Pen Labels without Manual Annotation
IDRBT has 112 cheques (9 writers, 14 pens: 7 blue, 7 black). Each was filled with pen 1 and altered with pen 2 **of the same colour**. `DATA.xlsx` records the exact text pen 2 wrote, which is always a prefix of the amount (e.g. `1` + `25000`, `one lakh` + `twenty five thousand`). The leading digits/words of the amount are labelled pen 2 and the rest pen 1 — **no manual annotation needed**.

---

## 🧠 Project Structure

```
├── app.py                      # Streamlit web app
├── config.py                   # paths & settings
├── make_samples.py             # builds demo cheques in data/samples/ from IDRBT
├── make_synthetic_genuine.py   # generates synthetic genuine cheques for testing
├── test_detection.py           # runs all models on the demo cheques
├── requirements.txt
├── src/
│   ├── ink/
│   │   ├── templates.py        # blank bank templates, layout matching, alignment, field boxes
│   │   ├── extraction.py       # handwriting isolation, ink units
│   │   ├── features.py         # optical-density ink features & patches
│   │   ├── idrbt.py            # dataset metadata, pen-2 prefix parsing
│   │   ├── dataset.py          # pen-labelled ink-unit dataset
│   │   ├── pairs.py            # within-cheque pairs, cheque score
│   │   └── models.py           # Classical / Siamese / CNN pen-difference models
│   ├── fraud_pipeline.py       # FraudDetector (end-to-end)
│   ├── train.py                # templates → dataset → CV → final models
│   ├── evaluate.py             # cross-validation, metrics, figures
│   └── gradcam.py              # Grad-CAM for the CNN models
└── tests/
    └── test_pipeline.py
```

---

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/ShwetShivang/Check-Fraud-Detection.git
cd Check-Fraud-Detection
```

### 2. Install dependencies

Make sure you have Python 3.10+ installed. Install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

Extract `IDRBT_Cheque_Image_Dataset.zip` into `data/raw/` (it contains `DATA.xlsx` and the `300/` folder of TIFF scans).

### 4. Train the models

```bash
python -m src.train          # ~15 min on CPU: templates, dataset, 5-fold CV, final models
```

### 5. (Optional) Generate demo cheques

```bash
python make_samples.py       # builds demo cheques for the app in data/samples/
```

### 6. Run the web application

```bash
streamlit run app.py
```

*The application will open in your default browser at `http://localhost:8501`.*

---

## 🛠️ Other Commands

```bash
# Run fraud detection on a single cheque image
python -m src.fraud_pipeline path/to/cheque.jpg --model classical

# Run detection on all demo cheques
python test_detection.py

# Run unit tests
python -m pytest tests -v
```

---

## ⚠️ Limitations

- Layouts other than the four IDRBT banks (Axis, ICICI, Canara, Syndicate) run in *generic* mode, where printed text cannot be subtracted; the app warns when this happens. Adding a bank only needs a few scans of its blank cheques to build a template plus field-box annotations.
- Training data is 112 cheques; same-colour pens are hard by design. The final models were trained on all IDRBT cheques, so results on the IDRBT demo cheques are optimistic — use the cross-validated numbers above for fair evaluation.

---

## 📄 Reference

> P. Dansena, S. Bag, R. Pal, *"Differentiating Pen Inks in Hand-written Bank Cheques Using Multi-Layer Perceptron"*, PReMI 2017.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/ShwetShivang/Check-Fraud-Detection/issues).

---

*Built with ❤️ by [ShwetShivang](https://github.com/ShwetShivang)*
