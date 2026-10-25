

# Evolutionary Algorithms for Machine Unlearning
*(Reproducibility Repository)*

This repository contains the official implementation and experimental setup for reproducing the results presented in the paper: **"Evolutionary Machine Unlearning: [Insert Your Paper Title Here]"**.

---

## 📌 Abstract / Overview


---

## 📂 Repository Structure

```text
.
├── data/                    # Dataset downloaders and split scripts (Forget / Retrain splits)
├── experiments/             # Pre-configured experiment runner scripts
├── checkpoints/             # Saved model weights 
├── results/                 # Raw logs, figures, and summary tables
├── requirements.txt         # Python package dependencies
├── environment.yml          # Anaconda environment specification
└── README.md                # Project documentation

```

---
### 📥 Model Checkpoints

Due to size constraints, pre-trained model weights are hosted externally:
- `poisoned_model.pth` (~139 MB): [Download from Zenodo](https://zenodo.org/record/TODO)

Place the downloaded file into the `resources/poisoned_model/` directory before running evaluations.

---

# 🛠️ Environment Setup & Installation

Download following files and place them in the `resources/` directory:
resources/test_set/
resources/unlearn_set/
Source: ![kaggle](https://www.kaggle.com/competitions/neural-debris-removal-in-streak-detection-models/data)
---

## 📊 Dataset

---

# 🔬 Reproducing Main Experiments

---

# 📈 Evaluation & Verification Metrics

---
## 📄 License