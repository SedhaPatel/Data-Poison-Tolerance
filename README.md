# Data Poison Tolerance & Fashion MNIST ML Pipeline

An end-to-end Machine Learning and Deep Learning pipeline designed to evaluate model robustness, performance, and tolerance on the Fashion MNIST dataset.

---

## 📌 Project Overview

This repository implements a modular, configuration-driven pipeline for training, evaluating, comparing, and running inference with two baseline architectures:
1. **Fine-Tuned Random Forest Classifier** (scikit-learn) with hyperparameter search and multi-core CPU scaling.
2. **Fine-Tuned Deep Convolutional Neural Network (CNN)** (PyTorch) with batch normalization, dropout regularization, and learning rate scheduling.

All training hyperparameters, paths, model architectures, and output destinations are centralized in [`config.yaml`](config.yaml).

---

## 📂 Project Structure

```text
Ml-Project/
├── config.yaml               # Centralized configuration (hyperparameters, paths, options)
├── config_utils.py           # Configuration loading, validation, and schema checks
├── model_architecture.py     # PyTorch CNN model definitions and checkpoint handlers
├── train.py                  # Master orchestration script
├── train_baseline_1_rf.py    # Random Forest training and hyperparameter tuning
├── train_baseline_2_cnn.py   # PyTorch CNN training loop with validation early stopping
├── compare_models.py         # Comparative analysis, metric tables, and visual benchmark plots
├── predict.py                # Unified CLI inference script for RF and CNN models
├── requirements.txt          # Python dependencies
└── .gitignore                # Excludes large binaries, dataset archives, and artifacts
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/SedhaPatel/Data-Poison-Tolerance.git
cd Data-Poison-Tolerance
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Dataset Setup
Download the Fashion MNIST dataset (`fashion-mnist_train.csv` and `fashion-mnist_test.csv`) and place them in the `data/` directory:
```text
data/
├── fashion-mnist_train.csv
└── fashion-mnist_test.csv
```

---

## ⚙️ Configuration

Modify [`config.yaml`](config.yaml) to customize:
* **Dataset paths and validation splits**
* **Random Forest hyperparameters** (`n_estimators`, `max_depth`, `criterion`, etc.)
* **CNN architecture and training settings** (`learning_rate`, `batch_size`, `epochs`, `optimizer`, `scheduler`, `early_stopping`)
* **Output paths** for saved weights (`models/`) and evaluation metrics/plots (`outputs/`)

---

## 🏋️‍♂️ Training & Evaluation

### Train All Models & Run Benchmark
```bash
python train.py --model all
```

### Train Individual Models
* **Random Forest:**
  ```bash
  python train.py --model rf
  ```
* **Deep CNN:**
  ```bash
  python train.py --model cnn
  ```

---

## 📊 Model Comparison
Run comparative evaluation on the test set:
```bash
python compare_models.py
```
This generates comparison metrics and confusion matrix plots under `outputs/`.

---

## 🔮 Inference & Predictions
Run predictions on test samples or custom pixel inputs:
```bash
# Predict using Random Forest
python predict.py --model rf --samples 5

# Predict using Deep CNN
python predict.py --model cnn --samples 5
```

---

## 📄 License
This project is licensed under the MIT License.
