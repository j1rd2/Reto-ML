# Pokémon Type Classification — Deep Learning Project

This repository contains the complete implementation, training pipeline, and report for a multi-label image classification model that predicts **Pokémon elemental types** using deep learning.  
The project explores **custom CNNs** and **transfer learning** approaches (EfficientNetB0) with **fine-tuning** and **K-Fold validation**.

---

## 📁 Repository Structure

```
Reto-ML/
├── app.py
├── test.py
├── train.py
├── README.md
├── report/
│   ├── report.pdf
│   └── figures/...
│
├── models/
│   ├── __init__.py
│   ├── custom_cnn.py
│   ├── mobilenet.py
│   ├── efficientnet_b0.py
│   ├── finetune_efficientnet_b0.py
│   ├── finetune2_efficientnet_b0.py
│   ├── kf_finetunning.py
│   └── ...
│
├── data/
│   ├── dataset/
│   │   ├── train/
│   │   ├── val/
│   │   ├── test/
│   │   ├── labels_train.csv
│   │   ├── labels_val.csv
│   │   └── labels_test.csv
│   └── (ignored from repo)
│
├── export/
│   ├── pokemon_custom_cnn_v2.h5
│   ├── pokemon_efficientnet_b0_ft.h5
│   ├── pokemon_efficientnet_b0_ft2.h5
│   ├── pokemon_efficientnet_b0_kf3_ft2.h5
│   └── (trained model weights)
│
├── checkpoints/   (ignored)
├── logs/          (ignored)
└── callbacks/
    ├── F1Callback.py
    ├── LrLogger.py
    └── ...
```

---

## ⚙️ Main Components

### `train.py`
Main training script.  
It allows selecting which model to train (Custom CNN, MobileNet, EfficientNet, etc.) and automatically logs metrics, saves checkpoints, and exports the final model.

```bash
python3 train.py
```

---

### `test.py`
Evaluates a selected trained model on the test dataset and generates plots of key metrics (AUC-PR, AUC-ROC, F1-scores, Loss, etc.).

```bash
python3 test.py
```

---

### `app.py`
Gradio-based local web interface for interactive prediction.  
You can upload Pokémon images and get the Top-2 predicted elemental types.

```bash
python3 app.py
```

After running, open the local link shown in terminal (e.g., `http://127.0.0.1:7860/`).

---

## Model Training Variants

### `models/finetune_efficientnet_b0.py`
Fine-tuning script (Phase 1) that loads EfficientNetB0 pretrained on ImageNet and unfreezes the last layers for transfer learning.

```bash
python3 models/finetune_efficientnet_b0.py   --base_model_name pokemon_efficientnet_b0   --unfreeze_last 60   --lr 1e-5   --epochs 10   --batch_size 16
```

---

### `models/finetune2_efficientnet_b0.py`
Fine-tuning script (Phase 2) that uses a weighted binary cross-entropy loss to correct class imbalance and continue training from the previous checkpoint.

```bash
python3 models/finetune2_efficientnet_b0.py   --base_model_name pokemon_efficientnet_b0   --unfreeze_last 60   --lr 1e-5   --epochs 10   --batch_size 16
```

---

### `models/kf_finetunning.py`
Implements **K-Fold Cross-Validation** (default: 3 splits) for the fine-tuned EfficientNet model, training multiple folds to improve robustness and reduce variance.

```bash
python3 models/kf_finetunning.py   --base_model_name pokemon_efficientnet_b0   --unfreeze_last 60   --lr 1e-5   --epochs 10   --batch_size 16   --n_splits 3
```

All three scripts generate models derived from the base **EfficientNetB0** backbone.

---

## Supporting Modules

### `config/`
Contains the global configuration (list of types, paths, and constants used in training).

### `data/`
Manages dataset loading, preprocessing, and TensorFlow data pipeline creation.  
> Ignored in the repository to avoid heavy file sizes.

### `callbacks/`
Includes custom Keras callbacks:
- `F1Callback`: calculates F1-micro and best threshold per epoch.
- `LrLogger`: logs learning rate dynamics.
- Standard callbacks: `ModelCheckpoint`, `ReduceLROnPlateau`, `EarlyStopping`.

### `models/`
Contains all architecture definitions and registry (`MODEL_REGISTRY`), allowing easy model selection and consistent training/evaluation interfaces.

### `export/`
Stores exported `.h5` models for later inference or testing.

### `checkpoints/` and `logs/`
Ignored from Git to reduce repository weight.  
These directories store intermediate training weights and TensorBoard logs locally.

---

## Report

The full academic report (in LaTeX + PDF) is located in the `/report` folder.  
It includes:
- Dataset description and class distribution.
- Model architectures and training strategies.
- Fine-tuning phases and results.
- Qualitative prediction analysis.
- General conclusions and future work.

---

## Summary

This project demonstrates the effectiveness of combining **custom convolutional networks** with **transfer learning** techniques for small-scale, imbalanced datasets.  
Despite data limitations, results show that **simple CNNs**, when well-tuned, can achieve competitive performance relative to pretrained models — confirming that **model efficiency and interpretability** can outweigh raw complexity.

---

## Ignored Directories

- `/data` → Excluded to avoid dataset size in repository.  
- `/logs` → TensorBoard logs (ignored).  
- `/checkpoints` → Saved model weights (ignored).  
- `/export` → Contains only final `.h5` models used for testing and prediction.

---

## Quick Start

1. **Train a model**
   ```bash
   python3 train.py
   ```

2. **Evaluate the trained model**
   ```bash
   python3 test.py
   ```

3. **Run interactive prediction UI**
   ```bash
   python3 app.py
   ```

4. **Run fine-tuning (Phase 1 / Phase 2 / K-Fold)**
   ```bash
   python3 models/finetune_efficientnet_b0.py
   python3 models/finetune2_efficientnet_b0.py
   python3 models/kf_finetunning.py
   ```

---

## Acknowledgements

Developed as part of a deep learning research project for **Pokémon Type Classification**.  
Special thanks to TensorFlow, Keras, and Gradio for enabling a clean experimentation workflow and interactive model deployment.
