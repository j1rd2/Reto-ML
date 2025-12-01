# ============================================================
# kf_fintunning.py
#
# 3-Fold Fine-tuning (ft2-style) for EfficientNetB0
# - Combines labels_train + labels_val as full training pool
# - For each fold:
#     * Build EfficientNetB0 model from MODEL_REGISTRY
#     * Load Phase 1 best weights (frozen backbone training)
#     * Unfreeze last N layers (except BatchNorm)
#     * Use class-weighted BCE based on fold's train labels
#     * Train, save history JSON, best weights, and final .h5
# ============================================================

"""
python3 models/kf_finetunning.py \
  --base_model_name pokemon_efficientnet_b0 \
  --unfreeze_last 60 \
  --lr 1e-5 \
  --epochs 10 \
  --batch_size 16 \
  --n_splits 3
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from pathlib import Path

from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import TYPES
from data import build_dataframe, make_dataset
from models import MODEL_REGISTRY
from callbacks import F1Callback, LrLogger


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_name", type=str,
                    default="pokemon_efficientnet_b0",
                    help="Base model name trained in Phase 1.")
    ap.add_argument("--train_csv", type=str,
                    default="data/dataset/labels_train.csv")
    ap.add_argument("--val_csv", type=str,
                    default="data/dataset/labels_val.csv")
    ap.add_argument("--train_dir", type=str,
                    default="data/dataset/train")
    ap.add_argument("--val_dir", type=str,
                    default="data/dataset/val")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="Learning rate for fine-tuning.")
    ap.add_argument("--unfreeze_last", type=int, default=60,
                    help="Number of last layers to unfreeze.")
    ap.add_argument("--n_splits", type=int, default=3,
                    help="Number of folds for K-Fold CV.")
    return ap.parse_args()


def unfreeze_last_layers(model, n_last: int):
    """
    Unfreeze the last n_last layers of the full model,
    keeping BatchNormalization layers frozen.
    """
    for layer in model.layers:
        layer.trainable = False

    for layer in model.layers[-n_last:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    trainable_count = sum(1 for l in model.layers if l.trainable)
    print(f"Unfreezing last {n_last} layers -> Trainable layers: {trainable_count}")


def build_weighted_bce(y_train: np.ndarray):
    """
    Compute inverse-frequency class weights on y_train and return weighted BCE.
    y_train shape: (N, num_classes)
    """
    freq = y_train.mean(axis=0)  # positive rate per class
    pos_weight = 1.0 / (freq + 1e-6)
    pos_weight = pos_weight / pos_weight.mean()

    print("\n=== Class Frequency and Weights (Fold Train) ===")
    for t, f, w in zip(TYPES, freq, pos_weight):
        print(f"{t:<10} freq={f:.4f}  weight={w:.2f}")

    def weighted_bce(y_true, y_pred):
        # y_true, y_pred: (batch, num_classes)
        w = y_true * pos_weight + (1.0 - y_true) * 1.0
        bce = keras.backend.binary_crossentropy(y_true, y_pred)
        return keras.backend.mean(bce * w, axis=-1)

    return weighted_bce


def main():
    args = parse_args()
    img_size = args.img_size
    batch_size = args.batch_size
    n_splits = args.n_splits

    base_model_name = args.base_model_name
    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    log_dir  = Path("logs");        log_dir.mkdir(exist_ok=True)
    out_dir  = Path("export");      out_dir.mkdir(exist_ok=True)

    # === 1) Cargar train y val originales y combinarlos ===
    print("Loading train CSV...")
    Xtr, Ytr = build_dataframe(args.train_csv, args.train_dir)

    print("Loading val CSV...")
    Xva, Yva = build_dataframe(args.val_csv, args.val_dir)

    # Concatenar
    X_all = np.concatenate([Xtr["path"].values, Xva["path"].values], axis=0)
    Y_all = np.concatenate([Ytr.values, Yva.values], axis=0)

    num_samples = X_all.shape[0]
    num_classes = Y_all.shape[1]
    print(f"\nTotal samples for K-Fold: {num_samples}, num_classes: {num_classes}")

    # === 2) K-Fold Split ===
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    base_ckpt = ckpt_dir / f"{base_model_name}.best.weights.h5"
    if not base_ckpt.exists():
        raise FileNotFoundError(
            f"Base checkpoint not found: {base_ckpt}\n"
            f"Run Phase 1 first with: python3 train.py --model {base_model_name}"
        )

    fold_idx = 0
    all_fold_results = []

    for train_index, val_index in kf.split(X_all):
        fold_idx += 1
        print("\n" + "=" * 60)
        print(f"Starting Fold {fold_idx}/{n_splits}")
        print("=" * 60)

        X_train_paths = X_all[train_index]
        Y_train = Y_all[train_index]
        X_val_paths = X_all[val_index]
        Y_val = Y_all[val_index]

        # === 3) Datasets para este fold ===
        train_ds = make_dataset(
            X_train_paths,
            Y_train,
            img_size=img_size,
            batch_size=batch_size,
            shuffle=True,
        )
        val_ds = make_dataset(
            X_val_paths,
            Y_val,
            img_size=img_size,
            batch_size=batch_size,
            shuffle=False,
        )

        # === 4) Construir modelo base y cargar pesos Phase 1 ===
        build_model_fn = MODEL_REGISTRY[base_model_name]
        model = build_model_fn(img_size=img_size, num_classes=num_classes)

        print(f"Loading Phase 1 best weights from: {base_ckpt}")
        model.load_weights(str(base_ckpt))

        # === 5) Fine-tuning config (ft2-style) ===
        unfreeze_last_layers(model, args.unfreeze_last)
        loss_fn = build_weighted_bce(Y_train)
        optimizer = keras.optimizers.Adam(learning_rate=args.lr)

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[
                keras.metrics.AUC(curve="PR", name="AUC-PR"),
                keras.metrics.AUC(curve="ROC", name="AUC-ROC"),
            ],
        )

        # === 6) Callbacks específicos por fold ===
        ft_name = f"{base_model_name}_kf{fold_idx}_ft2"
        fold_ckpt = ckpt_dir / f"{ft_name}.best.weights.h5"
        fold_log  = log_dir / ft_name

        f1_cb = F1Callback(val_ds)
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=str(fold_ckpt),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                verbose=1,
                min_lr=1e-7,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
            f1_cb,
            LrLogger(),
            keras.callbacks.TensorBoard(log_dir=str(fold_log)),
        ]

        # === 7) Entrenar este fold ===
        print(f"\n[Fold {fold_idx}] Training...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks,
        )

        # === 8) Guardar history y modelo del fold ===
        history_path = out_dir / f"{ft_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(history.history, f, indent=2)
        print(f"[Fold {fold_idx}] Saved history to: {history_path}")

        fold_model_path = out_dir / f"{ft_name}.h5"
        model.save(str(fold_model_path))
        print(f"[Fold {fold_idx}] Saved model to: {fold_model_path}")

        print(
            f"[Fold {fold_idx}] Best F1-micro: "
            f"{f1_cb.best_f1_:.4f} at thr={f1_cb.best_thr_:.2f}"
        )

        all_fold_results.append({
            "fold": fold_idx,
            "best_f1_micro": float(f1_cb.best_f1_),
            "best_thr": float(f1_cb.best_thr_),
        })

    # === 9) Resumen de folds ===
    summary_path = out_dir / f"{base_model_name}_kf_ft2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_fold_results, f, indent=2)
    print("\n=== K-Fold Summary ===")
    print(all_fold_results)
    print(f"Saved K-Fold summary to: {summary_path}")


if __name__ == "__main__":
    main()
