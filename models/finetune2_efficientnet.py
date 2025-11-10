# ============================================================
# Fine-tuning Phase 2 (finetune2) for EfficientNetB0
# - Loads Phase 1 best weights
# - Unfreezes last N layers (except BatchNorm)
# - Uses class-weighted BCE to handle imbalance
# - Saves history JSON + best weights + final .h5
# ============================================================

"""
python3 models/finetune2_efficientnet.py \
  --base_model_name pokemon_efficientnet_b0 \
  --unfreeze_last 60 \
  --lr 1e-5 \
  --epochs 10 \
  --batch_size 16
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from pathlib import Path
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
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="Learning rate for fine-tuning.")
    ap.add_argument("--unfreeze_last", type=int, default=60,
                    help="Number of last layers to unfreeze.")
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


def build_weighted_bce(Ytr):
    """
    Compute inverse-frequency class weights and return weighted BCE loss.
    """
    freq = Ytr.values.mean(axis=0)  # positive rate per class
    pos_weight = 1.0 / (freq + 1e-6)
    pos_weight = pos_weight / pos_weight.mean()

    print("\n=== Class Frequency and Weights (Phase 2) ===")
    for t, f, w in zip(TYPES, freq, pos_weight):
        print(f"{t:<10} freq={f:.4f}  weight={w:.2f}")

    def weighted_bce(y_true, y_pred):
        # y_true, y_pred: (batch, num_classes)
        # Broadcast pos_weight: shape (num_classes,)
        w = y_true * pos_weight + (1.0 - y_true) * 1.0
        bce = keras.backend.binary_crossentropy(y_true, y_pred)
        return keras.backend.mean(bce * w, axis=-1)

    return weighted_bce


def main():
    args = parse_args()

    print("Loading CSVs...")
    Xtr, Ytr = build_dataframe(args.train_csv, args.train_dir)
    Xva, Yva = build_dataframe(args.val_csv, args.val_dir)

    print("Building datasets...")
    train_ds = make_dataset(
        Xtr["path"].values,
        Ytr.values,
        img_size=args.img_size,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_ds = make_dataset(
        Xva["path"].values,
        Yva.values,
        img_size=args.img_size,
        batch_size=args.batch_size,
        shuffle=False,
    )

    base_model_name = args.base_model_name
    print(f"\nBuilding base model: {base_model_name}")
    build_model_fn = MODEL_REGISTRY[base_model_name]
    model = build_model_fn(img_size=args.img_size, num_classes=len(TYPES))

    ckpt_dir = Path("checkpoints")
    base_ckpt = ckpt_dir / f"{base_model_name}.best.weights.h5"
    if not base_ckpt.exists():
        raise FileNotFoundError(
            f"Base checkpoint not found: {base_ckpt}. "
            f"Train Phase 1 first with train.py --model {base_model_name}."
        )
    print(f"Loading Phase 1 best weights from: {base_ckpt}")
    model.load_weights(str(base_ckpt))

    # Fine-tuning setup
    unfreeze_last_layers(model, args.unfreeze_last)
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    loss_fn = build_weighted_bce(Ytr)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            keras.metrics.AUC(curve="PR", name="AUC-PR"),
            keras.metrics.AUC(curve="ROC", name="AUC-ROC"),
        ],
    )

    ft_name = base_model_name + "_ft2"
    ft_ckpt = ckpt_dir / f"{ft_name}.best.weights.h5"
    log_dir = Path("logs"); log_dir.mkdir(exist_ok=True)
    out_dir = Path("export"); out_dir.mkdir(exist_ok=True)

    f1_cb = F1Callback(val_ds)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(ft_ckpt),
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
        keras.callbacks.TensorBoard(log_dir=str(log_dir / ft_name)),
    ]

    print("\n=== Starting Fine-tuning Phase 2 (finetune2) ===")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Save training history for plot_history.py
    history_path = out_dir / f"{ft_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2)
    print(f"Saved fine-tuning history to: {history_path}")

    # Save final fine-tuned model (without custom_objects requirement)
    ft_model_path = out_dir / f"{ft_name}.h5"
    model.save(str(ft_model_path))
    print(f"Saved fine-tuned model to: {ft_model_path}")

    print(f"Best F1-micro (Phase 2): {f1_cb.best_f1_:.4f} at thr={f1_cb.best_thr_:.2f}")


if __name__ == "__main__":
    main()
