"""
Fine-tuning Phase 1 (ft1) for EfficientNetB0

How to run:

1) Train base EfficientNetB0 (Phase 1 - frozen backbone):
   python3 train.py --model pokemon_efficientnet_b0

2) Fine-tune last layers on top of Phase 1:
   python3 models/finetune_efficientnet_b0.py \
     --base_model_name pokemon_efficientnet_b0 \
     --unfreeze_last 40 \
     --lr 1e-5 \
     --epochs 10 \
     --batch_size 16
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
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
                    help="Nombre del modelo base entrenado en Fase 1.")
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
                    help="Learning rate para fine-tuning.")
    ap.add_argument("--unfreeze_last", type=int, default=40,
                    help="Número de capas finales a descongelar.")
    return ap.parse_args()


def unfreeze_last_layers(model, n_last: int):
    """
    Descongela las últimas n_last capas del modelo completo,
    manteniendo BatchNormalization layers congeladas (buena práctica en TL).
    """
    # Marcar todo como no entrenable
    for layer in model.layers:
        layer.trainable = False

    # Descongelar solo las últimas n_last capas (excepto BatchNorm)
    for layer in model.layers[-n_last:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    trainable_count = sum(1 for l in model.layers if l.trainable)
    print(f"Unfreezing last {n_last} layers. Trainable layers: {trainable_count}")


def main():
    args = parse_args()

    base_model_name = args.base_model_name

    # 1) Cargar data
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

    # 2) Reconstruir arquitectura base (misma que Fase 1)
    print(f"\nBuilding base model: {base_model_name}")
    build_model_fn = MODEL_REGISTRY[base_model_name]
    model = build_model_fn(img_size=args.img_size, num_classes=len(TYPES))

    # 3) Cargar best weights de Fase 1
    ckpt_dir = Path("checkpoints")
    base_ckpt = ckpt_dir / f"{base_model_name}.best.weights.h5"
    if not base_ckpt.exists():
        raise FileNotFoundError(
            f"Base checkpoint not found: {base_ckpt}. "
            f"Primero entrena Fase 1 con train.py --model {base_model_name}."
        )
    print(f"Loading base weights from: {base_ckpt}")
    model.load_weights(str(base_ckpt))

    # 4) Fine-tuning: descongelar últimas capas
    unfreeze_last_layers(model, args.unfreeze_last)

    # 5) Recompilar con LR bajo (fine-tuning)
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(curve="PR", name="AUC-PR"),
            keras.metrics.AUC(curve="ROC", name="AUC-ROC"),
            # Puedes agregar BinaryAccuracy si quieres, pero AUC+F1 son más informativos.
        ],
    )

    # 6) Callbacks para ft1
    ft_name = base_model_name + "_ft"
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

    # 7) Entrenar Fase ft1
    print("\nStarting EfficientNetB0 fine-tuning (ft1)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # 8) Guardar history para plot_history.py
    history_path = out_dir / f"{ft_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2)
    print(f"Saved fine-tuning history to {history_path}")

    # 9) Guardar modelo fine-tuned completo
    ft_model_path = out_dir / f"{ft_name}.h5"
    model.save(str(ft_model_path))
    print(f"\nSaved fine-tuned model to {ft_model_path}")
    print(f"Best F1-micro (ft1): {f1_cb.best_f1_:.4f} at thr={f1_cb.best_thr_:.2f}")


if __name__ == "__main__":
    main()
