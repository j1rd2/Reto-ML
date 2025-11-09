# train.py
import json
import argparse
from pathlib import Path
from tensorflow import keras
from config import TYPES
from data import build_dataframe, make_dataset
from callbacks import F1Callback, LrLogger
from models import MODEL_REGISTRY


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Nombre del modelo en MODEL_REGISTRY. Si se omite, se mostrará un menú interactivo."
    )
    ap.add_argument("--train_csv", type=str, default="data/dataset/labels_train.csv")
    ap.add_argument("--val_csv", type=str, default="data/dataset/labels_val.csv")
    ap.add_argument("--train_dir", type=str, default="data/dataset/train")
    ap.add_argument("--val_dir", type=str, default="data/dataset/val")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    return ap.parse_args()


def choose_model_interactively():
    """
    Muestra un menú en consola con los modelos disponibles en MODEL_REGISTRY
    y devuelve el nombre elegido.
    """
    print("\n=== Selección de modelo ===")
    keys = list(MODEL_REGISTRY.keys())
    for i, name in enumerate(keys, start=1):
        print(f"[{i}] {name}")

    while True:
        choice = input(f"Elige el modelo (1-{len(keys)}): ").strip()
        if not choice.isdigit():
            print("Por favor ingresa un número válido.")
            continue
        idx = int(choice)
        if 1 <= idx <= len(keys):
            selected = keys[idx - 1]
            print(f"Has seleccionado: {selected}\n")
            return selected
        else:
            print("Opción fuera de rango, intenta de nuevo.")


def main():
    args = parse_args()

    # Selección del modelo:
    if args.model is None:
        # Si no se pasó por CLI, mostrar menú interactivo
        args.model = choose_model_interactively()
    else:
        # Validar que exista en el registro
        if args.model not in MODEL_REGISTRY:
            raise ValueError(
                f"Modelo '{args.model}' no encontrado. Opciones disponibles: {list(MODEL_REGISTRY.keys())}"
            )

    build_model_fn = MODEL_REGISTRY[args.model]

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

    print(f"\nBuilding model: {args.model}")
    model = build_model_fn(img_size=args.img_size, num_classes=len(TYPES))
    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=args.lr)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(curve="PR", name="AUC-PR"),
            keras.metrics.AUC(curve="ROC", name="AUC-ROC"),
            keras.metrics.BinaryAccuracy(name="bin_acc", threshold=0.5),
        ],
    )

    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    log_dir  = Path("logs");        log_dir.mkdir(exist_ok=True)
    out_dir  = Path("export");      out_dir.mkdir(exist_ok=True)

    f1_cb = F1Callback(val_ds)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / f"{args.model}.best.weights.h5"),
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
            min_lr=1e-6,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        f1_cb,
        LrLogger(),
        keras.callbacks.TensorBoard(log_dir=str(log_dir)),
    ]

    print("Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # === Save history ===
    history_path = out_dir / f"{args.model}_history.json"
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2)
    print(f"Saved training history to {history_path}")

    export_path = out_dir / args.model
    model.save(str(export_path) + ".h5")
    print(f"Saved model to {export_path}")
    print(f"Best F1-micro: {f1_cb.best_f1_:.4f} at thr={f1_cb.best_thr_:.2f}")


if __name__ == "__main__":
    main()
