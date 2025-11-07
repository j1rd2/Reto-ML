#!/usr/bin/env python3
"""
pokemon_train_custom_cnn.py
---------------------------------
Entrena un clasificador multi-etiqueta (18 tipos) con una CNN propia (sin backbone),
usando data augmentation en GPU/MPS (Apple Silicon friendly).

Uso:
  python pokemon_train_custom_cnn.py \
    --train_csv dataset/labels_train.csv \
    --val_csv   dataset/labels_val.csv \
    --train_dir dataset/train \
    --val_dir   dataset/val \
    --epochs 25 --batch_size 32 --img_size 224
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# === 18 tipos (orden fijo esperado en el CSV) ===
TYPES = [
    "bug","dark","dragon","electric","fairy","fighting","fire","flying",
    "ghost","grass","ground","ice","normal","poison","psychic","rock","steel","water"
]

# Extensiones permitidas para las imágenes
IMG_EXTS = [".jpg"]


# =========================
# Utilidades de paths / CSV
# =========================

def resolve_path_py(stem: str, root: str):
    """
    Resuelve la ruta de la imagen a partir del id (stem) y del directorio root.
    """
    root_path = Path(root)
    # Probar extensiones conocidas
    for ext in IMG_EXTS + [e.upper() for e in IMG_EXTS]:
        p = root_path / f"{stem}{ext}"
        if p.exists():
            return str(p)
    # Último recurso: cualquier extensión
    cand = list(root_path.glob(f"{stem}.*"))
    if cand:
        return str(cand[0])
    raise FileNotFoundError(f"No image found for id/stem='{stem}' under {root_path}")


def build_dataframe(csv_path: str, root_dir: str):
    """
    Construye dos DataFrames:
      X: con columna 'path'
      Y: con columnas de tipos (multi-label)
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in TYPES if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing label columns: {missing}")

    paths = [resolve_path_py(stem, root_dir) for stem in df["id"].astype(str)]
    X = pd.DataFrame({"path": paths})
    Y = df[TYPES].astype("float32").copy()
    return X, Y


# =========================
# Datasets (tf.data)
# =========================

def make_dataset(paths,
                 labels,
                 img_size=224,
                 batch_size=32,
                 shuffle=True):
    """
    Crea un tf.data.Dataset para (paths, labels).
    Solo carga, decodifica, redimensiona y normaliza [0,1].
    El data augmentation se hace dentro del modelo en GPU/MPS.
    """
    AUTOTUNE = tf.data.AUTOTUNE

    def _load_preprocess(path, y):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img = tf.image.resize(img, [img_size, img_size])
        return img, y

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)
    ds = ds.map(_load_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


# =========================
# Modelo: CNN custom
# =========================

def conv_block(x, filters):
    """
    Bloque Conv2D -> BatchNorm -> ReLU
    """
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_custom_cnn(img_size=224, num_classes=18):
    """
    CNN ligera para clasificación multi-etiqueta.
    Data augmentation en GPU/MPS usando capas de Keras.
    """
    inputs = keras.Input(shape=(img_size, img_size, 3), name="input")

    # Data augmentation en el modelo (sólo activo en training)
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    # Normalización opcional extra (si quisieras centrar):
    # aquí ya vienen en [0,1] desde el dataset, así que lo dejamos así.
    x = data_augmentation(inputs)

    # Bloque 1
    x = conv_block(x, 16)
    x = layers.MaxPooling2D()(x)

    # Bloque 2
    x = conv_block(x, 32)
    x = layers.MaxPooling2D()(x)

    # Bloque 3
    x = conv_block(x, 64)
    x = layers.MaxPooling2D()(x)

    # De mapas 3D -> vector 1D
    x = layers.GlobalAveragePooling2D()(x)

    # Capa densa oculta
    x = layers.Dense(128, activation="relu")(x)

    # Regularización
    x = layers.Dropout(0.4)(x)

    # Salida multi-label
    outputs = layers.Dense(num_classes, activation="sigmoid", name="probs")(x)

    model = keras.Model(inputs, outputs, name="pokemon_custom_cnn")
    return model


# =========================
# Callbacks: F1 y LR logger
# =========================

class F1Callback(keras.callbacks.Callback):
    """
    Calcula F1-micro en validación barriendo umbral en [0.2, 0.8].
    Guarda el mejor F1 y el mejor umbral como atributos.
    """
    def __init__(self, val_ds, thresholds=np.linspace(0.2, 0.8, 13)):
        super().__init__()
        self.val_ds = val_ds
        self.thresholds = thresholds
        self.best_thr_ = 0.5
        self.best_f1_ = 0.0

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_prob = [], []
        for xb, yb in self.val_ds:
            y_true.append(yb.numpy())
            y_prob.append(self.model.predict(xb, verbose=0))

        Y = np.vstack(y_true).astype(int)
        P = np.vstack(y_prob)

        best_t, best_f1 = self.best_thr_, self.best_f1_

        for t in self.thresholds:
            pred = (P >= t).astype(int)
            tp = (pred & (Y == 1)).sum()
            fp = (pred & (Y == 0)).sum()
            fn = ((1 - pred) & (Y == 1)).sum()
            denom = (2 * tp + fp + fn)
            f1_micro = 0.0 if denom == 0 else (2 * tp) / denom
            if f1_micro > best_f1:
                best_f1, best_t = f1_micro, t

        self.best_f1_ = best_f1
        self.best_thr_ = best_t

        if logs is not None:
            logs["val_f1_micro"] = best_f1
            logs["val_best_thr"] = best_t

        print(f"\n[Val] best F1-micro={best_f1:.4f} at thr={best_t:.2f}")


class LrLogger(keras.callbacks.Callback):
    """
    Registra el learning rate actual en logs al final de cada epoch.
    Compatible con schedulers y distintos atributos.
    """
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        opt = self.model.optimizer
        # Soporta tanto opt.lr como opt.learning_rate, según versión
        lr = opt.lr if hasattr(opt, "lr") else opt.learning_rate
        lr = float(tf.keras.backend.get_value(lr))
        logs["lr"] = lr
        print(f" - lr: {lr:.6f}")


# =========================
# Entrenamiento
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="dataset/labels_train.csv")
    ap.add_argument("--val_csv", type=str, default="dataset/labels_val.csv")
    ap.add_argument("--train_dir", type=str, default="dataset/train")
    ap.add_argument("--val_dir", type=str, default="dataset/val")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

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

    print("Building model...")
    model = build_custom_cnn(img_size=args.img_size, num_classes=len(TYPES))
    model.summary()

    # Optimizador + pérdida + métricas
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

    # Paths de salida
    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    log_dir  = Path("logs");        log_dir.mkdir(exist_ok=True)
    out_dir  = Path("export");      out_dir.mkdir(exist_ok=True)

    f1_cb = F1Callback(val_ds)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "custom_cnn.best.weights.h5"),
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
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        f1_cb,
        LrLogger(),
        keras.callbacks.TensorBoard(log_dir=str(log_dir)),
    ]

    print("Training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Guardar modelo final
    model.save(str(out_dir / "pokemon_custom_cnn"))
    print("Saved model to export/pokemon_custom_cnn")
    print(f"Best F1-micro: {f1_cb.best_f1_:.4f} at thr={f1_cb.best_thr_:.2f}")


if __name__ == "__main__":
    main()
