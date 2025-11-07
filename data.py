# data.py

from pathlib import Path
import pandas as pd
import tensorflow as tf
from config import TYPES, IMG_EXTS


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
    Construye:
      X: DataFrame con columna 'path'
      Y: DataFrame con columnas de tipos (multi-label)
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in TYPES if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing label columns: {missing}")

    paths = [resolve_path_py(stem, root_dir) for stem in df["id"].astype(str)]
    X = pd.DataFrame({"path": paths})
    Y = df[TYPES].astype("float32").copy()
    return X, Y


def make_dataset(paths,
                 labels,
                 img_size=256,
                 batch_size=32,
                 shuffle=True):
    """
    Crea un tf.data.Dataset para (paths, labels).
    Solo:
      - lee
      - decodifica
      - normaliza a [0,1]
      - resize
    El data augmentation se hace dentro del modelo.
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