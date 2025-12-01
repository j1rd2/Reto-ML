# callbacks.py

import numpy as np
import tensorflow as tf
from tensorflow import keras


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
    Loguea el learning rate actual al final de cada epoch.
    """
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        opt = self.model.optimizer
        lr = opt.lr if hasattr(opt, "lr") else opt.learning_rate
        lr = float(tf.keras.backend.get_value(lr))
        logs["lr"] = lr
        print(f" - lr: {lr:.6f}")
