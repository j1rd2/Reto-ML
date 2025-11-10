#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from data import build_dataframe, make_dataset
from config import TYPES
from tensorflow import keras

IMG_SIZE = 256
BATCH_SIZE = 25
THRESHOLD = 0.25
MODEL_NAME = "pokemon_efficientnet_b0_kf3_ft2"
MODEL_PATH = f"export/{MODEL_NAME}.h5"

# 1. Load test
Xte, Yte = build_dataframe("data/dataset/labels_test.csv", "data/dataset/test")
test_ds = make_dataset(
    Xte["path"].values,
    Yte.values,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

if "ft2" in MODEL_NAME:
    # Cargar sin compilar (hay loss custom adentro)
    print(f"Loading (compile=False) model from: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH, compile=False)
    # recompilar con BCE estándar
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(curve="PR", name="AUC-PR"),
            keras.metrics.AUC(curve="ROC", name="AUC-ROC"),
        ],
    )
else:
    print(f"Loading model from: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH)


# 3. Evaluate
result = model.evaluate(test_ds, return_dict=True, verbose=1)
print("Keras evaluate() results on test:")
print(result)

# 4. Predict
y_prob = model.predict(test_ds, verbose=1)
y_true = Yte.values.astype(int)
y_pred = (y_prob >= THRESHOLD).astype(int)

# 5. Extra metrics
f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
precision_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
recall_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)

print("\nAdditional test metrics:")
print(f"F1-micro:        {f1_micro:.4f}")
print(f"F1-macro:        {f1_macro:.4f}")
print(f"Precision-micro: {precision_micro:.4f}")
print(f"Recall-micro:    {recall_micro:.4f}")

# 6. Bar plot
metrics = {
    "Loss":        result.get("loss", np.nan),
    "AUC-PR":      result.get("AUC-PR", np.nan),
    "AUC-ROC":     result.get("AUC-ROC", np.nan),
    "Binary Acc.": result.get("bin_acc", np.nan),
    "F1-micro":    f1_micro,
    "F1-macro":    f1_macro,
    "Prec-micro":  precision_micro,
    "Rec-micro":   recall_micro,
}

names = list(metrics.keys()) 
values = list(metrics.values())

plt.figure(figsize=(10, 5))
plt.bar(names, values)
plt.title(f"{MODEL_NAME} - Test Metrics")
plt.ylabel("Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{MODEL_NAME}_test_metrics.png", dpi=200)
plt.close()

print(f"\nSaved test metrics plot to: {MODEL_NAME}_test_metrics.png")