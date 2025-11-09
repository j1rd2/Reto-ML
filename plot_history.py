# plot_history.py

"""
How to use:
python plot_history.py --history export/pokemon_custom_cnn_history.json --model_name "Custom CNN"
python plot_history.py --history export/pokemon_custom_cnn_v2_history.json --model_name "Custom CNN v2"
python plot_history.py --history export/pokemon_efficientNet_history.json --model_name "EfficientNetB0"
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt


def plot_history(history_path, model_name=None, out_dir="plots"):
    history_path = Path(history_path)
    with open(history_path, "r") as f:
        hist = json.load(f)

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # Nombre bonito
    name = model_name or history_path.stem.replace("_history", "")

    # Ejes X
    epochs = range(1, len(hist["loss"]) + 1)

    def save_plot(y_train_key, y_val_key=None, ylabel="", title="", filename=""):
        if y_train_key not in hist:
            return

        plt.figure()
        plt.plot(epochs, hist[y_train_key], label="train")

        if y_val_key and y_val_key in hist:
            plt.plot(epochs, hist[y_val_key], label="val")

        plt.xlabel("Epoch")
        plt.ylabel(ylabel or y_train_key)
        plt.title(title or f"{name} - {ylabel or y_train_key}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=200)
        plt.close()

    # 1) Loss
    save_plot(
        "loss",
        "val_loss",
        ylabel="Binary crossentropy",
        title=f"{name} - Training vs Validation Loss",
        filename=f"{name}_loss.png",
    )

    # 2) AUC-PR
    save_plot(
        "AUC-PR",
        "val_AUC-PR",
        ylabel="AUC-PR",
        title=f"{name} - Precision-Recall AUC",
        filename=f"{name}_auc_pr.png",
    )

    # 3) AUC-ROC
    save_plot(
        "AUC-ROC",
        "val_AUC-ROC",
        ylabel="AUC-ROC",
        title=f"{name} - ROC AUC",
        filename=f"{name}_auc_roc.png",
    )

    # 4) F1-micro validación (solo valid, viene de tu callback)
    if "val_f1_micro" in hist:
        plt.figure()
        plt.plot(epochs, hist["val_f1_micro"], label="val_f1_micro")
        plt.xlabel("Epoch")
        plt.ylabel("F1-micro")
        plt.title(f"{name} - Validation F1-micro")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_f1_micro.png", dpi=200)
        plt.close()

    # 5) Learning Rate
    if "lr" in hist:
        plt.figure()
        plt.plot(epochs, hist["lr"])
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.title(f"{name} - Learning rate schedule")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_lr.png", dpi=200)
        plt.close()

    print(f"Plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--history",
        required=True,
        help="Ruta al archivo *_history.json generado por train.py",
    )
    ap.add_argument(
        "--model_name",
        default=None,
        help="Nombre opcional para títulos de gráficas",
    )
    ap.add_argument(
        "--out_dir",
        default="plots",
        help="Directorio donde guardar las imágenes",
    )
    args = ap.parse_args()

    plot_history(args.history, model_name=args.model_name, out_dir=args.out_dir)