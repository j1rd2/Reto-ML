#!/usr/bin/env python3
import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input

# === Types in the same order as the model ===
TYPES = [
    "bug","dark","dragon","electric","fairy","fighting","fire","flying",
    "ghost","grass","ground","ice","normal","poison","psychic","rock","steel","water"
]

# Trained model path
MODEL_PATH = "export/pokemon_efficientnet_b0_kf3_ft2.h5"

# Image size used in training
IMG_SIZE = 256

# ---------------------------------------------------
# Flexible loader: soporta modelo viejo (Lambda+preprocess_input)
# y modelo nuevo (Rescaling / sin Lambda) sin cambiar el resto.
# ---------------------------------------------------
def load_flexible_model(model_path: str):
    print(f"Loading model from: {model_path}")
    # Para modelos con pérdidas custom (como ft2), evitamos reconstruir el compile.
    model = keras.models.load_model(model_path, compile=False)
    print("Model loaded successfully with compile=False.")
    return model

model = load_flexible_model(MODEL_PATH)

# ---------------------------------------------------
# Detectar cómo fue entrenado el modelo para preprocesar bien
# ---------------------------------------------------
layer_names = [layer.name for layer in model.layers]

if "rescale" in layer_names:
    # Nuevo modelo: tiene capa Rescaling(1/255) adentro
    # => aquí le pasamos [0,255] normal y la red lo normaliza.
    INPUT_MODE = "RAW_255"
elif "preprocess" in layer_names:
    # Modelo viejo: tiene Lambda(preprocess_input) adentro
    # => también espera [0,255] porque preprocess_input se encarga.
    INPUT_MODE = "RAW_255"
else:
    # No encontramos pistas claras: asumimos que el modelo espera [0,1]
    INPUT_MODE = "SCALED_01"

print(f"Input mode detected for preprocessing: {INPUT_MODE}")


def preprocess(image):
    """
    image: numpy array (H, W, 3) en [0,255] desde Gradio.
    Devuelve tensor [1, IMG_SIZE, IMG_SIZE, 3].
    Ajusta según cómo fue definido el modelo.
    """
    if image is None:
        return None

    img = tf.convert_to_tensor(image, dtype=tf.float32)

    # Si el modelo YA normaliza adentro (Rescaling o preprocess_input),
    # le dejamos los valores crudos en [0,255].
    # Si no, aquí normalizamos a [0,1].
    if INPUT_MODE == "SCALED_01":
        img = img / 255.0

    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.expand_dims(img, axis=0)
    return img


def predict(image):
    """
    Gradio callback:
    - Recibe imagen como numpy array.
    - Regresa texto + tabla de probabilidades.
    - Top-2 tipos más probables.
    """
    x = preprocess(image)
    if x is None:
        return "No image received.", []

    probs = model.predict(x, verbose=0)[0]  # shape: (num_classes,)

    # Top-2 índices
    top_indices = np.argsort(probs)[-2:][::-1]
    top_types = [(TYPES[i], float(probs[i])) for i in top_indices]

    top_set = set(top_indices)
    prob_table = [
        {
            "type": t,
            "probability": float(p),
            "active": (idx in top_set),
        }
        for idx, (t, p) in enumerate(zip(TYPES, probs))
    ]

    msg = "Top-2 predicted types: " + ", ".join(
        f"{t} ({p:.3f})" for t, p in top_types
    )

    return msg, prob_table


# === Gradio Interface ===

with gr.Blocks(title="Pokemon Type Classifier") as demo:
    gr.Markdown("# 🐾 Pokemon Type Classifier\nUpload an image and view the Top-2 predicted types.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="numpy",
                label="Pokémon Image",
                sources=["upload", "clipboard"],
            )
            btn = gr.Button("Predict")

        with gr.Column():
            output_text = gr.Textbox(
                label="Result",
                interactive=False,
            )
            output_table = gr.Dataframe(
                headers=["type", "probability", "active"],
                label="Type probabilities",
                interactive=False,
            )

    btn.click(
        fn=predict,
        inputs=[image_input],
        outputs=[output_text, output_table],
    )

if __name__ == "__main__":
    demo.launch()
