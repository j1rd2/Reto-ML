#!/usr/bin/env python3
import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow import keras

# === Tipos en el mismo orden que el modelo ===
TYPES = [
    "bug","dark","dragon","electric","fairy","fighting","fire","flying",
    "ghost","grass","ground","ice","normal","poison","psychic","rock","steel","water"
]

# Ruta del modelo entrenado
MODEL_PATH = "export/pokemon_custom_cnn_v2.h5"

# Tamaño de imagen usado en el entrenamiento
IMG_SIZE = 256

# Umbral sugerido (puedes usar el que te dio F1Callback, por ej. 0.25)
DEFAULT_THRESHOLD = 0.25

# Cargar modelo una sola vez
print(f"Cargando modelo desde: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)

def preprocess(image):
    """
    image: viene como numpy array (H, W, 3) en [0,255] desde gradio.
    Devuelve tensor [1, IMG_SIZE, IMG_SIZE, 3] en [0,1].
    """
    if image is None:
        return None
    img = tf.convert_to_tensor(image, dtype=tf.float32)
    img = img / 255.0
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.expand_dims(img, axis=0)
    return img

def predict(image, threshold=DEFAULT_THRESHOLD):
    """
    Función que Gradio llama:
    - image: numpy array
    - threshold: float del slider
    """
    x = preprocess(image)
    if x is None:
        return "No se recibió imagen.", []

    # Predicción
    probs = model.predict(x, verbose=0)[0]  # shape: (num_classes,)

    # Binarizar con threshold
    preds = (probs >= threshold).astype(int)

    # Armar tabla de probabilidades
    prob_table = [
        {"type": t, "probability": float(p), "active": bool(b)}
        for t, p, b in zip(TYPES, probs, preds)
    ]

    # Etiquetas activas
    active = [f"{t} ({probs[i]:.3f})" for i, t in enumerate(TYPES) if preds[i] == 1]

    if not active:
        active_msg = "Ningún tipo supera el umbral. Prueba con un threshold más bajo."
    else:
        active_msg = "Tipos predichos: " + ", ".join(active)

    return active_msg, prob_table

# === Interfaz Gradio ===

with gr.Blocks(title="Pokemon Type Classifier") as demo:
    gr.Markdown("# 🐾 Pokemon Type Classifier\nArrastra una imagen y obtén los tipos probables.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="numpy",
                label="Imagen del Pokémon",
                sources=["upload", "clipboard"],  # sin 'drag-and-drop'
            )
            threshold_input = gr.Slider(
                minimum=0.05,
                maximum=0.9,
                value=DEFAULT_THRESHOLD,
                step=0.05,
                label="Umbral para activar etiquetas"
            )
            btn = gr.Button("Predecir")

        with gr.Column():
            output_text = gr.Textbox(
                label="Resultado",
                interactive=False
            )
            output_table = gr.Dataframe(
                headers=["type", "probability", "active"],
                label="Probabilidades por tipo",
                interactive=False
            )

    btn.click(
        fn=predict,
        inputs=[image_input, threshold_input],
        outputs=[output_text, output_table],
    )

if __name__ == "__main__":
    demo.launch()