# models/pokemon_efficientnet.py

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input


def build_pokemon_efficientnet(
    img_size=256,
    num_classes=18,
    base_trainable=False,
    dropout_rate=0.4,
):
    """
    Clasificación multi-etiqueta usando EfficientNetB0 como backbone.
    - Pesos preentrenados en ImageNet.
    - Salida con sigmoid para multi-label.
    """

    inputs = keras.Input(shape=(img_size, img_size, 3), name="input")

    # Data augmentation (solo en entrenamiento)
    x = layers.RandomFlip("horizontal", name="aug_flip")(inputs)
    x = layers.RandomRotation(0.08, name="aug_rot")(x)
    x = layers.RandomZoom(0.1, name="aug_zoom")(x)
    x = layers.RandomContrast(0.1, name="aug_contrast")(x)

    # Preprocesamiento EfficientNet (normaliza como espera el backbone)
    x = layers.Lambda(preprocess_input, name="preprocess")(x)

    # Backbone EfficientNetB0 (sin la cabeza final)
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )

    # Congelar o no el backbone
    base_model.trainable = base_trainable

    # Extracción de features
    x = base_model.output

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Cabeza densa
    x = layers.Dense(256, activation="relu", name="dense_hidden")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    # Salida multi-label
    outputs = layers.Dense(
        num_classes,
        activation="sigmoid",
        name="probs",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="pokemon_efficientNet")

    return model