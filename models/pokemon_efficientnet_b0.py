from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0


def build_pokemon_efficientnet_b0(img_size=256, num_classes=18):
    """
    EfficientNetB0 para clasificación MULTI-LABEL de tipos Pokémon.

    Asume:
    - make_dataset entrega imágenes float32 en rango [0,1].
    - Y son vectores multi-hot (0/1 por tipo).

    Estrategia:
    - Data augmentation en el modelo.
    - Escalamos de [0,1] a [0,255] ANTES de EfficientNetB0,
      porque EfficientNetB0 ya incluye su propio Rescaling(1/255) interno.
    - Cabeza con sigmoid para multi-label.
    """

    inputs = keras.Input(shape=(img_size, img_size, 3), name="input")

    # 1) Data augmentation (se aplica sobre [0,1])
    x = layers.RandomFlip("horizontal", name="aug_flip")(inputs)
    x = layers.RandomRotation(0.10, name="aug_rot")(x)
    x = layers.RandomZoom(0.15, name="aug_zoom")(x)
    x = layers.RandomContrast(0.15, name="aug_contrast")(x)

    # 2) EfficientNetB0 espera [0,255], porque dentro tiene Rescaling(1/255)
    x = layers.Rescaling(255.0, name="to_255")(x)

    # 3) Backbone EfficientNetB0 pre-entrenado
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
        pooling="avg",  # ya hace GlobalAveragePooling2D
    )
    base_model.trainable = False  # Fase 1: solo cabeza

    x = base_model(x)

    # 4) Cabeza para multi-label
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.5, name="dropout_256")(x)

    outputs = layers.Dense(num_classes, activation="sigmoid", name="probs")(x)

    model = keras.Model(inputs, outputs, name="pokemon_efficientnet_b0")
    return model
