# models/pokemon_custom_cnn.py

from tensorflow import keras
from tensorflow.keras import layers


def conv_block(x, filters):
    """
    Bloque Conv2D -> BatchNorm -> ReLU
    """
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_pokemon_custom_cnn(img_size=256, num_classes=18):
    """
    CNN para clasificación multi-etiqueta.
    Data augmentation en el modelo.
    """
    inputs = keras.Input(shape=(img_size, img_size, 3), name="input")

    # Data augmentation dentro del modelo.
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

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
    x = layers.Dense(256, activation="relu")(x)

    # Regularización
    x = layers.Dropout(0.4)(x)

    # Salida multi-label
    outputs = layers.Dense(num_classes, activation="sigmoid", name="probs")(x)

    model = keras.Model(inputs, outputs, name="pokemon_custom_cnn")
    return model
