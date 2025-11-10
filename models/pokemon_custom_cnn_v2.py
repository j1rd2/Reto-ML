from tensorflow import keras
from tensorflow.keras import layers


def conv_block(x, filters, stride=1):
    """
    Bloque Conv2D -> BatchNorm -> ReLU
    Permite cambiar stride para reducir tamaño espacial.
    """
    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_pokemon_custom_cnn_v2(img_size=256, num_classes=18):
    """
    CNN para clasificación multi-etiqueta.
    Downsampling con conv stride=2.
    """
    inputs = keras.Input(shape=(img_size, img_size, 3), name="input")

    # Data augmentation
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
    x = conv_block(x, 16)
    x = conv_block(x, 32, stride=2)

    # Bloque 2
    x = conv_block(x, 32)
    x = conv_block(x, 32)
    x = conv_block(x, 64, stride=2)

    # Bloque 3
    x = conv_block(x, 64)
    x = conv_block(x, 128, stride=2)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dense oculta
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    # Salida multi-label
    outputs = layers.Dense(num_classes, activation="sigmoid", name="probs")(x)

    model = keras.Model(inputs, outputs, name="pokemon_custom_cnn_v2")
    return model