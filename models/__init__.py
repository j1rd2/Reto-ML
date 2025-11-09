# models/__init__.py

from .pokemon_custom_cnn import build_pokemon_custom_cnn
from .pokemon_custom_cnn_v2 import build_pokemon_custom_cnn_v2
from .pokemon_efficientNet import build_pokemon_efficientnet

MODEL_REGISTRY = {
    "pokemon_custom_cnn": build_pokemon_custom_cnn,
    "pokemon_custom_cnn_v2": build_pokemon_custom_cnn_v2,
    "pokemon_efficientNet": build_pokemon_efficientnet
}
