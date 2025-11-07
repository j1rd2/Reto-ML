from data import build_dataframe, make_dataset
from config import TYPES
from tensorflow import keras

IMG_SIZE = 256
BATCH_SIZE = 25

# 1. Cargar test
Xte, Yte = build_dataframe("data/dataset/labels_test.csv", "data/dataset/test")
test_ds = make_dataset(
    Xte["path"].values,
    Yte.values,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# 2. Cargar mejor modelo
model = keras.models.load_model("export/pokemon_custom_cnn_v2.h5")

# 3. Evaluar
result = model.evaluate(test_ds, return_dict=True)
print(result)