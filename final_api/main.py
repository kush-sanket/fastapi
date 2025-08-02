from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import uuid
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import json

# ---- App setup ---- #
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 1) Load precomputed corr_map.json ---- #
with open("corr_map.json", "r") as f:
    corr_map = json.load(f)
# corr_map maps feature → correlation in [-1.0, 1.0]

# ---- 2) Load Keras model and attribute definitions ---- #
model = load_model("celeba_model.h5")

columns = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young"
]

male_features = {
    "5_o_Clock_Shadow", "Bags_Under_Eyes", "Bald", "Big_Nose", "Black_Hair",
    "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee",
    "Gray_Hair", "Mustache", "Receding_Hairline", "Sideburns",
    "Wearing_Hat", "Wearing_Necktie", "Young"
}

female_features = {
    "Arched_Eyebrows", "Bangs", "Big_Lips", "Blond_Hair", "Brown_Hair",
    "Heavy_Makeup", "High_Cheekbones", "No_Beard", "Oval_Face",
    "Pointy_Nose", "Rosy_Cheeks", "Smiling", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Lipstick", "Wearing_Necklace", "Young"
}

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# ---- Image preprocessing helper ---- #
def preprocess_single_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# ---- Predict endpoint ---- #
@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    # 1) Save upload to temp
    ext = Path(image.filename).suffix
    tmp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}{ext}")
    with open(tmp_path, "wb") as buf:
        buf.write(await image.read())

    try:
        # 2) Run model inference
        img_array = preprocess_single_image(tmp_path)
        preds = model.predict(img_array)[0]
        binary = [1 if v > 0.5 else 0 for v in preds]
        all_attrs = dict(zip(columns, binary))

        # 3) Filter to gender-specific features
        if all_attrs.get("Male", 0) == 1:
            filtered = {k: all_attrs[k] for k in male_features}
        else:
            filtered = {k: all_attrs[k] for k in female_features}

        # 4) Convert correlation → percentage for each feature
        corr_percent = {}
        for feat in filtered:
            corr_val = corr_map.get(feat, 0.0)
            # e.g. 0.251 → 25.1, -0.146 → -14.6
            corr_percent[feat] = round(corr_val * 100, 1)

        # 5) Return both the binary predictions and % contributions
        return {
            "predictions": filtered,
            "attractiveness_contribution_%": corr_percent
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
