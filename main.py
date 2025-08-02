from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import uuid
import json
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and data files
model = load_model('celeba_model.h5')

with open("corr_map.json") as f:
    corr_map = json.load(f)

with open("test.json") as f:
    attributes_info = json.load(f)

columns = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs",
    "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows",
    "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin",
    "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
    "Wearing_Necktie", "Young"
]

# Gender-specific feature sets
male_features = {
    '5_o_Clock_Shadow', 'Bags_Under_Eyes', 'Bald', 'Big_Nose', 'Black_Hair',
    'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
    'Gray_Hair', 'Mustache', 'Receding_Hairline', 'Sideburns',
    'Wearing_Hat', 'Wearing_Necktie', 'Young','Smiling'
}

female_features = {
    'Arched_Eyebrows', 'Bangs', 'Big_Lips', 'Blond_Hair', 'Brown_Hair',
    'Heavy_Makeup', 'High_Cheekbones', 'No_Beard', 'Oval_Face',
    'Pointy_Nose', 'Rosy_Cheeks', 'Smiling', 'Wavy_Hair',
    'Wearing_Earrings', 'Wearing_Lipstick', 'Wearing_Necklace', 'Young'
}

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

def preprocess_single_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    ext = Path(image.filename).suffix
    tmp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}{ext}")
    with open(tmp_path, "wb") as buf:
        buf.write(await image.read())

    try:
        img_array = preprocess_single_image(tmp_path)
        preds = model.predict(img_array)[0]
        binary = [1 if v > 0.5 else 0 for v in preds]
        all_attrs = dict(zip(columns, binary))

        gender = "Male" if all_attrs.get("Male", 0) == 1 else "Female"
        result = {}

        for attr, present in all_attrs.items():
            if attr == "Attractive":
                continue
            if attr in corr_map and attr in attributes_info:
                direction = corr_map[attr]
                attr_info = attributes_info[attr]

                # Updated logic: correlation sign only determines should_be_present
                should_be_present = direction > 0

                result[attr] = {
                    "present": bool(present),
                    "correlation_with_attractiveness": direction,
                    "should_be_present": should_be_present,
                    "description": attr_info.get("description", ""),
                    "importance": attr_info.get("importance", ""),
                    "how_to_improve": attr_info.get("how_to_improve", []),
                    "if_absent": attr_info.get("if_absent", ""),
                    "related_features": attr_info.get("related_features", {})
                }

        if gender == "Male":
            result = {k: v for k, v in result.items() if k in male_features}
        else:
            result = {k: v for k, v in result.items() if k in female_features}

        return {
            "gender": gender,
            "predicted_attributes": result
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)