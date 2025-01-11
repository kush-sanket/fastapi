from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import uuid
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or list specific origins like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods like GET, POST, etc.
    allow_headers=["*"],  # Allow all headers
)

# Load the pre-trained model
model = load_model('celeba_model.h5')
columns = [
"5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs",
"Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows",
"Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
"Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin",
"Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
"Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
"Wearing_Necktie", "Young"]

# Define temporary directory
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Preprocessing function
def preprocess_single_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)  # Load and resize
    img_array = img_to_array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        # Generate a unique filename
        file_extension = Path(image.filename).suffix
        print(file_extension)
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        
        file_path = os.path.join(TEMP_DIR, unique_filename)
        print(file_path)
        
        # Save the file to the temp directory
        with open(file_path, "wb") as buffer:
            buffer.write(await image.read())
        
        # Preprocess the image
        img_array = preprocess_single_image(file_path)
        
        # Predict
        predictions = model.predict(img_array).tolist()[0]  # Convert to a list for JSON serialization
        
        predictions = [1 if value > 0.5 else 0 for value in predictions]
        
        attribute_values = dict(zip(columns, predictions))
     
        return {"predictions": attribute_values}
    
    finally:
        # Clean up the file after processing
        if os.path.exists(file_path):
            os.remove(file_path)
