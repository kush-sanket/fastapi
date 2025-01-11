import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

df = pd.read_csv('list_attr_celeba.csv')
image_folder = 'image_path'  # Modify this with your actual image folder path

# Construct image file paths using the image ids or other identifiers from your df
# Assuming 'id' column is used to match image names (e.g., 'image_id_1.jpg', 'image_id_2.jpg', etc.)
image_paths = [os.path.join(image_folder, f"{img_id}") for img_id in df['image_id']]

# Preprocess the images
X = np.array([img_to_array(load_img(img_path, target_size=(128, 128))) for img_path in image_paths])
print(X)


df = pd.read_csv('list_attr_celeba.csv')

# Replace -1 with 0 (binary classification: -1 -> 0, 1 -> 1)
df.iloc[:, 1:] = df.iloc[:, 1:].replace(-1, 0)

image_folder = 'image_path'  # Path to the folder containing images
df['image_path'] = df['image_id'].apply(lambda x: os.path.join(image_folder, x))
y = df.iloc[:, 1:-1].values  # Select all columns except 'image_id' and 'image_path'

from tqdm import tqdm
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

# Function to preprocess only the first 50,000 images
def preprocess_images(image_paths, target_size=(128, 128)):
    images = []
    # Wrap the loop with tqdm to display a progress bar
    for img_path in tqdm(image_paths[:50000], desc="Processing Images", unit="image"):
        img = load_img(img_path, target_size=target_size)  # Load image
        img_array = img_to_array(img) / 255.0  # Convert to array and normalize
        images.append(img_array)
    return np.array(images)

# Limit both image paths and labels to the first 50,000 records
df_limited = df.iloc[:50000]

# Prepare the image data (first 50,000 images)
X = preprocess_images(df_limited['image_path'].values)

np.save("X_preprocessed.npy", X)
y = df_limited.iloc[:, 1:].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = np.array(y_train, dtype=float)
y_val = np.array(y_val, dtype=float)

# Save them as usual
np.save("X_train.npy", X_train)
np.save("X_val.npy", X_val)
np.save("y_train.npy", y_train)
np.save("y_val.npy", y_val)

##############################################################################################
#Training the data 
X = np.load("X_preprocessed.npy")
y_train = np.load("y_train.npy")
y_val = np.load("y_val.npy")
X_train = np.load("X_train.npy")
X_val = np.load("X_val.npy")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

model = Sequential([
    Input(shape=(128, 128, 3)),  # Explicitly specify the input shape
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='sigmoid')  # Sigmoid for multi-label classification
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

del X

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,  # Adjust epochs as needed
    batch_size=32  # Adjust batch size based on system memory
)

model.save('celeba_model.h5')