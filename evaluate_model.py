import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# Constants
IMG_SIZE = 64
EPOCHS = 15
BATCH_SIZE = 32

# Loading data
base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, 'Images')

data = []
labels = []

for label, currency_folder in enumerate(os.listdir(image_dir)):
    path = os.path.join(image_dir, currency_folder)
    for img in os.listdir(path):
        try:
            image = cv2.imread(os.path.join(path, img))
            resized_image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            data.append(resized_image)
            labels.append(label)
        except Exception as e:
            print(f"Error reading file {img} : {e}")

data = np.array(data, dtype=np.float32) / 255.0
labels = np.array(labels)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# One-hot encoding the labels
y_train = to_categorical(y_train, num_classes=7)
y_val = to_categorical(y_val, num_classes=7)

# Building the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Train the model
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])

# Optionally, you can save the entire model after training
# model.save("currency_classifier.h5")
