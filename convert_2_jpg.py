import numpy as np
import glob
import os
import cv2
from collections import Counter
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Prepare Data
dir_path = 'Images'
categories = os.listdir(dir_path)

# Determine the most common image shape in your dataset
all_images = []
for category in categories:
    for filename in glob.glob(os.path.join(dir_path, category, "*.jpg")):
        all_images.append(cv2.imread(filename, cv2.IMREAD_COLOR))

shapes = [img.shape for img in all_images]
most_common_shape = Counter(shapes).most_common(1)[0][0]

# ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

train_generator = datagen.flow_from_directory(
    dir_path,
    target_size=(most_common_shape[0], most_common_shape[1]),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dir_path,
    target_size=(most_common_shape[0], most_common_shape[1]),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Ensuring no shuffling for consistent order
)

# Model Architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(most_common_shape[0], most_common_shape[1], 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(len(categories), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_loss')

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint]
)

# Save training logs to file
with open('training_logs.txt', 'w') as log_file:
    for epoch, (loss, acc, val_loss, val_acc) in enumerate(zip(history.history['loss'], history.history['accuracy'], history.history['val_loss'], history.history['val_accuracy'])):
        log_file.write(f"Epoch {epoch+1} - loss: {loss:.4f}, accuracy: {acc:.4f}, val_loss: {val_loss:.4f}, val_accuracy: {val_acc:.4f}\n")

# Evaluation
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {accuracy*100:.2f}%")

# Predict on the validation set for the classification report
y_true_classes = validation_generator.classes
y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

report = classification_report(y_true_classes, y_pred_classes, target_names=categories)
print(report)

# Save the classification report to a file
with open('classification_report.txt', 'w') as report_file:
    report_file.write(report)
