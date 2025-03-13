import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# Verify dataset path exists
TRAIN_DIR = os.path.abspath("../dataset/train")
TEST_DIR = os.path.abspath("../dataset/test")

if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
    raise FileNotFoundError("Dataset directories not found. Make sure '../dataset/train' and '../dataset/test' exist.")

# Image size and batch size
IMG_SIZE = (64, 64)
BATCH_SIZE = 32

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, zoom_range=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")
test_generator = test_datagen.flow_from_directory(TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(len(train_generator.class_indices), activation="softmax")
])

# Compile model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(train_generator, validation_data=test_generator, epochs=10)

# Save model
model.save("model/sign_model.h5")

# Save labels
with open("labels.json", "w") as f:
    json.dump(train_generator.class_indices, f)

print("✅ Model and labels saved successfully!")
