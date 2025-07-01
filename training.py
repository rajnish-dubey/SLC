# # Importing the Keras libraries and packages
# import tensorflow as tf
# import os
# import math
# import matplotlib.pyplot as plt

# # Alias imports for cleaner syntax
# keras = tf.keras
# Sequential = keras.models.Sequential
# Conv2D = keras.layers.Conv2D
# MaxPooling2D = keras.layers.MaxPooling2D
# GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
# Dense = keras.layers.Dense
# Dropout = keras.layers.Dropout
# BatchNormalization = keras.layers.BatchNormalization
# ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator

# # Image size (matching preprocessed image size)
# sz = 128

# # Set visible GPU (optional)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# # Ensure models directory exists
# os.makedirs("models", exist_ok=True)

# # Build CNN model
# classifier = Sequential()

# # Block 1
# classifier.add(Conv2D(32, (3, 3), padding='same', input_shape=(sz, sz, 1)))
# classifier.add(BatchNormalization())
# classifier.add(tf.keras.layers.ReLU())
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(Dropout(0.25))

# # Block 2
# classifier.add(Conv2D(64, (3, 3), padding='same'))
# classifier.add(BatchNormalization())
# classifier.add(tf.keras.layers.ReLU())
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(Dropout(0.25))

# # Block 3
# classifier.add(Conv2D(128, (3, 3), padding='same'))
# classifier.add(BatchNormalization())
# classifier.add(tf.keras.layers.ReLU())
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(Dropout(0.3))

# # Global average pooling instead of Flatten
# classifier.add(GlobalAveragePooling2D())

# # Dense layers
# classifier.add(Dense(128, activation='relu'))
# classifier.add(Dropout(0.5))

# # Final output layer: 36 classes (0–9 + A–Z)
# classifier.add(Dense(36, activation='softmax'))

# # Compile the model
# optimizer = keras.optimizers.Adam(learning_rate=0.0005)
# classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# # Summary of architecture
# classifier.summary()

# # --- Data preprocessing ---
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=False,
#     fill_mode='nearest'
# )

# test_datagen = ImageDataGenerator(rescale=1./255)

# training_set = train_datagen.flow_from_directory(
#     'output2/train',
#     target_size=(sz, sz),
#     batch_size=64,
#     color_mode='grayscale',
#     class_mode='categorical'
# )

# test_set = test_datagen.flow_from_directory(
#     'output2/test',
#     target_size=(sz, sz),
#     batch_size=64,
#     color_mode='grayscale',
#     class_mode='categorical'
# )

# # Save class label mapping for prediction later
# import json
# with open('models/class_indices.json', 'w') as f:
#     json.dump(training_set.class_indices, f)

# steps_per_epoch = math.ceil(training_set.samples / training_set.batch_size)
# validation_steps = math.ceil(test_set.samples / test_set.batch_size)

# # --- Callbacks ---
# early_stopping = keras.callbacks.EarlyStopping(
#     monitor='val_accuracy',
#     patience=5,
#     restore_best_weights=True
# )

# model_checkpoint = keras.callbacks.ModelCheckpoint(
#     'models/model-bw-best.keras',
#     monitor='val_accuracy',
#     save_best_only=True,
#     mode='max'
# )

# # --- Train the model ---
# history = classifier.fit(
#     training_set,
#     steps_per_epoch=steps_per_epoch,
#     epochs=20,
#     validation_data=test_set,
#     validation_steps=validation_steps,
#     callbacks=[early_stopping, model_checkpoint]
# )

# # --- Plot Accuracy and Loss ---
# plt.figure(figsize=(12, 4))

# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
# plt.title('Model Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Training Loss', marker='o')
# plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# plt.tight_layout()
# plt.savefig('training_history.png')
# plt.close()

# # --- Save the final model ---
# print('Saving final model...')
# classifier.save('models/model-bw-final.keras')
# print('Model saved successfully!')







# Import necessary libraries
import tensorflow as tf
import os
import math
import matplotlib.pyplot as plt
import json

# Aliases for cleaner code
keras = tf.keras
Sequential = keras.models.Sequential
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
BatchNormalization = keras.layers.BatchNormalization
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator

# Image dimensions (128x128)
sz = 128

# Optional GPU device selection
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Ensure 'models' folder exists
os.makedirs("models", exist_ok=True)

# Building the CNN model
classifier = Sequential()

# --- Convolutional Block 1 ---
classifier.add(Conv2D(32, (3, 3), padding='same', input_shape=(sz, sz, 1)))  # 32 filters of 3x3 on grayscale 128x128 input
classifier.add(BatchNormalization())  # Normalize activations for faster training
classifier.add(tf.keras.layers.ReLU())  # Activation function
classifier.add(MaxPooling2D(pool_size=(2, 2)))  # Downsample by 2x2
classifier.add(Dropout(0.25))  # Drop 25% of neurons to prevent overfitting

# --- Convolutional Block 2 ---
classifier.add(Conv2D(64, (3, 3), padding='same'))  # 64 filters
classifier.add(BatchNormalization())
classifier.add(tf.keras.layers.ReLU())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

# --- Convolutional Block 3 ---
classifier.add(Conv2D(128, (3, 3), padding='same'))
classifier.add(BatchNormalization())
classifier.add(tf.keras.layers.ReLU())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.3))

# --- Global Pooling ---
classifier.add(GlobalAveragePooling2D())  # Reduce each feature map to a single value

# --- Fully Connected Layer ---
classifier.add(Dense(256, activation='relu'))  # Hidden layer with 256 units
classifier.add(Dropout(0.5))

# --- Output Layer ---
classifier.add(Dense(36, activation='softmax'))  # 36 classes: A-Z + 0-9

# Compile model
optimizer = keras.optimizers.Adam(learning_rate=0.0005)
classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
classifier.summary()  # Show model architecture

# --- Data Augmentation (for training only) ---
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,  # Shearing transformation
    zoom_range=0.15,
    brightness_range=(0.6, 1.4),  # Vary brightness
    channel_shift_range=50.0,  # Slight color shift (even for grayscale it may help)
    horizontal_flip=False,  # Don't flip as hand signs can change meaning
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load training and validation datasets
training_set = train_datagen.flow_from_directory(
    'output2/train',
    target_size=(sz, sz),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    'output2/test',
    target_size=(sz, sz),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

# Save class label mapping for future use (like prediction)
with open('models/class_indices.json', 'w') as f:
    json.dump(training_set.class_indices, f)

# Calculate steps for training and validation per epoch
steps_per_epoch = math.ceil(training_set.samples / training_set.batch_size)
validation_steps = math.ceil(test_set.samples / test_set.batch_size)

# Callbacks: Early stopping and saving best model
early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
model_checkpoint = keras.callbacks.ModelCheckpoint(
    'models/model-bw-augmented.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# --- Train the model ---
history = classifier.fit(
    training_set,
    steps_per_epoch=steps_per_epoch,
    epochs=30,
    validation_data=test_set,
    validation_steps=validation_steps,
    callbacks=[early_stopping, model_checkpoint]
)

# --- Plot Accuracy and Loss ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

# --- Save the final model ---
print('Saving final model...')
classifier.save('models/model-bw-augemented.keras')
print('Model saved successfully!')
