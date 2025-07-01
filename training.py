# Importing the Keras libraries and packages
import tensorflow as tf
import os
import math

# Alternative import style to avoid warnings
keras = tf.keras
Sequential = keras.models.Sequential
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
Flatten = keras.layers.Flatten
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
BatchNormalization = keras.layers.BatchNormalization
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator

# Set visible CUDA device if needed (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Set image size
sz = 128

# Step 1 - Building the CNN
classifier = Sequential()

# First convolution block - increased filters but simplified structure
classifier.add(Conv2D(64, (3, 3), padding='same', input_shape=(sz, sz, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

# Second convolution block
classifier.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

# Third convolution block
classifier.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

# Flattening the layers
classifier.add(Flatten())

# Simplified dense layers
classifier.add(Dense(units=256, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=35, activation='softmax'))  # 35 classes (A-Z and 0-9)

# Compile with slightly higher learning rate for faster convergence
optimizer = keras.optimizers.Adam(learning_rate=0.0005)
classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Summary
classifier.summary()

# Step 2 - Preparing the train/test data and training the model
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,  # Don't flip as it might change the meaning of signs
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'output/train',
    target_size=(sz, sz),
    batch_size=64,  # Increased batch size for faster training
    color_mode='grayscale',
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    'output/test',
    target_size=(sz, sz),
    batch_size=64,  # Increased batch size
    color_mode='grayscale',
    class_mode='categorical'
)

# Calculate steps per epoch based on dataset size
steps_per_epoch = math.ceil(training_set.samples / training_set.batch_size)
validation_steps = math.ceil(test_set.samples / test_set.batch_size)

# Add early stopping and model checkpoint callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    'models/model-bw-best.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# Training the model
history = classifier.fit(
    training_set,
    steps_per_epoch=steps_per_epoch,
    epochs=20,  # Reduced epochs
    validation_data=test_set,
    validation_steps=validation_steps,
    callbacks=[early_stopping, model_checkpoint]
)

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()


# Save the final model
print('Saving model...')
classifier.save('models/model-bw-final.keras')
print('Model saved successfully!')