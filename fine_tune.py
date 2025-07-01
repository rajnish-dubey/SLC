import tensorflow as tf
import math
from pathlib import Path

# Load the trained model
model_path = Path('models/model-bw-best.keras')  # Use the best model from previous training
if not model_path.exists():
    model_path = Path('models/model-bw-final.keras')  # Fallback to final model

print("Loading model from:", model_path)
classifier = tf.keras.models.load_model(str(model_path))

# Freeze the first two convolutional blocks
for layer in classifier.layers[:12]:  # First two conv blocks (6 layers each)
    layer.trainable = False

# Compile with a lower learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)  # 10x smaller learning rate
classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Data generators with more aggressive augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,  # Increased rotation
    width_shift_range=0.2,  # Increased shift
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=False,
    fill_mode='nearest'
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Load the data
sz = 128  # Same size as original training
training_set = train_datagen.flow_from_directory(
    'output/train',
    target_size=(sz, sz),
    batch_size=16,  # Smaller batch size for fine-tuning
    color_mode='grayscale',
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    'output/test',
    target_size=(sz, sz),
    batch_size=16,
    color_mode='grayscale',
    class_mode='categorical'
)

# Calculate steps
steps_per_epoch = math.ceil(training_set.samples / training_set.batch_size)
validation_steps = math.ceil(test_set.samples / test_set.batch_size)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=7,  # More patience for fine-tuning
    restore_best_weights=True
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'models/model-bw-finetuned.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# Fine-tune the model
print("\nFine-tuning the model...")
history = classifier.fit(
    training_set,
    steps_per_epoch=steps_per_epoch,
    epochs=50,  # More epochs for fine-tuning
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
plt.title('Fine-tuning Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Fine-tuning Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('finetuning_history.png')
plt.close()

print("\nFine-tuning complete! Best model saved as 'model-bw-finetuned.keras'") 