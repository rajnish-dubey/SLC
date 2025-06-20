# Importing the Keras libraries and packages
import tensorflow as tf
import os

# Alternative import style to avoid warnings
keras = tf.keras
Sequential = keras.models.Sequential
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
Flatten = keras.layers.Flatten
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator

# Set visible CUDA device if needed (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Set image size
sz = 128

# Step 1 - Building the CNN
classifier = Sequential()

# First convolution layer and pooling
classifier.add(Conv2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolution layer and pooling
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())

# Fully connected layers
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=96, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=35, activation='softmax'))  # Updated to match 35 classes found in your dataset

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary
classifier.summary()

# Step 2 - Preparing the train/test data and training the model
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'output/train',
    target_size=(sz, sz),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    'output/test',
    target_size=(sz, sz),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical'
)

# Training the model
classifier.fit(
    training_set,
    steps_per_epoch=1050,  # 33600 training images / batch_size=32
    epochs=10,
    validation_data=test_set,
    validation_steps=262    # 8400 test images / batch_size=32
)

# Saving the model - Multiple options provided
print('Saving model...')

# Option 1: Save model architecture as JSON (as you were doing)
model_json = classifier.to_json()
with open("models/model-bw.json", "w") as json_file:
    json_file.write(model_json)
print('Model architecture saved as JSON')

# Option 2: Save weights with correct filename format (FIXED)
classifier.save_weights('models/model-bw.weights.h5')
print('Weights saved as model-bw.weights.h5')

# Option 3: Save entire model in Keras format (RECOMMENDED)
classifier.save('model-bw.keras')
print('Complete model saved as model-bw.keras')

# Option 4: Save entire model in legacy H5 format (alternative)
# classifier.save('model-bw-complete.h5')
# print('Complete model saved as model-bw-complete.h5')

print('All model files saved successfully!')