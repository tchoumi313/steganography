import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Enhanced CNN with residual connections and batch normalization
def build_enhanced_cnn():
    inputs = layers.Input(shape=(64, 64, 3))

    # Block 1
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Residual Block 1
    shortcut = x
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Adjust shortcut to match the shape
    shortcut = layers.Conv2D(128, (1, 1), padding='same')(shortcut)  # 1x1 convolution to match dimensions
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Residual Block 2
    shortcut = x
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Adjust shortcut to match the shape
    shortcut = layers.Conv2D(256, (1, 1), padding='same')(shortcut)  # 1x1 convolution to match dimensions
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Global average pooling for efficient feature extraction
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Dropout for regularization
    outputs = layers.Dense(1, activation='sigmoid')(x)  # Use sigmoid for binary classification

    return models.Model(inputs, outputs)

# Load and preprocess the dataset
data_dir = '../data/train'  # Path to the dataset
batch_size = 32
img_height = 64
img_width = 64

# Use ImageDataGenerator for loading and augmenting images
train_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'  # Change to 'binary' if you have only one class
)

# Build and compile the model
cnn_model = build_enhanced_cnn()
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Use binary_crossentropy for binary classification

# Define the model checkpoint callback
checkpoint = ModelCheckpoint(
    'cnn_model.keras',  # Path to save the model
    monitor='accuracy',  # Metric to monitor
    save_best_only=False,  # Save every epoch
    mode='max',  # Save the model with the maximum accuracy
    verbose=1  # Print messages when saving
)

# Train the model
cnn_model.fit(train_generator, epochs=10, callbacks=[checkpoint])  # Adjust the number of epochs as needed
