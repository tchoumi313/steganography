import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load and build the CNN model
def build_cnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.summary()
    return model

# Feature extraction using CNN
def extract_features(cnn_model, image):
    img_resized = cv2.resize(image, (64, 64))
    img_resized = img_resized.astype('float32') / 255.0
    features = cnn_model.predict(np.expand_dims(img_resized, axis=0))
    print(features[0])
    return features[0]

# DCT on image blocks
def apply_dct(block):
    # Ensure the block is a 2D array and convert to float32
    if block.ndim == 2 and block.shape == (8, 8):
        return cv2.dct(np.float32(block))
    else:
        raise ValueError("Input block must be a 2D array of shape (8, 8)")

# Inverse DCT on image blocks
def inverse_dct(block):
    return cv2.idct(block)

# Embed secret data into DCT coefficients
def embed_secret_data(dct_coefficients, secret_bits):
    idx = 0
    for i in range(dct_coefficients.shape[0]):
        for j in range(dct_coefficients.shape[1]):
            if idx < len(secret_bits):
                dct_coefficients[i, j] = dct_coefficients[i, j] + (secret_bits[idx] * 0.01)
                idx += 1
    return dct_coefficients

# Extract secret data from DCT coefficients
def extract_secret_data(dct_coefficients):
    extracted_bits = []
    for i in range(dct_coefficients.shape[0]):
        for j in range(dct_coefficients.shape[1]):
            extracted_bits.append(1 if dct_coefficients[i, j] % 2 != 0 else 0)
    return extracted_bits

# Function to embed data in image using CNN-DCT steganography
def embed_data(cnn_model, cover_image, secret_message):
    # Extract features from the cover image
    encoded_features = extract_features(cnn_model, cover_image)

    # Check the shape of encoded_features
    print(f"Encoded features shape: {encoded_features.shape}")

    # Ensure the features can be divided into 8x8 blocks
    height, width = encoded_features.shape
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError("Encoded features must be divisible by 8 in both dimensions.")

    dct_coefficients = []
    
    # Iterate over the features to create 8x8 blocks
    for j in range(0, height, 8):
        for i in range(0, width, 8):
            block = encoded_features[j:j + 8, i:i + 8]
            dct_block = apply_dct(block)  # Apply DCT to the 8x8 block
            dct_coefficients.append(dct_block)

    # Continue with the embedding process...

    # Convert secret message to bits
    secret_bits = ''.join(format(ord(c), '08b') for c in secret_message)
    secret_bits = [int(bit) for bit in secret_bits]
    # Embed secret data in DCT coefficients
    stego_dct_coefficients = embed_secret_data(dct_coefficients, secret_bits)
    # Get inverse DCT
    modified_blocks = inverse_dct(stego_dct_coefficients)
    return modified_blocks

# Function to extract data from image using CNN-DCT steganography
def extract_data(cnn_model, stego_image):
    # Extract features using CNN
    encoded_features = extract_features(cnn_model, stego_image)
    # Apply DCT to image blocks
    dct_coefficients = apply_dct(encoded_features)
    # Extract secret data from high-frequency DCT coefficients
    extracted_secret_data = extract_secret_data(dct_coefficients)
    return ''.join(chr(int(''.join(map(str, extracted_secret_data[i:i+8])), 2)) for i in range(0, len(extracted_secret_data), 8))

# Example usage
if __name__ == "__main__":
    # Load a sample image and secret message
    cover_image = cv2.imread('lenna.png')
    secret_message = "This is a secret!"

    # Initialize the CNN model
    cnn_model = build_cnn_model()

    # Embed the secret message
    stego_image = embed_data(cnn_model, cover_image, secret_message)

    # Extract the secret message from the stego image
    recovered_message = extract_data(cnn_model, stego_image)

    print(f"Recovered message: {recovered_message}")
