import matplotlib.pyplot as plt
import numpy as np

# Create a 4x4 color image matrix with red, violet, and orange colors
color_image = np.array([
    [[255, 0, 0], [128, 0, 128], [255, 165, 0], [0, 0, 0]],  # Red, Violet, Orange, Black
    [[255, 0, 0], [128, 0, 128], [255, 165, 0], [0, 0, 0]],
    [[255, 0, 0], [128, 0, 128], [255, 165, 0], [0, 0, 0]],
    [[255, 0, 0], [128, 0, 128], [255, 165, 0], [0, 0, 0]]
])

# Step 1: Convert Color Image to Grayscale
def rgb_to_grayscale(image):
    assert image.shape[2] == 3, "Input image must be an RGB image"
    gray_image = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    return gray_image

grayscale_image = rgb_to_grayscale(color_image)
print("Grayscale image before embedding:\n", grayscale_image)

# Step 2: Apply Discrete Fourier Transform (DFT)
def dft_2d(image):
    return np.fft.fft2(image)

dft_coeffs = dft_2d(grayscale_image)
print("DFT Coefficients:\n", dft_coeffs)

# Step 3: Calculate JND Luminance/Texture Based Threshold
def calculate_jnd(gray_image, base_threshold=0.1):
    avg_luminance = np.mean(gray_image)
    jnd = base_threshold * (1 + gray_image / avg_luminance)
    return jnd

jnd = calculate_jnd(grayscale_image)

# Step 4: Embed Message Using JND Matrix Threshold
def embed_message(dft_coeffs, message, jnd):
    embedded_coeffs = dft_coeffs.real.copy()
    message_bits = [int(bit) for bit in message]
    for i, bit in enumerate(message_bits):
        if bit == 1:
            embedded_coeffs.flat[i] += jnd.flat[i]
    return embedded_coeffs

message = "1011"
embedded_dft_coeffs = embed_message(dft_coeffs, message, jnd)

# Step 5: Apply Inverse DFT to Obtain Modified Grayscale Image
def idft_2d(image):
    return np.fft.ifft2(image)

stego_image = idft_2d(embedded_dft_coeffs + 1j * np.zeros_like(embedded_dft_coeffs))

# Step 6: Convert Modified Grayscale Image Back to Color
def convert_to_color(original_color, modified_gray):
    color_after_embedding = np.zeros_like(original_color)
    for i in range(3):  # For each channel
        color_after_embedding[:, :, i] = (original_color[:, :, i] / np.max(original_color) * modified_gray).clip(0, 255)
    return color_after_embedding.astype(np.uint8)

color_after_embedding = convert_to_color(color_image, stego_image.real)

# Print the grayscale image after embedding
print("Grayscale image after embedding:\n", stego_image.real.astype(np.uint8))

# Display the images
plt.figure(figsize=(16, 8))

plt.subplot(2, 2, 1)
plt.title("Original Color Image")
plt.imshow(color_image)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Grayscale Image Before Embedding")
plt.imshow(grayscale_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("Grayscale Image After Embedding")
plt.imshow(stego_image.real.astype(np.uint8), cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("Color Image After Embedding")
plt.imshow(color_after_embedding)
plt.axis('off')

plt.tight_layout()
plt.show()