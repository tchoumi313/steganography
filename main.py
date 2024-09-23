import matplotlib.pyplot as plt
import numpy as np

# Create a 4x4 color image matrix with red, violet, and orange colors
color_image = np.array([
    [[255, 0, 0], [128, 0, 128], [255, 165, 0], [0, 0, 0]],  # Red, Violet, Orange, Black
    [[255, 0, 0], [128, 0, 128], [255, 165, 0], [0, 0, 0]],
    [[255, 0, 0], [128, 0, 128], [255, 165, 0], [0, 0, 0]],
    [[255, 0, 0], [128, 0, 128], [255, 165, 0], [0, 0, 0]]
])

# Convert color image to grayscale
def rgb_to_grayscale(image):
    assert image.shape[2] == 3, "Input image must be an RGB image"
    gray_image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    return gray_image

grayscale_image = rgb_to_grayscale(color_image)
print("Grayscale image before embedding:\n", grayscale_image)

# DFT functions
def dft_2d(image):
    return np.fft.fft2(image)

def idft_2d(image):
    return np.fft.ifft2(image)

# Calculate JND
def calculate_jnd(gray_image, base_threshold=0.1):
    avg_luminance = np.mean(gray_image)
    jnd = base_threshold * (1 + gray_image / avg_luminance)
    return jnd

# Embed and extract message functions
def embed_message(dft_coeffs, message, jnd):
    embedded_coeffs = dft_coeffs.real.copy()
    message_bits = [int(bit) for bit in message]
    for i, bit in enumerate(message_bits):
        if bit == 1:
            embedded_coeffs.flat[i] += jnd.flat[i]
    return embedded_coeffs

def extract_message(original_dft, stego_dft, jnd, message_length):
    extracted_bits = []
    epsilon = 1e-5
    for i in range(message_length):
        diff = (stego_dft.flat[i] - original_dft.flat[i]).real
        if abs(diff) >= jnd.flat[i] - epsilon:
            extracted_bits.append(1)
        else:
            extracted_bits.append(0)
    return ''.join(map(str, extracted_bits))

# Example usage
dft_coeffs = dft_2d(grayscale_image)
print("DFT Coefficients:\n", dft_coeffs)

jnd = calculate_jnd(grayscale_image)
message = "1011010110"
embedded_dft_coeffs = embed_message(dft_coeffs, message, jnd)
print("DFT after embedding:\n", embedded_dft_coeffs)
# Convert back to grayscale after embedding
stego_image = idft_2d(embedded_dft_coeffs + 1j * np.zeros_like(embedded_dft_coeffs))

# Print the grayscale image after embedding
grayscale_after_embedding = stego_image.real.astype(np.uint8)  # No need to convert to RGB
print("Grayscale image after embedding:\n", grayscale_after_embedding)

extracted_message = extract_message(dft_coeffs, embedded_dft_coeffs, jnd, len(message))
print("Extracted message\n", extracted_message)
# Display the images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original Color Image")
plt.imshow(color_image)

plt.subplot(1, 3, 2)
plt.title("Grayscale Image")
plt.imshow(grayscale_image, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Stego Image")
plt.imshow(grayscale_after_embedding, cmap='gray')

plt.show()



