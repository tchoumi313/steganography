"""import numpy as np

def rgb_to_grayscale(image):
    # Ensure the image is in the correct format (height, width, 3)
    assert image.shape[2] == 3, "Input image must be an RGB image"

    # Calculate the grayscale image
    gray_image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]

    return gray_image

# Example usage
image = np.array([
    [(1, 2, 3), (4, 5, 6)],
    [(7, 8, 9), (10, 11, 12)]
], dtype=np.float32)

gray_image = rgb_to_grayscale(image)
print(gray_image)


def grayscale_to_rgb(original_image, modified_gray_image):
    # Ensure the images are in the correct format
    assert original_image.shape[2] == 3, "Original image must be an RGB image"
    assert original_image.shape[:2] == modified_gray_image.shape, "Image dimensions must match"

    # Calculate the change in grayscale values
    delta_gray = modified_gray_image - rgb_to_grayscale(original_image)

    # Reconstruct the RGB image
    reconstructed_image = np.zeros_like(original_image)
    for i in range(original_image.shape[0]):
        for j in range(original_image.shape[1]):
            total = original_image[i, j, 0] + original_image[i, j, 1] + original_image[i, j, 2]
            reconstructed_image[i, j, 0] = original_image[i, j, 0] + delta_gray[i, j] * (original_image[i, j, 0] / total)
            reconstructed_image[i, j, 1] = original_image[i, j, 1] + delta_gray[i, j] * (original_image[i, j, 1] / total)
            reconstructed_image[i, j, 2] = original_image[i, j, 2] + delta_gray[i, j] * (original_image[i, j, 2] / total)

    return reconstructed_image

# Example usage
original_image = np.array([
    [(1, 2, 3), (4, 5, 6)],
    [(7, 8, 9), (10, 11, 12)]
], dtype=np.float32)

# Convert to grayscale
gray_image = rgb_to_grayscale(original_image)
print("Grayscale Image:\n", gray_image)

# Assume we have a modified grayscale image
modified_gray_image = np.array([
    [5.1204, 7.2049],
    [6.0599, 6.4104]
], dtype=np.float32)

# Reconstruct the RGB image
reconstructed_image = grayscale_to_rgb(original_image, modified_gray_image)
print("Reconstructed RGB Image:\n", reconstructed_image)
"""

import numpy as np


def rgb_to_grayscale(image):
    assert image.shape[2] == 3, "Input image must be an RGB image"
    gray_image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    print("Grayscale Image:\n", gray_image)
    return gray_image

def dft_2d(image):
    dft_result = np.fft.fft2(image)
    print("DFT Coefficients:\n", dft_result)
    return dft_result

def idft_2d(image):
    idft_result = np.fft.ifft2(image)
    print("IDFT Result (Stego Image):\n", idft_result)
    return idft_result

def calculate_jnd(gray_image, base_threshold=0.1):
    avg_luminance = np.mean(gray_image)
    print("avg luminance\n", avg_luminance)
    jnd = base_threshold * (1 + gray_image / avg_luminance)
    print("JND Thresholds:\n", jnd)
    return jnd

def embed_message(dft_coeffs, message, jnd):
    embedded_coeffs = dft_coeffs.real.copy()
    message_bits = [int(bit) for bit in message]
    print("Original DFT Coefficients:\n", dft_coeffs)
    for i, bit in enumerate(message_bits):
        if bit == 1:
            embedded_coeffs.flat[i] += jnd.flat[i]
            print(f"Embedding bit {bit} at index {i}, adding JND {jnd.flat[i]}")
    print("Embedded DFT Coefficients:\n", embedded_coeffs)
    return embedded_coeffs

def extract_message(original_dft, stego_dft, jnd, message_length):
    extracted_bits = []
    epsilon = 1e-5
    for i in range(message_length):
        diff = (stego_dft.flat[i] - original_dft.flat[i]).real
        print(f"Index {i}: Original DFT {original_dft.flat[i]}, Stego DFT {stego_dft.flat[i]}, Diff {diff}, JND {jnd.flat[i]}")
        if abs(diff) >= jnd.flat[i] - epsilon:
            extracted_bits.append(1)
        else:
            extracted_bits.append(0)
    extracted_message = ''.join(map(str, extracted_bits))
    print("Extracted Message:\n", extracted_message)
    return extracted_message

# Example usage
original_image = np.array([
    [(1, 2, 3), (4, 5, 6)],
    [(7, 8, 9), (10, 11, 12)]
], dtype=np.float32)

gray_image = rgb_to_grayscale(original_image)
dft_coeffs = dft_2d(gray_image)
jnd = calculate_jnd(gray_image)
message = "1011"
embedded_dft_coeffs = embed_message(dft_coeffs, message, jnd)
stego_image = idft_2d(embedded_dft_coeffs + 1j * np.zeros_like(embedded_dft_coeffs))
extracted_message = extract_message(dft_coeffs, embedded_dft_coeffs, jnd, len(message))
