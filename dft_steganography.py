import numpy as np

# 4x4 image matrix (grayscale values)
image_matrix = np.array([
    [100, 105, 110, 115],
    [120, 125, 130, 135],
    [140, 145, 150, 155],
    [160, 165, 170, 175]
])

# Secret message to be embedded (binary)
secret_message = np.array([1, 1, 1, 1, 0, 0, 0, 0])

# Apply DFT to the image matrix
dft_matrix = np.fft.fft2(image_matrix)
print("DFT Matrix:\n", dft_matrix)

# Embed the secret message in the DFT coefficients
embedded_dft_matrix = dft_matrix.copy()
for i in range(len(secret_message)):
    lsb = int(embedded_dft_matrix.flat[i].real) & 1
    embedded_dft_matrix.flat[i] = (int(embedded_dft_matrix.flat[i].real) & ~1) + secret_message[i] + 1j * embedded_dft_matrix.flat[i].imag

print("Embedded DFT Matrix:\n", embedded_dft_matrix)

# Apply inverse DFT to get the stego image
stego_image_matrix = np.fft.ifft2(embedded_dft_matrix).real
print("Stego Image Matrix:\n", stego_image_matrix)

# Apply DFT to the stego image
stego_dft_matrix = np.fft.fft2(stego_image_matrix)

# Extract the secret message from the DFT coefficients
extracted_message = np.array([int(embedded_dft_matrix.flat[i].real) & 1 for i in range(len(secret_message))])
print("Extracted Secret Message:\n", extracted_message)


import numpy as np
from scipy.fftpack import dct, idct

# 4x4 image matrix (grayscale values)
image_matrix = np.array([
    [100, 105, 110, 115],
    [120, 125, 130, 135],
    [140, 145, 150, 155],
    [160, 165, 170, 175]
])

# Secret message to be embedded (binary)
secret_message = np.array([1, 0, 1, 1, 1, 1, 1, 0])

# Apply DCT to the image matrix
dct_matrix = dct(dct(image_matrix.T, norm='ortho').T, norm='ortho')
print("DCT Matrix:\n", dct_matrix)

# Embed the secret message in the DCT coefficients
embedded_dct_matrix = dct_matrix.copy()
for i in range(len(secret_message)):
    lsb = int(embedded_dct_matrix.flat[i]) & 1
    embedded_dct_matrix.flat[i] = (int(embedded_dct_matrix.flat[i]) & ~1) | secret_message[i]

print("Embedded DCT Matrix:\n", embedded_dct_matrix)

# Apply inverse DCT to get the stego image
stego_image_matrix = idct(idct(embedded_dct_matrix.T, norm='ortho').T, norm='ortho')
print("Stego Image Matrix:\n", stego_image_matrix)

# Apply DCT to the stego image
stego_dct_matrix = dct(dct(stego_image_matrix.T, norm='ortho').T, norm='ortho')

# Extract the secret message from the DCT coefficients
extracted_message = np.array([int(embedded_dct_matrix.flat[i]) & 1 for i in range(len(secret_message))])
print("Extracted Secret Message:\n", extracted_message)
