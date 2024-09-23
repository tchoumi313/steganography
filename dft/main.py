import numpy as np
from PIL import Image
from scipy.fft import fft2, ifft2


def dft_embed(image_path, secret_message):
    image = Image.open(image_path).convert("L")
    data = np.array(image)
    
    # Apply DFT
    dft_matrix = fft2(data)
    
    # Embed secret message
    binary_message = ''.join(format(ord(char), '08b') for char in secret_message) + '11111111'
    for i in range(len(binary_message)):
        dft_matrix.flat[i] = (dft_matrix.flat[i].real & ~1) + int(binary_message[i]) + 1j * dft_matrix.flat[i].imag
    
    # Apply inverse DFT
    stego_image = ifft2(dft_matrix).real
    stego_image = Image.fromarray(np.clip(stego_image, 0, 255).astype(np.uint8))
    stego_image.save("stego_image_dft.png")

def dft_extract(stego_image_path):
    stego_image = Image.open(stego_image_path).convert("L")
    data = np.array(stego_image)
    
    # Apply DFT
    dft_matrix = fft2(data)
    
    binary_message = ""
    for i in range(len(dft_matrix.flat)):
        binary_message += str(int(dft_matrix.flat[i].real) & 1)
    
    message_bytes = [binary_message[i:i + 8] for i in range(0, len(binary_message), 8)]
    secret_message = ""
    
    for byte in message_bytes:
        if byte == '11111111':  # End marker
            break
        secret_message += chr(int(byte, 2))
    
    return secret_message

# Example usage
# dft_embed("input_image.png", "Secret Message")
# print(dft_extract("stego_image_dft.png"))