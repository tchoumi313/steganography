import numpy as np
import pywt
from PIL import Image


def dwt_embed(image_path, secret_message):
    image = Image.open(image_path).convert("L")
    data = np.array(image)
    
    # Apply DWT
    coeffs = pywt.dwt2(data, 'haar')
    cA, (cH, cV, cD) = coeffs
    
    # Embed secret message
    binary_message = ''.join(format(ord(char), '08b') for char in secret_message) + '11111111'
    for i in range(len(binary_message)):
        cH.flat[i] = (cH.flat[i] & ~1) | int(binary_message[i])
    
    # Apply inverse DWT
    coeffs = cA, (cH, cV, cD)
    stego_image = pywt.idwt2(coeffs, 'haar')
    stego_image = Image.fromarray(np.clip(stego_image, 0, 255).astype(np.uint8))
    stego_image.save("stego_image_dwt.png")

def dwt_extract(stego_image_path):
    stego_image = Image.open(stego_image_path).convert("L")
    data = np.array(stego_image)
    
    # Apply DWT
    coeffs = pywt.dwt2(data, 'haar')
    cA, (cH, cV, cD) = coeffs
    
    binary_message = ""
    for i in range(len(cH.flat)):
        binary_message += str(int(cH.flat[i]) & 1)
    
    message_bytes = [binary_message[i:i + 8] for i in range(0, len(binary_message), 8)]
    secret_message = ""
    
    for byte in message_bytes:
        if byte == '11111111':  # End marker
            break
        secret_message += chr(int(byte, 2))
    
    return secret_message

# Example usage
# dwt_embed("input_image.png", "Secret Message")
# print(dwt_extract("stego_image_dwt.png"))