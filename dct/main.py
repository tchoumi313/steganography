import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct


def dct_embed(image_path, secret_message):
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    data = np.array(image)
    
    # Apply DCT
    dct_matrix = dct(dct(data.T, norm='ortho').T, norm='ortho')
    
    # Convert DCT matrix to integer type for bitwise operations
    dct_matrix = np.round(dct_matrix).astype(int)
    
    # Embed secret message
    binary_message = ''.join(format(ord(char), '08b') for char in secret_message) + '11111111'
    for i in range(len(binary_message)):
        dct_matrix.flat[i] = (dct_matrix.flat[i] & ~1) | int(binary_message[i])
    
    # Apply inverse DCT
    stego_image = idct(idct(dct_matrix.T, norm='ortho').T, norm='ortho')
    stego_image = Image.fromarray(np.clip(stego_image, 0, 255).astype(np.uint8))
    stego_image.save("../output/dct/stego_image_dct.png")

def dct_extract(stego_image_path):
    stego_image = Image.open(stego_image_path).convert("L")
    data = np.array(stego_image)
    
    # Apply DCT
    dct_matrix = dct(dct(data.T, norm='ortho').T, norm='ortho')
    
    binary_message = ""
    for i in range(len(dct_matrix.flat)):
        binary_message += str(int(dct_matrix.flat[i]) & 1)
    
    message_bytes = [binary_message[i:i + 8] for i in range(0, len(binary_message), 8)]
    secret_message = ""
    
    for byte in message_bytes:
        if byte == '11111111':  # End marker
            break
        secret_message += chr(int(byte, 2))
    
    return secret_message

# Example usage
dct_embed("../input/example.png", "Secret Message")
print(dct_extract("../output/dct/stego_image_dct.png"))