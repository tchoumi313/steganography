import numpy as np
from PIL import Image


def calculate_jnd(gray_image, base_threshold=0.1):
    avg_luminance = np.mean(gray_image)
    jnd = base_threshold * (1 + gray_image / avg_luminance)
    return jnd

def jnd_embed(image_path, secret_message):
    image = Image.open(image_path).convert("L")
    data = np.array(image)
    
    jnd = calculate_jnd(data)
    binary_message = ''.join(format(ord(char), '08b') for char in secret_message) + '11111111'
    
    for i in range(len(binary_message)):
        if binary_message[i] == '1':
            data.flat[i] += jnd.flat[i]
    
    stego_image = Image.fromarray(np.clip(data, 0, 255).astype(np.uint8))
    stego_image.save("stego_image_jnd.png")

def jnd_extract(stego_image_path, original_image_path):
    stego_image = Image.open(stego_image_path).convert("L")
    original_image = Image.open(original_image_path).convert("L")
    
    stego_data = np.array(stego_image)
    original_data = np.array(original_image)
    
    binary_message = ""
    for i in range(len(stego_data.flat)):
        if abs(stego_data.flat[i] - original_data.flat[i]) >= calculate_jnd(original_data)[i]:
            binary_message += '1'
        else:
            binary_message += '0'
    
    message_bytes = [binary_message[i:i + 8] for i in range(0, len(binary_message), 8)]
    secret_message = ""
    
    for byte in message_bytes:
        if byte == '11111111':  # End marker
            break
        secret_message += chr(int(byte, 2))
    
    return secret_message

# Example usage
# jnd_embed("input_image.png", "Secret Message")
# print(jnd_extract("stego_image_jnd.png", "input_image.png"))