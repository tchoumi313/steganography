# steganography.py
import torch
import cv2
import numpy as np
from cnn_model import load_cnn

# Quantization matrix for DCT
quant = np.array([[16,11,10,16,24,40,51,61],
                  [12,12,14,19,26,58,60,55],
                  [14,13,16,24,40,57,69,56],
                  [14,17,22,29,51,87,80,62],
                  [18,22,37,56,68,109,103,77],
                  [24,35,55,64,81,104,113,92],
                  [49,64,78,87,103,121,120,101],
                  [72,92,95,98,112,100,103,99]])

def calculate_jnd(dct_coeff):
    """Calculate JND based on DCT coefficient using a simple formula."""
    k = 0.9
    alpha = 2
    return k * (1 / (np.abs(dct_coeff) + 1)) ** alpha

def dct_transform(image_block):
    """Apply DCT to an 8x8 image block."""
    return cv2.dct(image_block.astype(np.float32))

def message_to_bits(message):
    """Convert a string message to a list of binary bits."""
    return [int(bit) for char in message for bit in bin(ord(char))[2:].zfill(8)]

def embed_message_in_dct(image, message_bits, cnn_model):
    """Embed the message using CNN feature maps for guidance."""
    cnn_model.eval()
    with torch.no_grad():
        image_tensor = torch.Tensor(image).unsqueeze(0).permute(0, 3, 1, 2)
        features = cnn_model(image_tensor)

    feature_map = features.squeeze().cpu().numpy()
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())

    h, w = image.shape[:2]
    blocks = []
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            block = image[y:y+8, x:x+8]
            dct_block = dct_transform(block)
            if feature_map[y // 8, x // 8] > 0.5:
                jnd_threshold = calculate_jnd(dct_block[0, 0])
                if jnd_threshold > 0.1 and message_bits:
                    dct_block[0, 0] = (dct_block[0, 0] // 2 * 2) + message_bits.pop(0)
            blocks.append(dct_block)

    stego_image = np.zeros_like(image)
    idx = 0
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            stego_image[y:y+8, x:x+8] = cv2.idct(blocks[idx])
            idx += 1
    return stego_image

def extract_message_from_dct(stego_image, message_len):
    """Extract the hidden message from the DCT coefficients."""
    h, w = stego_image.shape[:2]
    extracted_bits = []
    
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            block = stego_image[y:y+8, x:x+8]
            dct_block = dct_transform(block)
            extracted_bits.append(int(dct_block[0, 0]) % 2)
            if len(extracted_bits) >= message_len * 8:
                break

    chars = [chr(int(''.join(map(str, extracted_bits[i:i+8])), 2)) for i in range(0, len(extracted_bits), 8)]
    return ''.join(chars)

if __name__ == "__main__":
    # Load the CNN model
    cnn_model = load_cnn()

    # Embed message
    image = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
    message_bits = message_to_bits("SecretMessage")
    stego_image = embed_message_in_dct(image, message_bits, cnn_model)
    cv2.imwrite('stego_image.png', stego_image)

    # Extract message
    extracted_message = extract_message_from_dct(stego_image, len("SecretMessage"))
    print("Extracted Message:", extracted_message)
