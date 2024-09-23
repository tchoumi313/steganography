'''
Author: Mason Edgar
ECE 529 - Algorithm Project
Image Steganography
'''
import struct

import bitstring
#------ External Libraries ------#
import cv2
#================================#
#---------- Source Files --------#
import data_embedding as stego
import image_preparation as img
import numpy as np
import run_stego_algorithm as src
import zigzag as zz

#================================#

# ============================================================================= #
# ============================================================================= #
# =========================== BEGIN CODE OPERATION ============================ #
# ============================================================================= #
# ============================================================================= #

stego_image     = cv2.imread(src.STEGO_IMAGE_FILEPATH, flags=cv2.IMREAD_COLOR)
stego_image_f32 = np.float32(stego_image)
stego_image_YCC = img.YCC_Image(cv2.cvtColor(stego_image_f32, cv2.COLOR_BGR2YCrCb))

# FORWARD DCT STAGE
dct_blocks = [cv2.dct(block) for block in stego_image_YCC.channels[0]]  # Only care about Luminance layer

# QUANTIZATION STAGE
dct_quants = [np.around(np.divide(item, img.JPEG_STD_LUM_QUANT_TABLE)) for item in dct_blocks]

# Sort DCT coefficients by frequency
sorted_coefficients = [zz.zigzag(block) for block in dct_quants]

# DATA EXTRACTION STAGE
recovered_data = stego.extract_encoded_data_from_DCT(sorted_coefficients)
print("recovered data\n",recovered_data, recovered_data.length)
# Check if any data was recovered
if recovered_data.length < 32:
    raise ValueError("Not enough data was recovered from the stego image.")

# Determine length of secret message
data_len = int(recovered_data.read('uint:32') / 8)

# Extract secret message from DCT coefficients
extracted_data = bytes()
for _ in range(data_len):
    extracted_data += struct.pack('>B', recovered_data.read('uint:8'))

# Print secret message back to the user
print(extracted_data.decode('ascii'))