#!/usr/local/bin/python
from __future__ import print_function

import itertools
import cv2
import numpy as np

# Quantization matrix for JPEG compression
quant = np.array([[16,11,10,16,24,40,51,61],
                   [12,12,14,19,26,58,60,55],
                   [14,13,16,24,40,57,69,56],
                   [14,17,22,29,51,87,80,62],
                   [18,22,37,56,68,109,103,77],
                   [24,35,55,64,81,104,113,92],
                   [49,64,78,87,103,121,120,101],
                   [72,92,95,98,112,100,103,99]])

class DCT():
    def __init__(self, imPath):
        self.imPath = imPath  # Path to the input image
        self.message = None  # Message to be encoded
        self.bitMess = None  # Bit representation of the message
        self.oriCol = 0  # Original image column size
        self.oriRow = 0  # Original image row size
        self.numBits = 0  # Number of bits in the message
        print("my DCT")
    
    def calculate_jnd(self, dct_coeff):
        """Calculate JND based on DCT coefficient using a simple formula."""
        k = 0.9  # Constant that can be adjusted for sensitivity
        alpha = 2  # Exponent that can be adjusted
        return k * (1 / (np.abs(dct_coeff) + 1)) ** alpha  # JND calculation

    def DCTEn(self, secret, outIm):
        """Encode a secret message into the image using DCT."""
        img = self.loadImage()  # Load the image
        if img is None:
            print("Error: File not found!")
            return

        self.message = str(len(secret)) + '*' + secret  # Prepare the message with its length
        self.bitMess = self.toBits()  # Convert message to bits
        
        row, col = img.shape[:2]  # Get image dimensions
        self.oriRow, self.oriCol = row, col  

        # Check if the image can hold the message
        if (col / 8) * (row / 8) < len(secret):
            print("Error: Message too large to encode in image")
            return      
        #print(f"length: {len(secret)}, bitlength: {len(self.bitMess)} , message: {self.message} \n size {(col / 8) * (row / 8)}" )
        # Pad the image if dimensions are not divisible by 8
        if row % 8 != 0 or col % 8 != 0:
            img = self.addPadd(img, row, col)
        
        row, col = img.shape[:2]  # Update dimensions after padding
        bImg, gImg, rImg = cv2.split(img)  # Split the image into RGB channels
        
        # Apply median blur before DCT encoding
        bImg = cv2.medianBlur(bImg, 9)  # Apply median blur with kernel size of 3
        #bImg = cv2.GaussianBlur(bImg, (9, 9), 0)  # Apply Gaussian blur to reduce artifacts
        #bImg = cv2.fastNlMeansDenoising(bImg, None, 30, 9, 27)  # Denoising step
        #bImg = cv2.bilateralFilter(bImg, 9, 75, 75)
        
        bImg = np.float32(bImg)  # Convert blue channel to float32 for DCT processing

        # Create 8x8 blocks from the blue channel
        imgBlocks = [np.round(bImg[j:j + 8, i:i + 8] - 128) for (j, i) in itertools.product(range(0, row, 8), range(0, col, 8))]
        dctBlocks = [np.round(cv2.dct(img_Block)) for img_Block in imgBlocks]  # Apply DCT to each block
        quantizedDCT = [np.round(dct_Block / quant) for dct_Block in dctBlocks]  # Quantize DCT coefficients

        messIndex = 0  # Index for the message bits
        letterIndex = 0  # Index for the current bit in the byte

        # Embed message bits into the DCT coefficients
        for quantizedBlock in quantizedDCT:
            DC = quantizedBlock[0][0]  # DC coefficient
            jnd = self.calculate_jnd(DC)  # Calculate JND for the DC coefficient
            
            #print(jnd)
            DC = np.uint8(DC)  # Convert to uint8 for bit manipulation
            DC = np.unpackbits(DC)  # Unpack bits for modification
            # Modify the least significant bit if JND allows it
            DC[7] = self.bitMess[messIndex][letterIndex] if jnd > 0 else DC[7]  
            DC = np.packbits(DC)  # Pack bits back into a byte
            DC = np.float32(DC) - 255  # Convert back to float32 for DCT processing
            quantizedBlock[0][0] = DC  # Update the DC coefficient

            letterIndex += 1  # Move to the next bit
            if letterIndex == 8:  # If a byte is complete
                letterIndex = 0  # Reset bit index
                messIndex += 1  # Move to the next byte
                if messIndex == len(self.message):  # If the entire message is encoded
                    break

        # Prepare the encoded image from quantized DCT blocks
        sImgBlocks = [quantizedBlock * quant + 128 for quantizedBlock in quantizedDCT]
        sImg = []

        # Reconstruct the image from the encoded blocks
        for chunkRowBlocks in self.chunks(sImgBlocks, col // 8):
            for rowBlockNum in range(8):
                for block in chunkRowBlocks:
                    sImg.extend(block[rowBlockNum])
        sImg = np.array(sImg).reshape(row, col)  # Reshape to original dimensions
        sImg = np.uint8(sImg)  # Convert to uint8
        sImg = cv2.merge((sImg, gImg, rImg))  # Merge back the RGB channels
        print(f"Message Length: {len(secret)}, Bit Length: {len(self.bitMess)}")
        print(f"Capacity: {(col / 8) * (row / 8)}")

        cv2.imwrite(outIm, sImg)  # Save the encoded image
        return sImg
    def DCTDe(self):
        """Decode the hidden message from the image using DCT."""
        img = cv2.imread(self.imPath, cv2.IMREAD_UNCHANGED)  # Load the encoded image

        row, col = img.shape[:2]  # Get image dimensions
        messSize = None  # Initialize message size
        messageBits = []  # List to store extracted bits
        buff = 0  # Buffer for constructing characters

        bImg, gImg, rImg = cv2.split(img)  # Split the image into RGB channels
        bImg = np.float32(bImg)  # Convert blue channel to float32

        # Create 8x8 blocks from the blue channel
        imgBlocks = [bImg[j:j + 8, i:i + 8] - 128 for (j, i) in itertools.product(range(0, row, 8), range(0, col, 8))]
        quantizedDCT = [img_Block / quant for img_Block in imgBlocks]  # Dequantize DCT coefficients

        i = 0  # Bit index for the message
        for quantizedBlock in quantizedDCT:
            DC = quantizedBlock[0][0]  # DC coefficient
            DC = np.uint8(DC)  # Convert to uint8
            DC = np.unpackbits(DC)  # Unpack bits for reading
            # Read the least significant bit to extract the message
            if DC[7] == 1:
                buff += (0 & 1) << (7 - i)
            elif DC[7] == 0:
                buff += (1 & 1) << (7 - i)
            i += 1  # Move to the next bit
            if i == 8:  # If a byte is complete
                messageBits.append(chr(buff))  # Convert buffer to character
                buff = 0  # Reset buffer
                i = 0  # Reset bit index

                # Check for message size delimiter
                if messageBits[-1] == '*' and messSize is None:
                    try:
                        messSize = int(''.join(messageBits[:-1]))  # Extract message size
                    except:
                        pass
            # Check if the entire message has been extracted
            if len(messageBits) - len(str(messSize)) - 1 == messSize:
                return ''.join(messageBits)[len(str(messSize)) + 1:]  # Return the hidden message
            
                
        print(f"Extracted Bits: {len(messageBits)}, Expected: {messSize}")
        return ''  # Return empty if no message found
    
    def chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        m = int(n)
        for i in range(0, len(l), m):
            yield l[i:i + m]
    
    def loadImage(self):
        """Load an image from the specified path."""
        img = cv2.imread(self.imPath, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None  
        return img

    def addPadd(self, img, row, col):
        img = cv2.resize(img, (col + (8 - col % 8), row + (8 - row % 8)))    
        return img
    
    def toBits(self):
        bits = []
        for char in self.message:
            binval = bin(ord(char))[2:].rjust(8, '0')
            bits.append(binval)
        self.numBits = bin(len(bits))[2:].rjust(8, '0')
        return bits