#!/usr/local/bin/python
from __future__ import print_function

import cv2
import numpy as np
from pywt import dwt2, idwt2


class DWT:
    def __init__(self, imPath):
        self.imPath = imPath
        self.message = None
        self.bitMess = None
        self.oriCol = 0
        self.oriRow = 0
        self.numBits = 0

    def DWTEn(self, secret, outIm):
        # Load image for processing
        img = self.loadImage()
        if img is None:
            print("Error: File not found!")
            return

        self.message = str(len(secret)) + '*' + secret  # Keep the '*' for separation
        self.bitMess = self.toBits()

        # Get size of image in pixels
        row, col = img.shape[:2]
        self.oriRow, self.oriCol = row, col

        # Check if the message fits in the image
        if ((col // 8) * (row // 8) < len(self.message)):
            print("Error: Message too large to encode in image")
            return

        # Ensure dimensions are divisible by 8
        if row % 8 != 0 or col % 8 != 0:
            img = self.addPadd(img)

        row, col = img.shape[:2]

        # Split image into RGB channels
        bImg, gImg, rImg = cv2.split(img)

        # Message to be hidden in blue channel, converted to float32 for DWT function
        bImg = np.float32(bImg)

        # Apply 2D DWT
        coeffs = dwt2(bImg, 'haar')
        cA, (cH, cV, cD) = coeffs

        # Embed message in approximation coefficients cA
        messIndex = 0
        letterIndex = 0
        for i in range(cA.shape[0]):
            for j in range(cA.shape[1]):
                if messIndex < len(self.message):
                    bit = self.bitMess[messIndex][letterIndex]
                    if bit == '1':
                        cA[i, j] = cA[i, j] + 0.01  # Slight increment to avoid visible changes
                    elif bit == '0':
                        cA[i, j] = cA[i, j] - 0.01  # Slight decrement
                    letterIndex += 1
                    if letterIndex == 8:
                        letterIndex = 0
                        messIndex += 1
                        if messIndex == len(self.message):
                            break

        # Apply inverse DWT
        coeffs = (cA, (cH, cV, cD))
        sImg = idwt2(coeffs, 'haar')

        # Normalize and convert back to uint8
        sImg = np.clip(sImg, 0, 255)  # Clip values to valid range
        sImg = np.uint8(sImg)

        # Merge back the channels
        sImg = cv2.merge((sImg, gImg, rImg))
        cv2.imwrite(outIm, sImg)
        print(f"Message encoded: {self.message}")  # Debugging line
        return sImg

    def DWTDe(self):
        img = cv2.imread(self.imPath, cv2.IMREAD_UNCHANGED)

        if img is None:
            print("Error: File not found!")
            return

        # Split image into RGB channels
        bImg, gImg, rImg = cv2.split(img)

        # Convert to type float32 for DWT function
        bImg = np.float32(bImg)

        # Apply 2D DWT
        coeffs = dwt2(bImg, 'haar')
        cA, (cH, cV, cD) = coeffs

        # Extract message from approximation coefficients cA
        messageBits = []
        buff = 0
        i = 0
        messSize = None  # Initialize messSize to None
        for row in cA:
            for value in row:
                if value > 0:
                    buff += (0 & 1) << (7 - i)  # Slight positive modification treated as '0'
                else:
                    buff += (1 & 1) << (7 - i)  # Slight negative modification treated as '1'
                i += 1
                if i == 8:
                    messageBits.append(chr(buff))
                    buff = 0
                    i = 0
                    if messageBits[-1] == '*' and messSize is None:
                        try:
                            messSize = int(''.join(messageBits[:-1]))
                        except ValueError:
                            pass
                    # Check if we have enough bits to determine the message size

                    if messSize is not None and len(messageBits) - len(str(messSize)) - 1 == messSize:
                        return ''.join(messageBits)[len(str(messSize)) + 1:]
        print(f"message: {''.join(messageBits)[len(str(messSize)) + 1:]}")
        return 'No hidden message found'

    def chunks(self, l, n):
        m = int(n)
        for i in range(0, len(l), m):
            yield l[i:i + m]

    def loadImage(self):
        img = cv2.imread(self.imPath, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        return img

    def addPadd(self, img):
        row, col = img.shape[:2]
        padded_img = np.zeros((row + (8 - row % 8), col + (8 - col % 8), 3), dtype=np.uint8)
        padded_img[:row, :col] = img  # Add zero padding at the edges
        return padded_img

    def toBits(self):
        bits = []
        for char in self.message:
            binval = bin(ord(char))[2:].rjust(8, '0')  # Convert each char to 8-bit binary
            bits.append(binval)
        self.numBits = bin(len(bits))[2:].rjust(8, '0')
        return bits
