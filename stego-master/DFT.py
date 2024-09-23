#!/usr/local/bin/python
from __future__ import print_function

import itertools
import sys

import cv2
import numpy as np


class DFT():    
    def __init__(self, imPath):
        self.imPath = imPath
        self.message = None
        self.bitMess = None
        self.oriCol = 0
        self.oriRow = 0
        self.numBits = 0   
    
    def DFTEn(self, secret, outIm):
        # Load image for processing
        img = self.loadImage()
        if img is None:
            print("Error: File not found!")
            return

        self.message = str(len(secret)) + '*' + secret
        self.bitMess = self.toBits()
        
        # Get size of image in pixels
        row, col = img.shape[:2]
        self.oriRow, self.oriCol = row, col  

        if ((col // 8) * (row // 8) < len(secret)):
            print("Error: Message too large to encode in image")
            return      
        
        # Make divisible by 8x8
        if row % 8 != 0 or col % 8 != 0:
            img = self.addPadd(img, row, col)
        
        row, col = img.shape[:2]

        # Split image into RGB channels
        bImg, gImg, rImg = cv2.split(img)

        # Message to be hidden in blue channel so converted to type float32 for DFT function
        bImg = np.float32(bImg)
    
        # Apply DFT
        dft = cv2.dft(bImg, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Embed message in frequency components
        messIndex = 0
        letterIndex = 0
        for i in range(dft_shift.shape[0]):
            for j in range(dft_shift.shape[1]):
                if messIndex < len(self.message):
                    bit = self.bitMess[messIndex][letterIndex]
                    if bit == '1':
                        dft_shift[i, j][0] += 1
                    elif bit == '0':
                        dft_shift[i, j][0] -= 1
                    letterIndex += 1
                    if letterIndex == 8:
                        letterIndex = 0
                        messIndex += 1
                        if messIndex == len(self.message):
                            break

        # Apply inverse DFT
        dft_ishift = np.fft.ifftshift(dft_shift)
        sImg = cv2.idft(dft_ishift)
        sImg = cv2.magnitude(sImg[:, :, 0], sImg[:, :, 1])

        # Merge back the channels
        sImg = np.uint8(sImg)
        sImg = cv2.merge((sImg, gImg, rImg))
        cv2.imwrite(outIm, sImg)
        return sImg
    
    def DFTDe(self):
        img = cv2.imread(self.imPath, cv2.IMREAD_UNCHANGED)

        row, col = img.shape[:2]

        messSize = None
        messageBits = []
        buff = 0

        # Split image into RGB channels
        bImg, gImg, rImg = cv2.split(img)

        # Convert to type float32 for DFT function
        bImg = np.float32(bImg)
    
        # Apply DFT
        dft = cv2.dft(bImg, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Extract message from frequency components
        i = 0
        for row in dft_shift:
            for value in row:
                real = value[0]
                if real > 0:
                    buff += (0 & 1) << (7 - i)
                else:
                    buff += (1 & 1) << (7 - i)
                i += 1
                if i == 8:
                    messageBits.append(chr(buff))
                    buff = 0
                    i = 0
                    if messageBits[-1] == '*' and messSize is None:
                        try:
                            messSize = int(''.join(messageBits[:-1]))
                        except:
                            pass
                if len(messageBits) - len(str(messSize)) - 1 == messSize:
                    return ''.join(messageBits)[len(str(messSize)) + 1:]

        return 'Not thing'
    
    def chunks(self, l, n):
        m = int(n)
        for i in range(0, len(l), m):
            yield l[i:i + m]
    
    def loadImage(self):
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
