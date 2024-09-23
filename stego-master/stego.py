#!/usr/bin/python3

import os
import sys
import time
from argparse import ArgumentParser

from DCT import DCT
from DFT import DFT  # Import DFT class
from DWT import DWT  # Import DWT class
from LSB import LSB
from myDCT import DCT as MYDCT


def parser():
    # Set the command line arguments
    parser = ArgumentParser(description="Stego: DCT, DWT, DFT, and LSB Image Steganography")

    parser.add_argument('-d', dest='encrypt', action='store_false',
                        help="Set method to decode, default is encode",
                        default=True)

    # Separate flags for each algorithm
    parser.add_argument('--lsb', action='store_true', help="Use LSB algorithm")
    parser.add_argument('--dct', action='store_true', help="Use DCT algorithm")
    parser.add_argument('--mydct', action='store_true', help="Use MYDCT algorithm")
    parser.add_argument('--dwt', action='store_true', help="Use DWT algorithm")
    parser.add_argument('--dft', action='store_true', help="Use DFT algorithm")

    parser.add_argument("-i", dest="inputfile", required=True,
                        help="Specify input file name", metavar="FILE")

    parser.add_argument("-o", dest="outputfile", required=False,
                        help="Specify output file name (optional)", metavar="FILE")

    parser.add_argument("-s", dest="string", required=False,
                        help="Specify message to encrypt")

    parser.add_argument("-f", dest="file", required=False,
                        help="Specify text file containing message", metavar="FILE")

    args = parser.parse_args()

    # Determine which algorithm to use based on flags
    if args.lsb:
        args.algorithm = 'LSB'
    elif args.dct:
        args.algorithm = 'DCT'
    elif args.mydct:
        args.algorithm = 'MYDCT'
    elif args.dwt:
        args.algorithm = 'DWT'
    elif args.dft:
        args.algorithm = 'DFT'
    else:
        args.algorithm = 'DCT'  # Default to DCT if no algorithm is specified

    return args

def main():
    args = parser()
    
    algo = args.algorithm
    inFile = args.inputfile
    message = args.string
    outFile = args.outputfile
    msgFile = args.file


    #encryption input check
    if args.encrypt is True and args.string is None and args.file is None:
        raise ValueError("Encryption requires an input string")

    # read file msg
    if args.file is not None:
        with open(msgFile, 'r') as textFile:
            message = textFile.read().replace('\n', '')
        
    #encryption
    if args.encrypt:

        #set output file if not specified
        if not args.outputfile:
            rawName = os.path.basename(os.path.normpath(inFile))
            dirName = os.path.dirname(os.path.normpath(inFile))

            outFile = dirName + '/' + algo + rawName

        #LSB implementation
        if algo == "LSB":
            start = time.time()
            x = LSB(inFile)
            encoded = x.hide(message, outFile)
            end = time.time()-start
            print ('time: ')
            print (end)
            #print ('Message encoded = ' + x.message)
        #DCT implementation
        elif algo == "MYDCT":
            start = time.time()
            x = MYDCT(inFile)
            secret = x.DCTEn(message, outFile)
            end = time.time()-start
            print ('time: ')
            print (end)
            #print('Message encoded = '+ x.message)
        elif algo == "DCT":
            start = time.time()
            x = DCT(inFile)
            secret = x.DCTEn(message, outFile)
            end = time.time()-start
            print ('time: ')
            print (end)
        #DWT implementation
        elif algo == "DWT":
            start = time.time()
            print(inFile)
            x = DWT(inFile)
            secret = x.DWTEn(message, outFile)
            end = time.time()-start
            print ('time: ')
            print (end)
            #print('Message encoded = '+ x.message)
        #DFT implementation
        elif algo == "DFT":
            start = time.time()
            x = DFT(inFile)
            secret = x.DFTEn(message, outFile)
            end = time.time()-start
            print ('time: ')
            print (end)
            #print('Message encoded = '+ x.message)

    #decryption
    else:
        #LSB implementation
        if algo == 'LSB':
            start = time.time()
            y = LSB(inFile)
            secret = y.extract()
            end = time.time()-start
            print('Hidden Message:\n' + secret)
            print ('time: ')
            print (end)
        #DCT implementation
        elif algo == 'DCT':
            start = time.time()
            y = DCT(inFile)
            decode = y.DCTDe()
            end = time.time()-start
            print('Hidden Message:\n' + decode)
            print ('time: ')
            print (end)
        #My DCT 
        elif algo == 'MYDCT':
            start = time.time()
            y = MYDCT(inFile)
            decode = y.DCTDe()
            end = time.time()-start
            print('Hidden Message:\n' + decode)
            print ('time: ')
            print (end)
        #DWT implementation "source.organizeImports": "explicit"
        elif algo == 'DWT':
            start = time.time()
            print(inFile)
            y = DWT(inFile)
            decode = y.DWTDe()
            end = time.time()-start
            print('Hidden Message:\n' + decode)
            print ('time: ')
            print (end)
        #DFT implementation
        elif algo == 'DFT':
            start = time.time()
            y = DFT(inFile)
            decode = y.DFTDe()
            end = time.time()-start
            print('Hidden Message:\n' + decode)
            print ('time: ')
            print (end)

if __name__ == "__main__":
    main()
