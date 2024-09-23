from skimage.metrics import structural_similarity as structSim
import matplotlib.pyplot as plot
import numpy
import cv2


def meanSquareError(im1, im2):
	error = numpy.sum((im1.astype('float') - im2.astype('float')) ** 2)
	error /= float(im1.shape[0] * im1.shape[1]);
	return error

def calculate_mse(original_image, compressed_image):
    """Calculate the Mean Squared Error (MSE) between two images."""
    mse = numpy.mean((original_image - compressed_image) ** 2)
    return mse

def peakSignalToNoiseRatio(im1, im2):
	mse = meanSquareError(im1, im2)
	if mse == 0:
		return float('inf')  # PSNR is infinite if there is no noise
	max_pixel = 255.0  # Assuming 8-bit images
	psnr = 20 * numpy.log10(max_pixel / numpy.sqrt(mse))
	return psnr

def calculate_psnr(original_image, compressed_image):
    """Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images."""
    mse = calculate_mse(original_image, compressed_image)
    if mse == 0:  # MSE is zero means no noise is present in the images
        return float('inf')  # PSNR is infinite
    max_pixel_value = 255.0  # Maximum pixel value for 8-bit image
    psnr = 10 * numpy.log10((max_pixel_value ** 2) / mse)
    return psnr
def PSNR(original, compressed): 
    mse = numpy.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return float('inf')
    max_pixel = 256.0
    psnr = 20 * numpy.log10(max_pixel / numpy.sqrt(mse)) 
    return psnr 

def compareImages(im1, im2, title):
	mse = calculate_mse(im1, im2)
	psnr = PSNR(im1, im2)
	
	# Specify win_size explicitly if images are small
	win_size = 3  # Use a smaller odd value if images are small
	ss = structSim(im1, im2, multichannel=True, win_size=win_size)

	# Make figure
	print(title)
	print('MSE: %.2f, PSNR: %.2f, SSIM: %.2f' % (mse, psnr, ss))
	fig = plot.figure(title)
	plot.suptitle('MSE: %.2f, PSNR: %.2f, SSIM: %.2f' % (mse, psnr, ss))

	ax = fig.add_subplot(1, 2, 1)
	plot.imshow(im1, cmap=plot.cm.gray)
	plot.axis('off')

	ax = fig.add_subplot(1, 2, 2)
	plot.imshow(im2, cmap=plot.cm.gray)
	plot.axis('off')

	plot.show()

def main():
	original = cv2.imread('img/lenna.png')
	lsbEncoded = cv2.imread('img/LSBlenna.png')
	dctEncoded = cv2.imread('img/DCTlenna.png')
	mydctEncoded = cv2.imread('img/MYDCTlenna.png')
	
	original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
	lsbEncoded = cv2.cvtColor(lsbEncoded, cv2.COLOR_BGR2RGB)
	dctEncoded = cv2.cvtColor(dctEncoded, cv2.COLOR_BGR2RGB)
	mydctEncoded = cv2.cvtColor(mydctEncoded, cv2.COLOR_BGR2RGB)
	# figure
	fig = plot.figure("Images")
	images = ('Original', original), ('LSB', lsbEncoded), ('DCT', dctEncoded), ('MYDCT', mydctEncoded)

	for (i, (name, image)) in enumerate(images):
		ax = fig.add_subplot(1,4,i+1)
		ax.set_title(name)
		plot.imshow(image, cmap=plot.cm.gray)
		plot.axis('off')

	plot.show()

	# Compare all the images
	compareImages(original, original, "Original vs Original")
	compareImages(original, lsbEncoded, "Original vs LSB")
	compareImages(original, dctEncoded, "Original vs DCT")
	compareImages(original, mydctEncoded, "Original vs MYDCT")
	#compareImages(dctEncoded, mydctEncoded, "DCT vs MYDCT")


if __name__=='__main__':
	main()