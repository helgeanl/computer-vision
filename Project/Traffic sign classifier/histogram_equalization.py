import cv2
from skimage import color, exposure, transform
from skimage import io
from keras.preprocessing import image as image_utils
import argparse
import numpy as np

from matplotlib import pyplot as plt

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-o","--output", default="out.jpg",required=False,
	help="Filename of the equalized image")
ap.add_argument("-n","--normalize", default=True,type=bool,
	help="Normalize the image from 0-255 to 0-1")
ap.add_argument("-r","--resize", default=32,type=int,
	help="Resize the image")
ap.add_argument("--histogram_equalization", default=True,
	help="Use histogram equalization")
args = vars(ap.parse_args())

def process(image,output,resize, normalize,histogram_equalization):


    # img = cv2.imread('stop.jpg',0)
    #
    # hist,bins = np.histogram(img.flatten(),256,[0,256])
    #
    # cdf = hist.cumsum()
    # cdf_normalized = cdf * hist.max()/ cdf.max()
    #
    # plt.plot(cdf_normalized, color = 'b')
    # plt.hist(img.flatten(),256,[0,256], color = 'r')
    # plt.xlim([0,256])
    # plt.legend(('cdf','histogram'), loc = 'upper left')
    # plt.show()
    #
    # cdf_m = np.ma.masked_equal(cdf,0)
    # cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    # cdf = np.ma.filled(cdf_m,0).astype('uint8')
    #
    # img2 = cdf[img]

    img = cv2.imread('20.jpg',0)
    equ = cv2.equalizeHist(img)
    #res = np.hstack((img,equ)) #stacking images side-by-side
    cv2.imwrite('res.png',equ)

    # create a CLAHE object (Arguments are optional).
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    cv2.imwrite('res2.jpg',cl1)

    # Load the image
    #image = cv2.imread(image)

    # Resize the image
    #resized_image = cv2.resize(image, (resize, resize))
    #image_array = image_utils.img_to_array(resized_image)

    # Normalize the image
    # if normalize:
    #     image_array = np.array(image_array)*1./255
    #
    # # Process with histogram equalization
    # if histogram_equalization:
    #     hsv = color.rgb2hsv(image_array)
    #     hsv[:,:,:] = exposure.equalize_hist(hsv[:,:,:])
    #     img = color.hsv2rgb(hsv)
    #
    # #image_array = np.expand_dims(image_array, axis=0)
    #
    # # Save the pocessed image
    # #io.imsave(output,image_array[0,:,:,:])
    # io.imsave(output,image_array)


process(args["image"],args["output"],args["resize"],args["normalize"],args["histogram_equalization"])
