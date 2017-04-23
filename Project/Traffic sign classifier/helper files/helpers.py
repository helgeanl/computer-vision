# import the necessary packages
import imutils
from keras.models import load_model
from keras.preprocessing import image as image_utils
#from keras.preprocessing import image as image_utils
from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input
from keras.utils import plot_model
import numpy as np
import argparse
import cv2
import time
import pandas as pd
from PIL import Image
from skimage import io
from skimage import color, exposure, transform



def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield image

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


model = load_model("model.h5")
data_pd = pd.read_csv('sign_names.csv')
#image_input = image_utils.load_img(args["test_data/road_big.jpg"], target_size=(1360, 800))
#image_array = image_utils.img_to_array(image_input)
image = cv2.imread("test_data/road_big.jpg")
#image = image_utils.load_img("test_data/road_big.jpg")
(winW, winH) = (48, 48)
i = 0
for (x, y, window) in sliding_window(image, stepSize=32, windowSize=(winW, winH)):
    # if the window does not meet our desired window size, ignore it
    if window.shape[0] != winH or window.shape[1] != winW:
        continue

        # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
        # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
        # WINDOW

        #print()
        #print("[INFO] loading and preprocessing image...")
        #image_input = image_utils.load_img(args["image"], target_size=(32, 32))
        #image_array = image_utils.img_to_array(image_input)
        #if args["normalize"]:
    #        image_array = np.array(image_array)*1./255
    #    image = np.expand_dims(image_array, axis=0)

    # classify the image
    #print("[INFO] classifying image...")
    resized_image = cv2.resize(window, (32, 32))
    image_array = image_utils.img_to_array(resized_image)
    image_array = np.array(image_array)*1./255
    #hsv = color.rgb2hsv(image_array)
    #hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    #img = color.hsv2rgb(hsv)

    image_array = np.expand_dims(image_array, axis=0)
    preds = model.predict(image_array)

    #if args["probabilities"]:

    top_indices = preds.argsort()[0][-1]


    if preds[0][top_indices] > 0.9:
        i += 1
        #print(preds)
        sign = data_pd['SignName'][top_indices]
        index =top_indices
        print('[SIGN] ' + sign)
        #print('[INDEX] ' + str(index))
        #print()
        #result = Image.fromarray((image_array * 255).astype(np.uint8))
        #result.save(str(i)+'_'+str(index)+'.bmp')
        io.imsave(str(i)+'_'+str(index)+'.jpg',image_array[0,:,:,:])
    #else:
    #    sign = 'No sign found'
    #    index = 100


    #t2 = time.time()
    #print ('[TIME] ' + "{} ms".format ( (t2 - t1) * 1000.0))


    # since we do not have a classifier, we'll just draw the window
    # clone = image.copy()
    # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    # cv2.imshow("Window", clone)
    # cv2.waitKey(1)
    # time.sleep(0.025)
