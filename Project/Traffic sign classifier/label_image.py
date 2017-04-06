"""
Predict the class of an image

args : [-h] -i IMAGE -m MODEL [--info INFO] [--plot_model PLOT_MODEL]
             [--plot_name PLOT_NAME] [-n NORMALIZE] [-p PROBABILITIES]

"""

import argparse
import numpy as np
import pandas as pd
import time as time

from keras.models import load_model
from keras.preprocessing import image as image_utils
from keras.utils import plot_model

#from sklearn.preprocessing import OneHotEncoder
#from sklearn.cross_validation import train_test_split



# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-m","--model", required=True,
	help="path to the model in .h5 format")
ap.add_argument("--info", required=False, default=False,
	help="Show a summary of the model")
ap.add_argument("--plot_model", required=False, default=False,
	help="Save an image of the model")
ap.add_argument("--plot_name", required=False, default="model.png",
	help="Normalize the data from 0-255 to 0-1")
ap.add_argument("-n","--normalize", required=False, default=True,
	help="Name of the model")
ap.add_argument("-p","--probabilities", required=False, default=False,
	help="Show the probabilities")
args = vars(ap.parse_args())

# Load the CNN model with weights (.h5 file)
model = load_model(args["model"])

# If specified, print out the model
if args["info"]:
    print(model.summary())

# If specified, plot the model
if args["plot_model"]:
    plot_model(model, to_file=args["plot_name"])

# Start time
t1 = time.time()

# Load image file
print()
print("[INFO] loading and preprocessing image...")
image_input = image_utils.load_img(args["image"], target_size=(32, 32))
image_array = image_utils.img_to_array(image_input)

# If specified, normalize the image from 0-255 to 0-1
if args["normalize"]:
    image_array = np.array(image_array)*1./255
image = np.expand_dims(image_array, axis=0)

# Predict the class of the image
print("[INFO] classifying image...")
preds = model.predict(image)

# If specified, print out list of probabilities
if args["probabilities"]:
    print(preds)

# Find the sign-index with the highest probability
top_indices = preds.argsort()[0][-1]

# Read label names
data_pd = pd.read_csv('sign_names.csv')

# Check if there is a probaliblity of a sign present
if preds[0][top_indices] > 0.8:
    sign = data_pd['SignName'][top_indices]
    index =top_indices
else:
    sign = 'No sign found'
    index = 100

# Print out the results
print('[SIGN] ' + sign)
print('[INDEX] ' + str(index))
t2 = time.time() # End time of process
print ('[TIME] ' + "{} ms".format ( (t2 - t1) * 1000.0))
print()
