from keras.models import load_model
from keras.preprocessing import image as image_utils
from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing import image as image_utils
from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input
from keras.utils import plot_model
import numpy as np
import argparse
import cv2

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#import math
import cv2
import time as time
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.cross_validation import train_test_split
import numpy as np
#%matplotlib inline
#import tensorflow as tf
#import prettytensor as pt
from PIL import Image
#import time
#from datetime import timedelta
import pandas as pd




# score = model.evaluate_generator(validation_generator, steps = 100)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# image_data_dir = 'image_test'
# image_datagen = ImageDataGenerator(
#                 rescale=1./255,
#                 zca_whitening=True,
#                 shear_range=0.2,
#                 zoom_range=0.2,
#                 rotation_range=20)
# image_generator = image_datagen.flow_from_directory(
#         image_data_dir,
#         target_size=(32, 32),
#         batch_size=1,
#         #color_mode =  "grayscale",
#         save_to_dir = 'augmented_image',
#         save_format = 'jpeg',
#         class_mode= None)


# construct the argument parse and parse the arguments
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

model = load_model(args["model"])
if args["info"]:
    print(model.summary())
if args["plot_model"]:
    plot_model(model, to_file=args["plot_name"])

#predict = model.predict_generator(image_generator,steps=5)


# load the original image via OpenCV so we can draw on it and display
# it to our screen later
#orig = cv2.imread(args["image"])
t1 = time.time()
# NumPy array
print()
print("[INFO] loading and preprocessing image...")
image_input = image_utils.load_img(args["image"], target_size=(128, 128))
image_array = image_utils.img_to_array(image_input)
if args["normalize"]:
    image_array = np.array(image_array)*1./255
image = np.expand_dims(image_array, axis=0)

# classify the image
print("[INFO] classifying image...")
preds = model.predict(image)

if args["probabilities"]:
    print(preds)
top_indices = preds.argsort()[0][-1]

data_pd = pd.read_csv('sign_names.csv')
if preds[0][top_indices] > 0.8:
    sign = data_pd['SignName'][top_indices]
    index =top_indices
else:
    sign = 'No sign found'
    index = 100

print('[SIGN] ' + sign)
print('[INDEX] ' + str(index))
t2 = time.time()
print ('[TIME] ' + "{} ms".format ( (t2 - t1) * 1000.0))
print()

#eprocess_input(image)
#img = image.load_img('70.jpg', target_size=(32, 32))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#print(x.shape)
#image = preprocess_input(x)
#print(image.shape)
#image = load_img('70.jpg')
#image_array = img_to_array(image)

#print('Predicted:', decode_predictions(preds, top=3)[0])



#(inID, label) = decode_predictions(preds)[0]
#CLASS_INDEX = json.load(open('class_index.json'))
#results = []

# for pred in preds:
#     top_indices = pred.argsort()[-top:][::-1]
#     result = [pred[i] for i in top_indices]
#     results.append(result)
#label= results[0][0]
#print(label)
#id = label[0]
#label = label[1]
#P = decode_predictions(preds)
#(imagenetID, label, prob) = P[0][0]

# plt.figure(figsize = (5,1.5))
# gs = gridspec.GridSpec(1, 2,width_ratios=[2,3])
# plt.subplot(gs[0])
# plt.imshow(image_array)
# plt.axis('off')
# plt.subplot(gs[1])
# #plt.barh(6-np.arange(5),top_indices[0], align='center')
#
# for i_label in range(5):
#     plt.text(top_indices[i_label]+.02,6-i_label-.25,
#         data_pd['SignName'][top_indices[i_label]])
# plt.axis('off');
# #plt.text(0,6.95,namenewdata.split('.')[0]);
# plt.show();

# display the predictions to our screen
#print("ImageNet ID: {}, Label: {}".format(id, label))
# cv2.putText(orig, "Label: {}".format(label), (10, 30),
# 	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
# cv2.imshow("Classification", orig)
# cv2.waitKey(0)
