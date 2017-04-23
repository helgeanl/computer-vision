import cv2
from keras.preprocessing import image as image_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
import PIL

def histogram_eq(img):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    img = img[:,:,0].astype(np.uint8)
    img = clahe.apply(img)
    img = img.astype(np.float64)
    img = np.reshape(img,(32,32,1))
    return img




#image_input = image_utils.load_img("../stop.jpg", target_size=(32, 32))
#image_input = cv2.imread("../test_data/70.jpg",0)
#print('Shape input',image_input.shape)
#image_array = image_utils.img_to_array(image_input)
#print(image_array.shape)
#print(image_input.shape)

#img_output = histogram_eq(image_input)
#print(img.shape)

#cv2.imwrite('../test_data/in.jpg', image_input)
#cv2.imwrite('../test_data/out.jpg', img_output)


'''


"""
Augmentation configuration for the training data
"""
train_datagen = ImageDataGenerator(
    rescale = 1./255, # Normalize the image data from 0-255 to 0-1
    #shear_range=0.2,
    zca_whitening = True,
    zoom_range=0.3,
    rotation_range=20,
    preprocessing_function= histogram_eq
    )
train_generator = train_datagen.flow_from_directory(
    '../image_test',
    target_size=(32, 32),
    batch_size=4,
    color_mode = 'grayscale',
    save_to_dir = './',#'../augmented_image/equalized',
    save_format = 'jpeg',
    save_prefix = 'zca_hist',
    class_mode='categorical')

model = load_model('../model_underfit.h5')

model.fit_generator(train_generator, steps_per_epoch = 4,epochs=1)
'''
