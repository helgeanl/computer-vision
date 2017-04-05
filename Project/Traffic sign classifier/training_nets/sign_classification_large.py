from keras.preprocessing.image import ImageDataGenerator, load_img,img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing import image as image_utils
#from keras.preprocessing import image as image_utils
from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input
import numpy as np
import argparse
import cv2

#from parser import load_data


# dimensions of our images.
img_width, img_height = 128, 128

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

#training_data = load_data(train_data_dir)
#validation_data = load_data(validation_data_dir)


nb_train_samples = 32000
nb_validation_samples = 12000
epochs = 150
batch_size = 128

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


## Convlayer 0
filter_size0 = 1
num_filters0 = 3

## Convlayer 1
filter_size1 = 5
num_filters1 = 32
## Convlayer 2
filter_size2 = 5
num_filters2 = 32

## Convlayer 3
filter_size3 = 5
num_filters3 = 64
## Convlayer 4
filter_size4 = 5
num_filters4 = 64

## Convlayer 5
filter_size5 = 3
num_filters5 = 128
## Convlayer 6
filter_size6 = 3
num_filters6 = 128

## Dropout
drop_prob = 0.5



model = Sequential()

# Layer 0
model.add(Conv2D(num_filters0, filter_size0, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Layer 1
model.add(Conv2D(num_filters1, filter_size1, activation='relu', padding='same'))
model.add(Conv2D(num_filters2, filter_size2, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
# Layer 3
model.add(Conv2D(num_filters3, filter_size3, activation='relu', padding='same'))
model.add(Conv2D(num_filters4, filter_size4, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
# Layer 5
model.add(Conv2D(num_filters5, filter_size5, activation='relu', padding='same'))
model.add(Conv2D(num_filters6, filter_size6, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
# Layer 6
model.add(Conv2D(num_filters5, filter_size5, activation='relu', padding='same'))
model.add(Conv2D(num_filters6, filter_size6, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

# Fist fully connected layer
model.add(Flatten())
model.add(Dense(1024,activation='relu', name='fc1'))
model.add(Dropout(0.5))
# Second fully connected layer
#model.add(Dense(1024, activation='relu', name='fc2'))
#model.add(Dropout(0.5))
# Last fully connected layer
model.add(Dense(43,activation='softmax', name='predictions'))
# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.5,
        zoom_range=0.5,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        fill_mode='nearest',
        zca_whitening = True,
        #horizontal_flip=True)
        rotation_range=20)
        #width_shift_range=0.2,
        #fill_mode='nearest')

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        train_data_dir,  # this is the target directory
        target_size=(img_width, img_height),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')


model.fit_generator(
         train_generator,
         steps_per_epoch=nb_train_samples // batch_size,
         epochs=epochs,
         validation_data=validation_generator,
         validation_steps=nb_validation_samples // batch_size)
model.save('model_large.h5')
# model.save_weights('second_try.h5')  # always save your weights after training or during training

#load weights into new model
#model.load_weights("second_try.h5")
#print("Loaded model from disk")
#model.save('model.h5')
#model.load_model('model.h5')
score = model.evaluate_generator(validation_generator, steps = 100)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
