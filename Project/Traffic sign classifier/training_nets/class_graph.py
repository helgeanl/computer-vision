from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, merge,Input
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import keras
#from preprocessing import histogram_eq
import numpy as np
import cv2

# dimensions of our images.
img_width, img_height = 32, 32

train_data_dir = '../data/train'
validation_data_dir = '../data/validation'


nb_train_samples = 39000
nb_validation_samples = 12000
epochs = 50
batch_size = 128

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)


def histogram_eq(img):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    img = img[:,:,0].astype(np.uint8)
    img = clahe.apply(img)
    img = img.astype(np.float64)
    img = np.reshape(img,(32,32,1))
    return img


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
filter_size5 = 5
num_filters5 = 128
## Convlayer 6
filter_size6 = 5
num_filters6 = 128

inputs = Input(shape=input_shape)
# Layer 0
layer0 = Conv2D(num_filters0, filter_size0, activation='relu')(inputs)
# Layer 1
layer1 = Conv2D(num_filters1, filter_size1, activation='relu', padding='same')(layer0)
layer1 = Conv2D(num_filters2, filter_size2, activation='relu', padding='same')(layer1)
layer1 = MaxPooling2D(pool_size=(2, 2))(layer1)
layer1 = BatchNormalization()(layer1)
layer1 = Dropout(0.2)(layer1)
layer1_flat = Flatten()(layer1)
# Layer 2
layer2 = Conv2D(num_filters3, filter_size3, activation='relu', padding='same')(layer1)
layer2 = Conv2D(num_filters4, filter_size4, activation='relu', padding='same')(layer2)
layer2 = MaxPooling2D(pool_size=(2, 2))(layer2)
layer2 = BatchNormalization()(layer2)
layer2 = Dropout(0.2)(layer2)
layer2_flat = Flatten()(layer2)
# Layer 3
layer3 = Conv2D(num_filters5, filter_size5, activation='relu', padding='same')(layer2)
layer3 = Conv2D(num_filters6, filter_size6, activation='relu', padding='same')(layer3)
layer3 = MaxPooling2D(pool_size=(2, 2))(layer3)
layer3 = BatchNormalization()(layer3)
layer3 = Dropout(0.2)(layer3)
layer3_flat = Flatten()(layer3)
# Fist fully connected layer
layer4 = keras.layers.concatenate([layer1_flat,layer2_flat,layer3_flat])
layer4 = Dense(1024,activation='relu', name='fc1')(layer4)
layer4 = BatchNormalization()(layer4)
layer4 = Dropout(0.2)(layer4)
# Second fully connected layer
layer5 = Dense(1024, activation='relu', name='fc2')(layer4)
layer5 = BatchNormalization()(layer5)
layer5 = Dropout(0.3)(layer5)
# Last fully connected layer
predictions = Dense(43,activation='softmax', name='predictions')(layer5)
model = Model(inputs=inputs, outputs=predictions)
# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())
plot_model(model, to_file='model_class_graph.png')

# Don't train just yet?
if True:

    """
    Augmentation configuration for the training data
    """
    train_datagen = ImageDataGenerator(
        rescale = 1./255, # Normalize the image data from 0-255 to 0-1
        preprocessing_function= histogram_eq,
        zoom_range=0.3,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=20,
        )


    """
    Augmentation configuration for the validation data
    """
    test_datagen = ImageDataGenerator(
                    rescale=1./255,
                    preprocessing_function= histogram_eq)

    """
    The generator will go through the subfolders in 'train_data_dir' and find
    all images in .png and .jpg format, and return batches of training data
    """
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode = 'grayscale',
        class_mode='categorical')

    """
    The generator will go through the subfolders in 'validation_data_dir' and find
    all images in .png and .jpg format, and return batches of test data
    """
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode = 'grayscale',
        class_mode='categorical')

    """
    Train the model with the training data
    """
    model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            verbose = 1,
            callbacks=[#LearningRateScheduler(lr_schedule),
                       ModelCheckpoint('model_class_graph.h5',save_best_only=True)],
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size)

    #model.save_weights('model_batch_weights.h5')
    #model.save('model_batch_drop_hist.h5')

    """
    Evaulate the trained model with the validation data
    """
    score = model.evaluate_generator(validation_generator, steps = 10000)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
