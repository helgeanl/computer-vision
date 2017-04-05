from skimage import io
import os
import glob

import numpy as np
from skimage import color, exposure, transform
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

NUM_CLASSES = 43
IMG_SIZE = 48

def preprocess_img(img):
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
    io.imsave('out2.jpg',img)
    # roll color axis to axis 0
    img = np.rollaxis(img,-1)
    #result = Image.fromarray(img )
    #result.save('out.bmp')

    return img


def get_class(img_path):
    return int(img_path.split('/')[-2])

root_dir = 'data/train'
imgs = []
labels = []

all_img_paths = glob.glob(os.path.join(root_dir, '*/*.jpg'))
#np.random.shuffle(all_img_paths)
for img_path in all_img_paths:
    img = preprocess_img(io.imread(img_path))
    label = get_class(img_path)
    imgs.append(img)
    labels.append(label)

X = np.array(imgs, dtype='float32')
# Make one hot targets
Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]


def cnn_model():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=(3, IMG_SIZE, IMG_SIZE),
                            activation='relu'))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 3, 3, border_mode='same',
                            activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, 3, 3, border_mode='same',
                            activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model


model = cnn_model()
print(model.summary())

# let's train the model using SGD + momentum
lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy',
#           optimizer=sgd,
#           metrics=['accuracy'])
#
def lr_schedule(epoch):
     return lr*(0.1**int(epoch/10))

batch_size = 32
nb_epoch = 30

# model.fit(X, Y,
#           batch_size=batch_size,
#           nb_epoch=nb_epoch,
#           validation_split=0.2,
#           callbacks=[LearningRateScheduler(lr_schedule),
#                     ModelCheckpoint('model.h5',save_best_only=True)]
#          )



from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(featurewise_center=False,
                            featurewise_std_normalization=False,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10.,)

datagen.fit(X_train)

# Reinitialize model and compile
model = cnn_model()
model.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])

# Train again
nb_epoch = 30
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            validation_data=(X_val, Y_val),
                            callbacks=[LearningRateScheduler(lr_schedule),
                                       ModelCheckpoint('model_pre.h5',save_best_only=True)]
                    )
