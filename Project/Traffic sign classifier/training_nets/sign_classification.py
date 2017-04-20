from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import plot_model

# dimensions of our images.
img_width, img_height = 32, 32

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

nb_train_samples = 32000
nb_validation_samples = 12000
epochs = 100
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

model = Sequential()
# Layer 0
model.add(Conv2D(num_filters0, filter_size0, activation='relu', input_shape=input_shape))
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
# Fist fully connected layer
model.add(Flatten())
model.add(Dense(1024,activation='relu', name='fc1'))
model.add(Dropout(0.5))
# Second fully connected layer
model.add(Dense(1024, activation='relu', name='fc2'))
model.add(Dropout(0.5))
# Last fully connected layer
model.add(Dense(43,activation='softmax', name='predictions'))
# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

# Don't train just yet?
if False:

    """
    Augmentation configuration for the training data
    """
    train_datagen = ImageDataGenerator(
        rescale = 1./255, # Normalize the image data from 0-255 to 0-1
        shear_range=0.2,
        zoom_range=0.3,
        rotation_range=20,
        )


    """
    Augmentation configuration for the validation data
    """
    test_datagen = ImageDataGenerator(
                    rescale=1./255)

    """
    The generator will go through the subfolders in 'train_data_dir' and find
    all images in .png and .jpg format, and return batches of training data
    """
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    """
    The generator will go through the subfolders in 'validation_data_dir' and find
    all images in .png and .jpg format, and return batches of test data
    """
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    """
    Train the model with the training data
    """
    model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            verbose = 1,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size)

    model.save_weights('model_weights.h5')
    model.save('model.h5')

    """
    Evaulate the trained model with the validation data
    """
    score = model.evaluate_generator(validation_generator, steps = 10000)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
