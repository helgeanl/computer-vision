from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Input, AveragePooling2D,BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense, Concatenate
from keras import backend as K
from keras.utils import plot_model
from keras.models import Model
import keras

#from preprocessing import histogram_eq

"""
Inception module as given in the GoogLeNet paper, Szegedy et. al, 2014
"""
def inception_module(prev_layer):
    #1x1
    strain_1 = Conv2D(128, 1,strides=1,activation='relu')(prev_layer)
    strain_1 = BatchNormalization()(strain_1)
    #3x3
    strain_2 = Conv2D(128, 1,strides=1,activation='relu')(prev_layer)
    strain_2 = BatchNormalization()(strain_2)
    strain_2 = Conv2D(192, 3,padding='same',activation='relu')(strain_2)
    strain_2 = BatchNormalization()(strain_2)
    #5x5
    strain_3 = Conv2D(32, 1,strides=1,activation='relu')(prev_layer)
    strain_3 = BatchNormalization()(strain_3)
    strain_3 = Conv2D(96, 5,padding='same',activation='relu')(strain_3)
    strain_3 = BatchNormalization()(strain_3)
    #Pool
    strain_4 = MaxPooling2D(pool_size=(3, 3),strides=1,padding='same')(prev_layer)
    strain_4 = Conv2D(64, 1,activation='relu')(strain_4)
    strain_4 = BatchNormalization()(strain_4)
    # Depth concatenate
    return keras.layers.concatenate([strain_1, strain_2,strain_3,strain_4], axis=-1)


"""
Directories to training data and validation data
"""
train_data_dir = '../data/train'
validation_data_dir = '../data/validation'

"""
Training configuration
"""
nb_train_samples = 39000
nb_validation_samples = 12000
epochs = 50
batch_size = 128

"""
Define the size of the input image, and check which format to use
(Tensorflow or Theano format)
"""
img_width, img_height = 224, 224
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

"""
Model inspired by the GoogLeNet paper, Szegedy et. al, 2014
(With batch normalization, Szegedy et. al, 2015 )
"""
inputs = Input(shape=input_shape)

low_layer = Conv2D(64, 7, strides=2, padding='same',activation='relu')(inputs)
low_layer = BatchNormalization()(low_layer)
low_layer = MaxPooling2D(pool_size=(3, 3),strides=2,padding='same')(low_layer)
low_layer = Conv2D(192, 3, strides=1, padding='same',activation='relu')(low_layer)
low_layer = BatchNormalization()(low_layer)

inception_layer1 = MaxPooling2D(pool_size=(3, 3),padding='same',strides=2)(low_layer)
inception_layer1a = inception_module(inception_layer1)
inception_layer1b = inception_module(inception_layer1a)
inception_layer1c = inception_module(inception_layer1b)

inception_layer2 = MaxPooling2D(pool_size=(3, 3),padding='same',strides=2)(inception_layer1c)
inception_layer2a = inception_module(inception_layer2)
inception_layer2b = inception_module(inception_layer2a)
inception_layer2c = inception_module(inception_layer2b)
inception_layer2d = inception_module(inception_layer2c)
inception_layer2e = inception_module(inception_layer2d)

inception_layer3 = MaxPooling2D(pool_size=(3, 3),padding='same',strides=2)(inception_layer2e)
inception_layer3a = inception_module(inception_layer3)
inception_layer3b = inception_module(inception_layer3a)

higher_layer = AveragePooling2D(pool_size=(7,7),strides=1)(inception_layer3b)
higher_layer = Dropout(0.1)(higher_layer)
higher_layer = Flatten()(higher_layer)
higher_layer = Dense(1000,activation='relu')(higher_layer)
higher_layer = BatchNormalization()(higher_layer)

predictions = Dense(43,activation='softmax')(higher_layer)

model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# initial learning rate of 0.0015?
#The initial learning rate was increased by a factor of 5, to 0.0075.
"""
Print and plot model
"""
print(model.summary())
plot_model(model, to_file='model_incpetion.png')


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
        width_shift_range=0.2,
        #preprocessing_function= histogram_eq,
        # Preprocessing with local histogram equalization/
        # contrast normalization could be an advantage
        height_shift_range=0.2)


    """
    Augmentation configuration for the validation data
    """
    test_datagen = ImageDataGenerator(
                    #preprocessing_function= histogram_eq,
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

    model.save_weights('incpetion_weights.h5')
    model.save('inception.h5')

    """
    Evaulate the trained model with the validation data
    """
    score = model.evaluate_generator(validation_generator, steps = 10000)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
