from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Activation, Dropout, Flatten, Dense, Concatenate
from keras import backend as K
from keras.utils import plot_model
from keras.models import Model
import keras

#from parser import load_data


#Total params: 16,571,223

# dimensions of our images.
img_width, img_height = 32, 32

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

#training_data = load_data(train_data_dir)
#validation_data = load_data(validation_data_dir)


nb_train_samples = 32000
nb_validation_samples = 12000
epochs = 50
batch_size = 128

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


## Convlayer 0
filter_size1 = 1
num_filters1 = 3

## Convlayer 1
filter_size2 = 5
num_filters2 = 32
## Convlayer 2
filter_size3 = 5
num_filters3 = 32

## Convlayer 3
filter_size4 = 5
num_filters4 = 64
## Convlayer 4
filter_size5 = 5
num_filters5 = 64

## Convlayer 5
filter_size6 = 5
num_filters6 = 128
## Convlayer 6
filter_size7 = 5
num_filters7 = 128

## Dropout
drop_prob = 0.5


inputs = Input(shape=input_shape)

conv_1 = Conv2D(num_filters1, filter_size1, padding='same', activation='relu')(inputs)

conv_2 = Conv2D(num_filters2, filter_size2, padding='same', activation='relu')(conv_1)
conv_3 = Conv2D(num_filters3, filter_size3, padding='same', activation='relu')(conv_2)
pooling_1 = MaxPooling2D(pool_size=(2, 2))(conv_3)
dropout_1 = Dropout(0.5)(pooling_1)
flatten_1 = Flatten()(dropout_1)

conv_4 = Conv2D(num_filters4, filter_size4, padding='same', activation='relu')(dropout_1)
conv_5 = Conv2D(num_filters5, filter_size5, padding='same', activation='relu')(conv_4)
pooling_2 = MaxPooling2D(pool_size=(2, 2))(conv_5)
dropout_2 = Dropout(0.5)(pooling_2)
flatten_2 = Flatten()(dropout_2)

conv_6 = Conv2D(num_filters6, filter_size6, padding='same', activation='relu')(dropout_2)
conv_7 = Conv2D(num_filters7, filter_size7, padding='same', activation='relu')(conv_6)
pooling_3 = MaxPooling2D(pool_size=(2, 2))(conv_7)
dropout_3 = Dropout(0.5)(pooling_3)
flatten_3 = Flatten()(dropout_3)

merged_vector = keras.layers.concatenate([flatten_1, flatten_2,flatten_3], axis=1)

#merged_vector = keras.layers.concatenate([dropout_1, dropout_2,dropout_3], axis=-1)
#flatten = Flatten()(merged_vector)

#fc_1 = Dense(1024,activation='relu', name='fc1')(flatten)
fc_1 = Dense(1024,activation='relu', name='fc1')(merged_vector)
dropout_4 = Dropout(0.5)(fc_1)
fc_2 = Dense(1024,activation='relu', name='fc2')(dropout_4)
dropout_5 = Dropout(0.5)(fc_2)

predictions = Dense(43, activation='softmax')(dropout_5)
model = Model(inputs=inputs, outputs=predictions)


# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())
#plot_model(model, to_file='model_graph.png')



# Don't train just yet
if False:
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        #horizontal_flip=True)
        rotation_range=40)
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
    # the predict_generator method returns the output of a model, given
    # a generator that yields batches of numpy data
    #bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    # save the output as a Numpy array
    #np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')


    #bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
    #np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)


    # model.fit_generator(
    #         train_generator,
    #         steps_per_epoch=nb_train_samples // batch_size,
    #         epochs=epochs,
    #         validation_data=validation_generator,
    #         validation_steps=nb_validation_samples // batch_size)

    #model.save_weights('second_try.h5')  # always save your weights after training or during training
    # # serialize model to JSON
    # model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)



    #load weights into new model
    model.load_weights("graph_weights.h5")
    #print("Loaded model from disk")
    model.save('model_graph.h5')
    #score = model.evaluate_generator(validation_generator, steps = 10000)
    #print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    #
    #


#predictions = model.predict_generator(validation_generator, steps=1)
#print(predictions)

#plot_model(model, to_file='model.png')
# show_shapes = false

# evaluate loaded model on test data
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)
#pred = model.predict_classes(self, x, batch_size=32, verbose=1)
#score = model.evaluate(validation_generator, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
