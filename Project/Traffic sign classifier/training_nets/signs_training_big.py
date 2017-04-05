from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import plot_model


# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 32000
nb_validation_samples = 12000
epochs = 50
batch_size = 10

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (1, 1)))
model.add(Activation('relu'))

model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(Activation('relu'))

model.add(Conv2D(16,(1,1)))
model.add(Activation('relu'))

model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (1, 1)))
model.add(Activation('relu'))

model.add(Conv2D(256,(3,3)))
model.add(Activation('relu'))

model.add(Conv2D(32,(1,1)))
model.add(Activation('relu'))

model.add(Conv2D(256,(3,3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(1,1)))
model.add(Activation('relu'))

model.add(Conv2D(512,(3,3)))
model.add(Activation('relu'))

model.add(Conv2D(64,(1,1)))
model.add(Activation('relu'))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(43))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

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
# model.save_weights('first_try.h5')  # always save your weights after training or during training
# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)


#
# print('Test score:',score[0])
# print('Test accuracy:', score[1])


# load weights into new model
model.load_weights("first_try.h5")
print("Loaded model from disk")

predictions = model.predict_generator(validation_generator, steps=1)
print(predictions)

plot_model(model, to_file='model.png')
# show_shapes = false

# evaluate loaded model on test data
#model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)
#pred = model.predict_classes(self, x, batch_size=32, verbose=1)
#score = model.evaluate(validation_generator, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
