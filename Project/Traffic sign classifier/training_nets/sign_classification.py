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





# dimensions of our images.
img_width, img_height = 32, 32

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
# Layer 1
model.add(Conv2D(num_filters1, filter_size1, activation='relu', padding='same'))
# Layer 2
model.add(Conv2D(num_filters2, filter_size2, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
# Layer 3
model.add(Conv2D(num_filters3, filter_size3, activation='relu', padding='same'))
# Layer 4
model.add(Conv2D(num_filters4, filter_size4, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
# Layer 5
model.add(Conv2D(num_filters5, filter_size5, activation='relu', padding='same'))
# Layer 6
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

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
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
# model.save_weights('second_try.h5')  # always save your weights after training or during training
# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)



#load weights into new model
model.load_weights("second_try.h5")
print("Loaded model from disk")
model.save('model.h5')
model.load_model('model.h5')
# score = model.evaluate_generator(validation_generator, steps = 100)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# image_data_dir = 'image_test'
# image_datagen = ImageDataGenerator(rescale=1./255)
# image_generator = test_datagen.flow_from_directory(
#         image_data_dir,
#         target_size=(img_width, img_height),
#         batch_size=1,
#         save_to_dir = 'augmented_image',
#         class_mode= None)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# load the original image via OpenCV so we can draw on it and display
# it to our screen later
orig = cv2.imread(args["image"])

# NumPy array
print("[INFO] loading and preprocessing image...")
image = image_utils.load_img(args["image"], target_size=(32, 32))
image = image_utils.img_to_array(image)
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)
#img = image.load_img('70.jpg', target_size=(32, 32))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#print(x.shape)
#image = preprocess_input(x)
#print(image.shape)
#image = load_img('70.jpg')
#image_array = img_to_array(image)

#print('Predicted:', decode_predictions(preds, top=3)[0])

# classify the image
print("[INFO] classifying image...")
preds = model.predict(image)
#(inID, label) = decode_predictions(preds)[0]
results = []
top = 5
for pred in preds:
    top_indices = pred.argsort()[-top:][::-1]
    result = [tuple('CLASS_INDEX[str(i)]') + (pred[i],) for i in top_indices]
    results.append(result)
(inID, label) = results[0]
#P = decode_predictions(preds)
#(imagenetID, label, prob) = P[0][0]

# display the predictions to our screen
print("ImageNet ID: {}, Label: {}".format(inID, label))
cv2.putText(orig, "Label: {}".format(label), (10, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)


# predictions = model.predict_generator(image_generator, steps=4, verbose=1)
# print(predictions)

#plot_model(model, to_file='model.png')
# show_shapes = false

# evaluate loaded model on test data
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)
#pred = model.predict_classes(self, x, batch_size=32, verbose=1)
#score = model.evaluate(validation_generator, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
