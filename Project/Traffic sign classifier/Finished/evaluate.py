from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import argparse


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-d","--validation_data_dir", required=True,
	help="path to the validation data directory")
ap.add_argument("-m","--model", required=True,
	help="path to the model in .h5 format")
args = vars(ap.parse_args())

batch_size = 128
img_height = 32
img_width = 32

test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
        args["validation_data_dir"],
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')


model = load_model(args["model"])


score = model.evaluate_generator(validation_generator, steps = 10000)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
