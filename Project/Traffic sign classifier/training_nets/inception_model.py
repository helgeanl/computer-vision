#from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
from keras.applications.inception_v3 import InceptionV3
model = InceptionV3(weights='imagenet', include_top=True)

model.summary()
#plot_model(model, to_file='inception_v3.png')
