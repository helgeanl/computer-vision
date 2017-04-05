from keras.preprocessing import image as image_utils
import numpy as np
from sklearn.utils import shuffle
from skimage import exposure
import matplotlib
from matplotlib import pyplot
from PIL import Image
import warnings
num_classes = 43

def preprocess_dataset(X, y = None):
    """
    Performs feature scaling, one-hot encoding of labels and shuffles the data if labels are provided.
    Assumes original dataset is sorted by labels.

    Parameters
    ----------
    X                : ndarray
                       Dataset array containing feature examples.
    y                : ndarray, optional, defaults to `None`
                       Dataset labels in index form.
    Returns
    -------
    A tuple of X and y.
    """
    print("Preprocessing dataset with {} examples:".format(X.shape[0]))

    #Convert to grayscale, e.g. single channel Y
    X = 0.299 * X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]
    #Scale features to be in [0, 1]
    X = (X / 255.).astype(np.float32)

    for i in range(X.shape[0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X[i] = exposure.equalize_adapthist(X[i])
        #print_progress(i + 1, X.shape[0])

    if y is not None:
        # Convert to one-hot encoding. Convert back with `y = y.nonzero()[1]`
        y = np.eye(num_classes)[y]
        X, y = shuffle(X, y)

    # Add a single grayscale channel
    X = X.reshape(X.shape + (1,))
    return X, y

#image_input = image_utils.load_img("stop.jpg", target_size=(32, 32))
#image_array = np.array([image_utils.img_to_array(image_input)])
#(img,y) = preprocess_dataset(image_array)
img = Image.open("stop.jpg").convert('L')
X = np.array(img)
#fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(im)))
#(X,y) = preprocess_dataset(np.array([[im]]))
#Convert to grayscale, e.g. single channel Y
print(X.shape)
X = 0.299 * X[:, :, 0] + 0.587 * X[:, :, 1] + 0.114 * X[:, :, 2]
X = exposure.equalize_adapthist(X)
#X = X.reshape(X.shape + (1,))
#visual = np.log(fft_mag)
#visual = (visual - visual.min()) / (visual.max() - visual.min())

result = Image.fromarray((X * 255).astype(np.uint8))
result.save('out.bmp')

#result = Image.fromarray(image_input )#.astype(np.uint8))
#result.save('out.jpg')

#fig = pyplot.figure(figsize = (1, 1))
#fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)
#axis = fig.add_subplot(1, 10, 1 + 1, xticks=[], yticks=[])
#axis.imshow(img)
#pyplot.show()

#im = Image.open(origin_folder+dest_directory+'/'+filename)
#im.save(dest_directory+'/'+filenameJPG)
