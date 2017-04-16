import cv2
from keras.preprocessing import image as image_utils

def histogram_eq(img):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    #img = clahe.apply(img)
    # equalize the histogram of the Y channel
    #img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])



    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    img_yuv[:,:,1] = clahe.apply(img_yuv[:,:,1])
    img_yuv[:,:,2] = clahe.apply(img_yuv[:,:,2])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img


#image_input = image_utils.load_img("../stop.jpg", target_size=(32, 32))
image_input = cv2.imread("../60_2.jpg",0)
#print(image_input)
#image_array = image_utils.img_to_array(image_input)
#print(image_array.shape)

img_output = histogram_eq(image_input)
#print(img.shape)

cv2.imwrite('in.jpg', image_input)
cv2.imwrite('out.jpg', img_output)
