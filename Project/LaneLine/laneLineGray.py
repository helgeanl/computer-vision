# Start of Algorithm: 

# 1. Convert from RGB colors to HSV
# 2. Apply Gaussian Blur to remove noice and removing unnecessary details and pixels
# 3. Turn every pixel that is neither white or yellow black. 
# 4. Apply Canny Edge detection to detect quick changes in color in neighbouring pixels
# 5. Define region of interest
# 6. Hough Transform inside the region of interest. This will extract lines passing through all the edge points and group by similaritty. This wil group the edge points located on the same line.
# 7. Filter the lines with horizontal slopes
# 8. Distinguish beetween left and right lane
# 9. Compute Linear Regression
# 10.Add Lines to Original Image
# 11.Some smoothing for videos

# End of Algorithm


import cv2
import numpy as np 
#import matplotlib as mpl
#import matplotlib
import matplotlib.image as mimg
import matplotlib.pyplot as mplt
from moviepy.video.io.VideoFileClip import VideoFileClip
from numpy.polynomial import Polynomial as P



BASE_IMAGE = None
CANNY_IMAGE = None
HOUGH_IMG = None

PREV_LEFT_X1 = None
PREV_LEFT_X2 = None
PREV_RIGHT_X1 = None 
PREV_RIGHT_X2 = None


#image = matplotlib.image.imread('test1.jpg')

def pause():
    programPause = input("Press the <ENTER> key to continue...")

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

def rgbToHsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def rgbToGray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussianBlur(img, kSize):
 	return cv2.GaussianBlur(img,(kSize, kSize), 0)

def cannyEdgeDetection(img, lTreshold, hTreshold):
 	return cv2.Canny(img, lTreshold, hTreshold)

def colorFilter(img):
    minYellow = np.array([20, 80, 80], np.uint8)
    maxYellow = np.array([60, 255, 255], np.uint8)
    maskedYellow = cv2.inRange(img, minYellow, maxYellow)
    
    minWhite = np.array([0, 0, 150],np.uint8 )
    maxWhite = np.array([255, 50, 255],np.uint8)
    maskedWhite = cv2.inRange(img, minWhite, maxWhite)
    
    
    img = cv2.bitwise_and(img, img, mask = cv2.bitwise_or(maskedYellow, maskedWhite))
    
    return img

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = [255,]*channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


#Lines returned from Hough Transform is a vector on the form (x1, y1, x2, y2), that is start and end points. 
def findSlopes(line):
    return (float(line[3]) - line[1]) / (float(line[2]) - line[0])



def drawLines(img, lines, color=[255, 0, 255], thickness=10):
    global PREV_LEFT_X1, PREV_LEFT_X2, PREV_RIGHT_X1, PREV_RIGHT_X2
    left_x = []
    left_y = []
    right_x = []
    right_y = []
    try:
        try:
            for line in lines:
                line = line[0]
                s = findSlopes(line)
                
                if -0.2 < s < 0.2:
                    continue

                if s < 0:
                    if line[0] > img.shape[1] / 2 + 40:
                        continue

                    left_x += [line[0], line[2]]
                    left_y += [line[1], line[3]]
                    #cv2.line(img, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), [0, 0, 255], thickness)
                else:
                    if line[0] < img.shape[1] / 2 - 40:
                        continue

                    right_x += [line[0], line[2]]
                    right_y += [line[1], line[3]]
                    #cv2.line(img, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), [255, 255, 0], thickness)
        except TypeError:
            print('No lines was found in current frame. Check if parameter values are suitable')
            return
        y1 = img.shape[0]
        y2 = img.shape[0] / 2 + 90

        if len(left_x) <= 1 or len(right_x) <= 1:
            if PREV_LEFT_X1 is not None:
                cv2.line(img, (int(PREV_LEFT_X1), int(y1)), (int(PREV_LEFT_X2), int(y2)), color, thickness)
                cv2.line(img, (int(PREV_LEFT_X2), int(y1)), (int(PREV_RIGHT_X2), int(y2)), color, thickness)
            return

        left_poly = P.fit(np.array(left_x), np.array(left_y), 1)
        right_poly = P.fit(np.array(right_x), np.array(right_y), 1)

        left_x1 = (left_poly - y1).roots()
        right_x1 = (right_poly - y1).roots()

        left_x2 = (left_poly - y2).roots()
        right_x2 = (right_poly - y2).roots()

        if PREV_LEFT_X1 is not None:
            left_x1 = PREV_LEFT_X1 * 0.7 + left_x1 * 0.3
            left_x2 = PREV_LEFT_X2 * 0.7 + left_x2 * 0.3
            right_x1 = PREV_RIGHT_X1 * 0.7 + right_x1 * 0.3
            right_x2 = PREV_RIGHT_X2 * 0.7 + right_x2 * 0.3

        PREV_LEFT_X1 = left_x1
        PREV_LEFT_X2 = left_x2
        PREV_RIGHT_X1 = right_x1
        PREV_RIGHT_X2 = right_x2
        if left_x1:
            cv2.line(img, (int(left_x1), int(y1)), (int(left_x2), int(y2)), color, thickness)
        if right_x1:
            cv2.line(img, (int(right_x1), int(y1)), (int(right_x2), int(y2)), color, thickness)
    except:
        print('Unexpected error in drawlines()')
        return
#    ysize = img.shape[0]
#    xsize = img.shape[1]
#    
#    pts =  np.array(
#            [[left_x1, left_x2, right_x1, right_x2]],
#            dtype=np.int32
#        )
#    cv2.fillPoly(img, pts,(0, 255, 0))



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    global HOUGH_IMG
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    try:
        if not lines:
            print('No lines found')
            HOUGH_IMG = line_img
            return line_img
    except: #Truth value for array with several elements is ambigous and an exception is raised
        drawLines(line_img, lines, thickness=10)
        HOUGH_IMG = line_img
        return line_img
    


def weightImage(img, initial_img, α=1., β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)


def processImage(base_img):
    global BASE_IMG, CANNY_IMG
    BASE_IMG = base_img
    image = histogram_eq(base_img)
    ysize = base_img.shape[0]
    xsize = base_img.shape[1]
    colorsize = base_img.shape[2]
    if colorsize == 1:
        image = gaussianBlur(image, 3)
    elif colorsize == 3:
        image = rgbToGray(base_img)
        image = gaussianBlur(image, 3)
    else:
        print('\n --- The dimensions of video frames is not 1 (grayscale) or 3 (color) --- \n')

    image = cannyEdgeDetection(image, 30, 130)

    CANNY_IMG = image
    image = region_of_interest(
        image,
        np.array(
            [[(40, ysize), (xsize / 2, ysize / 2 + 40), (xsize / 2, ysize / 2 + 40), (xsize - 40, ysize)]],
            dtype=np.int32
        )
    )
    image = hough_lines(image, 1, np.pi / 90, 100, 15, 10)
    #print(BASE_IMG.shape, ' - ', CANNY_IMG.shape, ' - ', HOUGH_IMG.shape)

    return weightImage(image, base_img, β=250.)


# image = mimg.imread('test3.jpg')
# new_img = processImage(image)
# new_file = 'test3Processed.jpg'
# # mplt.imshow(new_img, cmap='gray')
# # mplt.show()
# mimg.imsave(new_file,new_img)


inputfile = 'challenge2_1'
outputfile = inputfile + '_outputGRAY.mp4'
clip1 = VideoFileClip(inputfile+'.mp4')  
white_clip = clip1.fl_image(processImage)  # NOTE: this function expects color images!!
white_clip.write_videofile(outputfile, audio=False)








