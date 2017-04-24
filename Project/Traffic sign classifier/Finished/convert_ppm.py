from PIL import Image
import os.path
import csv
import cv2

origin_folder = 'images/'
new_folder = 'train/'


for c in range(0,43):
    prefix = origin_folder + '/' + format(c, '05d') + '/' # subdirectory for class
    #prefix = origin_folder
    prefix_new = new_folder + '/' + format(c, '05d') + '/' # subdirectory for class
    #prefix_new = new_folder
    gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
    #gtFile = open(prefix + 'GT-final_test_labels.csv')
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    next(gtReader) # skip header

    if not os.path.exists(prefix_new):
      os.makedirs(prefix_new)
    # create a CLAHE object (Arguments are optional).
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # loop over all images in current annotations file
    for row in gtReader:
        classID = row[7]
        #if not os.path.exists(prefix_new + classID.zfill(5)):
        #    os.makedirs(prefix_new + classID.zfill(5))
        #img = plt.imread(prefix + row[0])
        #images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
        #labels.append(row[7]) # the 8th column is the label
        #str(1).zfill(2)
        filename    = prefix + row[0]
        #filenameJPG = prefix_new +classID.zfill(5) +'/'+ row[0].strip('.ppm')+'.jpg'
        filenameJPG = prefix_new + row[0].strip('.ppm')+'.jpg'

        img = cv2.imread(filename,0)
        cl1 = clahe.apply(img)
        cv2.imwrite(filenameJPG,cl1)
        #im = Image.open(filename)
        #im.save(filenameJPG)
    gtFile.close()
