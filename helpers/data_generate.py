import os # file handling
import time
import uuid # uniform unique ID (used to give name to images)
import cv2 as cv # reading images

IMAGES_PATH = os.path.join('data','images')
number_images = 30

cap = cv.VideoCapture(0)
for imgnum in range(number_images):
    print('Collecting image {}'.format(imgnum+1))
    ret, frame = cap.read()
    imgname = os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')
    cv.imwrite(imgname, frame)
    cv.imshow('frame', frame)
    time.sleep(0.5) # time window between captures

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

# `labelme` run this command in terminal
# a new window launches, where we have to annotate the images we captured
# open the images directory
# change the output directory to labels
# select the `make rectangle boxes tool` and create bounding boxes for images
# name the bounding box a class and save
# repeat for all images
# where no face is detected, leave the image as it is