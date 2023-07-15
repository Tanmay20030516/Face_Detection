import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

face_detector = load_model('model/face_recog.h5')

cap = cv.VideoCapture(0) # access the web cam
while cap.isOpened(): # continue the loop until the webcam is open
    _ , frame = cap.read() # read the frames
    frame = frame[60:700, 60:700,:] # crop the frames

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120,120)) # cv2 reads color as BGR

    yhat = face_detector.predict(np.expand_dims(resized/255,0))
    sample_coords = yhat[1][0] # b box coordinates

    if yhat[0] > 0.7: # confidence of detection of face
        # Controls the main rectangle
        cv.rectangle(frame,
                      tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)),
                            (255,0,0), 2)
        # Controls the label rectangle
        cv.rectangle(frame,
                      tuple(np.add(np.multiply(sample_coords[:2], [640,480]).astype(int),
                                    [0,-30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [640,480]).astype(int),
                                    [80,0])),
                            (255,0,0), -1)

        # Controls the text rendered
        cv.putText(frame,'face', tuple(np.add(np.multiply(sample_coords[:2], [640,480]).astype(int),
                                               [0,-5])),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)

    cv.imshow('Face Track', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()