import cv2
import numpy as np

img = cv2.imread('csm1.jpg')
assert (img is not None), 'cannot read given image'

img_gray = ...

# trained data: https://github.com/opencv/opencv/tree/master/data/haarcascades
haar_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces = haar_face.detectMultiScale(img_gray, ...) 
for (x, y, w, h) in faces: 
    cv2.rectangle(img, ...) 

cv2.imshow('Faces', img) 
cv2.waitKey(0) 


