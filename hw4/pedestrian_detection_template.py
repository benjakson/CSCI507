
import cv2
  
img = cv2.imread('csm3.jpg')
assert (img is not None), 'cannot read given image'

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
  
(regions, _) = hog.detectMultiScale(img, ...)
  
# Drawing the regions in the Image
for (x, y, w, h) in regions:
    cv2.rectangle(img,...)
 
cv2.imshow("Pedestrians", img)
cv2.waitKey(0)
