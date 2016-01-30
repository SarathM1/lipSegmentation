import cv2
import numpy as np

im = cv2.imread('test2.jpg')


# Convert BGR to HSV
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_blue = np.array([-5,50,50])
upper_blue = np.array([5,255,255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(im,im, mask= mask)
res2 = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
dilate = cv2.dilate(res2,None,iterations=5)
(cnts,_) = cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#(x,y,w,h)  = cv2.boundingRect(cnts[0])
#cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.drawContours(im, cnts, -1, (0,255,0), 2,maxLevel=0)
print cv2.arcLength(cnts[0],True)

cv2.imshow('res',res2)
cv2.imshow('Output',im)
cv2.imshow('mask',dilate)

cv2.imwrite('output2.jpg',im)
cv2.waitKey()


cv2.destroyAllWindows()
