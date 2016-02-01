import cv2
import numpy as np


img = cv2.imread('test5.jpg')

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower mask (0-10)
lower_red = np.array([0,50,50])
upper_red = np.array([1,255,255])
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

# join my masks
mask = mask0+mask1

# set my output img to zero everywhere except my mask
output_img = img.copy()
output_img[np.where(mask==0)] = 0

# or your HSV image, which I *believe* is what you want
output_hsv = img_hsv.copy()
output_hsv[np.where(mask==0)] = 0

output_gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(output_gray,100,255,cv2.THRESH_BINARY)
output_thresh = cv2.dilate(thresh,None,iterations=6)

(cnts,_) = cv2.findContours(output_thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for c in cnts:
	contourArea = cv2.contourArea(c,True)
	#print abs(contourArea)
	if abs(contourArea) > 2000:
		print contourArea
		cv2.drawContours(img, c, -1, (0,255,0), 2,maxLevel=1)

#cv2.drawContours(img, cnts, -1, (0,255,0), 1,maxLevel=1)

cv2.imshow('Output GRAY',output_gray)
cv2.imshow('Output threshold',output_thresh)
cv2.imshow('Output ',img)

cv2.waitKey()


cv2.destroyAllWindows()
