import cv2
import numpy as np


def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

img = cv2.imread('test4.jpg')

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

edges = auto_canny(output_gray)
#median = cv2.medianBlur(output_gray,5)
(cnts,_) = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for c in cnts:
	arcLength = cv2.arcLength(c,True)
	cv2.drawContours(img, c, -1, (0,255,0), 2,maxLevel=0)
	"""if arcLength < 20.0:
		print arcLength
		cv2.drawContours(img, c, 0, (0,255,0), 2,maxLevel=1000)"""

cv2.imshow('Output GRAY',output_gray)
cv2.imshow('Canny',edges)
cv2.imshow('Output image',img)

cv2.waitKey()


cv2.destroyAllWindows()
