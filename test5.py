import cv2
import numpy as np

img=cv2.imread("test4.jpg")
img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower mask (0-10)
lower_red = np.array([0,50,50])
upper_red = np.array([5,255,255])
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([180,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

# join my masks
mask = mask0+mask1

res = cv2.bitwise_and(img,img, mask= mask)

# set my output img to zero everywhere except my mask
output_img = img.copy()
output_img[np.where(mask==0)] = 0

# or your HSV image, which I *believe* is what you want
output_hsv = img_hsv.copy()
output_hsv[np.where(mask==0)] = 0

cv2.imshow("ORiginal",img)
cv2.imshow("HSV",output_hsv)
cv2.imshow("RGB",output_img)

cv2.waitKey()


cv2.destroyAllWindows()
