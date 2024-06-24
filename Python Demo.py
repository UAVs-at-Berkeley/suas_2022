import cv2
import numpy as np

x = cv2.imread("./tests/colorful.png")

gray_image = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)

#contours, _ = cv2.findContours(gray_image, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#print(cv2.contourArea(contours[0]))

mask = cv2.inRange(x, np.array([0, 100, 100]), np.array([60, 120, 120])) 

cv2.imshow("image", mask)



cv2.waitKey(0)

cv2.destroyAllWindows()
