#!/usr/bin/env python3

import cv2

# Initialize the ORB detector
orb = cv2.ORB_create()

# Read the image
img = cv2.imread('google_earth_2.png', cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, 50, 200)
blurred = cv2.GaussianBlur(img, (5,5), 0)

# thresholding an image to black and white
threshold_value = 100
_, thresholded = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

# Find the keypoints and descriptors with ORB
kp, des = orb.detectAndCompute(img, None)
kp2, des2 = orb.detectAndCompute(edges, None)
kp3, des3 = orb.detectAndCompute(blurred, None)
kp4, des4 = orb.detectAndCompute(thresholded, None)


# Draw keypoints on the image
img_kps = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
edges_kps = cv2.drawKeypoints(edges, kp2, None, color=(0,255,0), flags=0)
blurred_kps = cv2.drawKeypoints(blurred, kp3, None, color=(0,255,0), flags=0)
thresholded_kps = cv2.drawKeypoints(thresholded, kp4, None, color=(0,255,0), flags=0)

# Display the image
cv2.imshow('Standard ORB Keypoints', img_kps)
cv2.imshow('Edge ORB Keypoints', edges_kps) # this would pretty much depend on the edge detector output because it is binary, so don't use this
cv2.imshow('Blurred ORB Keypoints', blurred_kps)
cv2.imshow("funny thresholded image", thresholded_kps)
cv2.waitKey(0)
cv2.destroyAllWindows()