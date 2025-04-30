#!/usr/bin/env python3

import cv2
import math
import numpy as np


def drawRectangles(img_gray, r, min_gap, white_thresh):
    h, w = img_gray.shape
    # convert to BGR so we can overwrite with white easily
    out = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # --- find all candidate (white) pixels -------------------------------------
    ys, xs = np.where(img_gray >= white_thresh)
    candidates = list(zip(xs, ys))          # (x, y) coordinates

    # --- greedy placement of rectangles ---------------------------------------
    centres: list[tuple[int, int]] = []     # accepted rectangle centres

    for x, y in candidates:
        # skip if too close to an existing rectangle
        if any(math.hypot(x - cx, y - cy) < min_gap for cx, cy in centres):
            continue

        # clamp rectangle to stay inside image borders
        tl = (max(0, x - r), max(0, y - r))          # top-left
        br = (min(w - 1, x + r), min(h - 1, y + r))  # bottom-right

        # cv2.rectangle(out, tl, br, color=(255, 255, 255), thickness=-1)
        cv2.circle(out, tl, 2, color=(255, 255, 255), thickness=-1)
        centres.append((x, y))

    return out




# Initialize the ORB detector
orb = cv2.ORB_create()

# Read the image
# img = cv2.imread('google_earth.png', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('earth1.png', cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(img, (5,5), 0)
edges = cv2.Canny(blurred, 50, 200)

# thresholding an image to black and white
threshold_value = 120
_, thresholded = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
mod_rec = drawRectangles(edges, 3, 20, 200)

# Find the keypoints and descriptors with ORB
kp, des = orb.detectAndCompute(img, None)
kp2, des2 = orb.detectAndCompute(edges, None)
kp3, des3 = orb.detectAndCompute(blurred, None)
kp4, des4 = orb.detectAndCompute(thresholded, None)
kp5, des5 = orb.detectAndCompute(mod_rec, None)


# Draw keypoints on the image
img_kps = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
edges_kps = cv2.drawKeypoints(edges, kp2, None, color=(0,255,0), flags=0)
blurred_kps = cv2.drawKeypoints(blurred, kp3, None, color=(0,255,0), flags=0)
thresholded_kps = cv2.drawKeypoints(thresholded, kp4, None, color=(0,255,0), flags=0)
mod_kps = cv2.drawKeypoints(mod_rec, kp5, None, color=(0,255,0), flags=0)

# Display the image
# cv2.imshow('Standard ORB Keypoints', img_kps)
# cv2.imshow('Edge ORB Keypoints', edges_kps) # this would pretty much depend on the edge detector output because it is binary, so don't use this
# cv2.imshow('Blurred ORB Keypoints', blurred_kps)
# cv2.imshow("funny thresholded image", thresholded_kps)


cv2.imshow("drawing vid", mod_kps)

cv2.waitKey(0)
cv2.destroyAllWindows()