import cv2
import numpy as np

# Load two grayscale images
img1 = cv2.imread('media/earth1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('ref/earth7.png', cv2.IMREAD_GRAYSCALE)


# ---------- ORB Keypoint Detection ----------
orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Match ORB descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_kp = bf.match(des1, des2)
matches_kp = sorted(matches_kp, key=lambda x: x.distance)

# Draw top 50 ORB matches
img_kp_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches_kp[:50], None, flags=2)


# ---------- Line Detection + Descriptor ----------
lsd = cv2.createLineSegmentDetector()
lines1 = lsd.detect(img1)[0]
lines2 = lsd.detect(img2)[0]

# Convert to required format for LBD

keylines1 = [cv2.KeyLine() for _ in range(len(lines1))]
keylines2 = [cv2.KeyLine() for _ in range(len(lines2))]

for i, l in enumerate(lines1):
    x1, y1, x2, y2 = l[0]
    keylines1[i].startPointX = x1
    keylines1[i].startPointY = y1
    keylines1[i].endPointX = x2
    keylines1[i].endPointY = y2
    keylines1[i].class_id = i
    keylines1[i].octave = 0
    keylines1[i].angle = np.arctan2((y2 - y1), (x2 - x1))
    keylines1[i].response = 1.0
    keylines1[i].size = np.hypot(x2 - x1, y2 - y1)
    keylines1[i].pt = ((x1 + x2) / 2, (y1 + y2) / 2)

# Do the same for img2
for i, l in enumerate(lines2):
    x1, y1, x2, y2 = l[0]
    keylines2[i].startPointX = x1
    keylines2[i].startPointY = y1
    keylines2[i].endPointX = x2
    keylines2[i].endPointY = y2
    keylines2[i].class_id = i
    keylines2[i].octave = 0
    keylines2[i].angle = np.arctan2((y2 - y1), (x2 - x1))
    keylines2[i].response = 1.0
    keylines2[i].size = np.hypot(x2 - x1, y2 - y1)
    keylines2[i].pt = ((x1 + x2) / 2, (y1 + y2) / 2)

# Compute descriptors with LBD
lbd = cv2.ximgproc.createBinaryDescriptor()
lbd_desc1 = lbd.compute(img1, keylines1)
lbd_desc2 = lbd.compute(img2, keylines2)

# Match line descriptors
bf_line = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_line = bf_line.match(lbd_desc1, lbd_desc2)
matches_line = sorted(matches_line, key=lambda x: x.distance)

# Draw top 30 line matches
img_line_matches = cv2.drawMatches(img1, keylines1, img2, keylines2, matches_line[:30], None, flags=2)

# ---------- Display ----------
cv2.imshow("ORB Matches", img_kp_matches)
cv2.imshow("Line Matches (LSD + LBD)", img_line_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
