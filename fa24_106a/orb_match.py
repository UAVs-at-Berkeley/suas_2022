import cv2
import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
from operator import itemgetter
import math

# img_gray is the processed image
# r is radius of the shape
# min_gap is distance between each shape
# white_thresh is the threshold of whiteness
# drawing_img is the image at which is being drawn
def drawRectangles(img_gray, r, min_gap, white_thresh, drawing_img):
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

        # cv2.rectangle(drawing_img, tl, br, color=(0, 0, 0), thickness=-1)
        cv2.circle(drawing_img, tl, 2, color=(0, 0, 0), thickness=-1)
        # cv2.circle(drawing_img, tl, 2, color=(255, 255, 255), thickness=-1)
        centres.append((x, y))

    return drawing_img


# 1. Load the still image and the video
still_image = cv2.imread('dji_screenshot.png', cv2.IMREAD_GRAYSCALE)
still_image2 = cv2.imread('ref/earth7.png', cv2.IMREAD_GRAYSCALE)

# Image preprocess
blur = cv2.GaussianBlur(still_image, (13,13), 0)
edges = cv2.Canny(blur, 50, 200)
mod_rec = drawRectangles(edges, 2, 20, 150, still_image)

# # Adjust the brightness and contrast 
# # Adjusts the brightness by adding 10 to each pixel value 
# brightness = 50
# # Adjusts the contrast by scaling the pixel values by 2.3 
# contrast = 2.3  
# still_image = cv2.addWeighted(still_image, contrast, np.zeros(still_image.shape, still_image.dtype), 0, brightness) 


# Makes lane markings, building edges, and tree trunks stand out against the fields without blowing out highlights.
# clahe = cv2.createCLAHE(clipLimit=0.75, tileGridSize=(8,8)) 
# still_image = clahe.apply(still_image)


# Image preprocess 2
blurred_2 = cv2.GaussianBlur(still_image2, (5,5), 0)
edges_2 = cv2.Canny(blurred_2, 50, 200)
mod_rec_2 = drawRectangles(edges_2, 2, 20, 150, still_image2)

# still_image2 = clahe.apply(still_image2)


# 2. Detect keypoints and descriptors in the still image using ORB
orb = cv2.ORB_create(   nfeatures      = 500,    # more keypoints
                        scaleFactor    = 1.2,     # finer image pyramid
                        nlevels        = 2,
                        edgeThreshold  = 20,      # detect closer to borders
                        fastThreshold  = 7)       # lower ⇒ pick weaker corners


# 6. Detect keypoints and descriptors in the frame

kpts_still = orb.detect(still_image, None)
desc = cv2.xfeatures2d.BEBLID_create(0.75)
kp_still, des_still = desc.compute(still_image, kpts_still)
print(kp_still)
still_kps = cv2.drawKeypoints(still_image, kp_still, None, color=(0,255,0), flags=0)


kpts_frame = orb.detect(still_image2, None)
kp_frame, des_frame = desc.compute(still_image2, kpts_frame)
frame_kps = cv2.drawKeypoints(still_image2, kp_frame, None, color=(0,255,0), flags=0)
print(kp_frame)


# 3. Initialize the FLANN-based matcher
index_params = dict(algorithm=6,  # FLANN_INDEX_LSH for ORB
                    table_number=6,  # number of hash tables
                    key_size=12,     # size of the hashed key
                    multi_probe_level=1)  # multi-probe level
search_params = dict(checks=1000)  # number of checks (higher is more accurate)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# 4. Open the video file




# 7. Match descriptors using FLANN
if des_frame is not None:
    matches = flann.knnMatch(des_still, des_frame, k=2)

    # 8. Apply the ratio test to filter matches (Lowe's ratio test)
    good_matches = []
    good_matches_xy = {
        'still_x_pt':[],
        'still_y_pt':[],
        'frame_x_pt':[],
        'frame_y_pt':[]

    }
    for m in matches:
        if len(m) == 2:  # Ensure that we have two matches
            # Apply Lowe's ratio test
            if m[0].distance < 0.55 * m[1].distance:
                still_pt = kp_still[m[0].queryIdx].pt
                frame_pt = kp_frame[m[0].trainIdx].pt
                good_matches.append(m[0])
                good_matches_xy['frame_x_pt'].append(frame_pt[0])
                good_matches_xy['frame_y_pt'].append(frame_pt[1])
                good_matches_xy['still_x_pt'].append(still_pt[0])
                good_matches_xy['still_y_pt'].append(still_pt[1])
                #add = np.array([still_pt[0], still_pt[1]])
                #good_matches_xy = np.vstack((good_matches_xy, add))
                # cv2.putText(still_kps, text=str(str((int(still_pt[0]), int(still_pt[1])))), org=(int(still_pt[0]), int(still_pt[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2, lineType=cv2.LINE_AA)
    dataset = pd.DataFrame(good_matches_xy)
    print(good_matches)

    # for row in y.itertuples():
    #     print(row)

    # plt.scatter(dataset['still_x_pt'], dataset['still_y_pt'], c=dataset['cluster'])
    # plt.legend()
    # plt.colorbar()
    # plt.show()
    # 9. Draw the matches
    matched_img = cv2.drawMatches(still_kps, kp_still, frame_kps, kp_frame, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show the matched image
    # cv2.imshow("Still image key points", still_kps)
    cv2.imshow('Matches', matched_img)
    # out.write(matched_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
