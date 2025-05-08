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
from utils import *



# 1. Load the still image and the video
# still_image = cv2.imread('pair2.png', cv2.IMREAD_GRAYSCALE)
# video_path = 'pair2.mp4'
still_image = cv2.imread('media/earth1.png', cv2.IMREAD_GRAYSCALE)
video_path = 'DJI_20250507154925_0025_D.MP4'
#video_path = 'lowaltflight.mp4'

# earth_2 and vid_2 are pairs

# Image preprocess
blurred = cv2.GaussianBlur(still_image, (5,5), 0)
edges = cv2.Canny(blurred, 50, 200)
# mod_rec = drawRectangles(edges, 3, 20, 200, still_image)

# 2. Detect keypoints and descriptors in the still image using ORB
orb = cv2.ORB_create(nfeatures=450, nlevels=30)
#kp_still, des_still = orb.detectAndCompute(still_image, None)
kpts_still = orb.detect(still_image, None)
desc = cv2.xfeatures2d.BEBLID_create(1)
kp_still, des_still = desc.compute(still_image, kpts_still)

still_kps = cv2.drawKeypoints(still_image, kp_still, None, color=(0,255,0), flags=0)



# 3. Initialize the FLANN-based matcher
# index_params = dict(algorithm=6,  # FLANN_INDEX_LSH for ORB
#                     table_number=6,  # number of hash tables
#                     key_size=12,     # size of the hashed key
#                     multi_probe_level=1)  # multi-probe level
index_params = dict(algorithm=6,  # FLANN_INDEX_LSH for ORB
                    table_number=6,  # number of hash tables
                    key_size=12,     # size of the hashed key
                    multi_probe_level=1)  # multi-probe level
search_params = dict(checks=1000)  # number of checks (higher is more accurate)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# 4. Open the video file
print("cap is unopen")
cap = cv2.VideoCapture(video_path)


cluster_count = 6
ret, frame = cap.read()
fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
fps     = cap.get(cv2.CAP_PROP_FPS) or 30       # fall back if camera gives 0
h, w    = frame.shape[:2]                       # size must stay constant
out     = cv2.VideoWriter("output.mp4", fourcc, fps, (w, h))

# 5. Process each frame of the video
while cap.isOpened():
    # for i in range(30):
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Image preprocess
    blurred = cv2.GaussianBlur(gray_frame, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 200)
    # mod_rec = drawRectangles(edges, 3, 20, 200,gray_frame )

    # 6. Detect keypoints and descriptors in the frame
    #kp_frame, des_frame = orb2.detectAndCompute(gray_frame, None)
    kpts_frame = orb.detect(gray_frame, None)
    kp_frame, des_frame = desc.compute(gray_frame, kpts_frame)

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
                if m[0].distance < 0.85 * m[1].distance:
                    still_pt = kp_still[m[0].queryIdx].pt
                    frame_pt = kp_frame[m[0].trainIdx].pt
                    good_matches.append(m[0])
                    good_matches_xy['frame_x_pt'].append(frame_pt[0])
                    good_matches_xy['frame_y_pt'].append(frame_pt[1])
                    good_matches_xy['still_x_pt'].append(still_pt[0])
                    good_matches_xy['still_y_pt'].append(still_pt[1])
                    #add = np.array([still_pt[0], still_pt[1]])
                    #good_matches_xy = np.vstack((good_matches_xy, add))
                    cv2.putText(still_kps, text=str(str((int(still_pt[0]), int(still_pt[1])))), org=(int(still_pt[0]), int(still_pt[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2, lineType=cv2.LINE_AA)
        dataset = pd.DataFrame(good_matches_xy)
        # print(dataset.head())
        X = dataset.drop(columns=['frame_x_pt', 'frame_y_pt'])
        # print(X.head())
        #good_matches_xy = np.delete(good_matches_xy, 0, 0)
        # print(good_matches_xy.shape)
        kmeans = KMeans(n_clusters=cluster_count, n_init=10)
        label = kmeans.fit_predict(X)
        dataset['cluster'] = kmeans.labels_
        
        # print(X.head())
        count = Counter(kmeans.labels_)
        # print(count)
        count_list = sorted(count.items(), key=itemgetter(1), reverse=True)



        
        largest_cluster_idx = count_list[0][0]
        # print(largest_cluster_idx)
        max_centroid = kmeans.cluster_centers_[largest_cluster_idx]
        # print(max_centroid)
        y = dataset[dataset['cluster'] == largest_cluster_idx]

        # for row in y.itertuples():
        #     print(row)

        # plt.scatter(dataset['still_x_pt'], dataset['still_y_pt'], c=dataset['cluster'])
        # plt.legend()
        # plt.colorbar()
        # plt.show()
        # 9. Draw the matches
        matched_img = cv2.drawMatches(still_image, kp_still, gray_frame, kp_frame, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Show the matched image
        # cv2.imshow("Still image key points", still_kps)
        matched_img = cv2.resize(matched_img, (0,0), fx=0.5, fy=0.5) 

        cv2.imshow('Matches', matched_img)
        # out.write(matched_img)

    time.sleep(0.4)

    # Press 'q' to quit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
out.release() 
cv2.destroyAllWindows()
