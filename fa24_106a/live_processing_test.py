import cv2
import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
from operator import itemgetter

# 1. Load the still image and the video
still_image = cv2.imread('37.872312N_122.319072W_235.56H_364.08W.png', cv2.IMREAD_GRAYSCALE)
video_path = 'dji_flight.MOV'
#video_path = 'lowaltflight.mp4'

# 2. Detect keypoints and descriptors in the still image using ORB
orb = cv2.ORB_create(nfeatures=250)
#kp_still, des_still = orb.detectAndCompute(still_image, None)
kpts_still = orb.detect(still_image, None)
desc = cv2.xfeatures2d.BEBLID_create(0.75)
kp_still, des_still = desc.compute(still_image, kpts_still)

still_kps = cv2.drawKeypoints(still_image, kp_still, None, color=(0,255,0), flags=0)

# 3. Initialize the FLANN-based matcher
index_params = dict(algorithm=6,  # FLANN_INDEX_LSH for ORB
                    table_number=6,  # number of hash tables
                    key_size=12,     # size of the hashed key
                    multi_probe_level=1)  # multi-probe level
search_params = dict(checks=150)  # number of checks (higher is more accurate)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# 4. Open the video file
cap = cv2.VideoCapture(video_path)

orb2 = cv2.ORB_create(nfeatures=150)

cluster_count = 6

# 5. Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
                if m[0].distance < 0.95 * m[1].distance:
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
        print(dataset.head())
        X = dataset.drop(columns=['frame_x_pt', 'frame_y_pt'])
        print(X.head())
        #good_matches_xy = np.delete(good_matches_xy, 0, 0)
        #print(good_matches_xy.shape)
        kmeans = KMeans(n_clusters=cluster_count, n_init=10)
        #print(good_matches_xy)
        label = kmeans.fit_predict(X)
        dataset['cluster'] = kmeans.labels_
        print(dataset.head())
        
        print(X.head())
        count = Counter(kmeans.labels_)
        print(count)
        count_list = sorted(count.items(), key=itemgetter(1), reverse=True)
        largest_cluster_idx = count_list[0][0]
        print(largest_cluster_idx)
        max_centroid = kmeans.cluster_centers_[largest_cluster_idx]
        print(max_centroid)
        y = dataset[dataset['cluster'] == largest_cluster_idx]
        for row in y.itertuples():
            print(row)

        plt.scatter(dataset['still_x_pt'], dataset['still_y_pt'], c=dataset['cluster'])
        plt.legend()
        plt.colorbar()
        plt.show()
        # 9. Draw the matches
        matched_img = cv2.drawMatches(still_image, kp_still, frame, kp_frame, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Show the matched image
        cv2.imshow("Still image key points", still_kps)
        cv2.imshow('Matches', matched_img)

    time.sleep(0.4)

    # Press 'q' to quit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
