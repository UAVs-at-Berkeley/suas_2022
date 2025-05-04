import cv2
import math
import time

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
from operator import itemgetter


still_image_dict = {
    0:('37.872310N_122.322454W_231.23H_297.8W.png', 37.872310, 122.322454, 231.23, 297.8), 
    # 1:('pair1.png', 37.872312, 122.319072, 170.3, 318), 
    1:('pair2.png', 37.8722765, 122.3193286, 279.09, 318), 
    2:('37.874496H_122.322454W_242.73H_297.8W.png', 37.874496, 122.322454, 242.73, 297.8), 
    3:('37.874496N_122.319072W_242.73H_364.08W.png', 37.874496, 122.319072, 242.73, 364.08)
}


r_earth = 6378000

# Video File
video_path = "pair2.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
# print("frame", frame.shape[1])


# TODO: Replace this part with google map video, fill out starting 
# starting:  
# ending: 
drone_alt = 30
# last_prediction_lat = 37.872312
# last_prediction_lon = 122.319072
last_prediction_lat = 37.8714191 #vehicle.home_location.lat
last_prediction_lon = 122.3171289 # vehicle.home_location.lon
# h_fov = 71.5
# d_fov = 79.5
cam_size = (frame.shape[1], frame.shape[0])
droner_lat = 0
droner_lon = 0

cam_x = 318
cam_y = 170.3
# print(cam_y)
cam_x_size = cam_x / cam_size[0]
# print(cam_x_size)
cam_y_size = cam_y / cam_size[1]
# print(cam_y_size)


def get_distance_metres_pts(aLocation1, aLocation2):
    """
    Returns the ground distance in metres between two `LocationGlobal` or `LocationGlobalRelative` objects.

    This method is an approximation, and will not be accurate over large distances and close to the
    earth's poles. It comes from the ArduPilot test code:
    https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
    """
    dlat = aLocation2.lat - aLocation1.lat
    dlong = aLocation2.lon - aLocation1.lon
    return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5

def get_distance_metres(lat1, lon1, lat2, lon2):
    """
    Returns the ground distance in metres between two `LocationGlobal` or `LocationGlobalRelative` objects.

    This method is an approximation, and will not be accurate over large distances and close to the
    earth's poles. It comes from the ArduPilot test code:
    https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
    """
    dlat = lat2 - lat1
    dlong = lon2 - lon1
    return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5


# 1. Load the still image and the video
still_image = cv2.imread(still_image_dict[1][0], cv2.IMREAD_GRAYSCALE)
# video_path = 'google_movie.mp4'
horizontal_size = still_image.shape[:2][1]
print(horizontal_size)
vertical_size = still_image.shape[:2][0]
print(vertical_size)
x_size = still_image_dict[1][4] / horizontal_size
print(x_size)
y_size = still_image_dict[1][3] / vertical_size
print(y_size)

# 2. Detect keypoints and descriptors in the still image using ORB
orb = cv2.ORB_create(nfeatures=200)
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
search_params = dict(checks=100)  # number of checks (higher is more accurate)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# 4. Open the video file
# rtsp_url = "rtsp://192.168.144.25:8554/main.264"


print(cap.get(3))
print(cap.get(4))

orb2 = cv2.ORB_create(nfeatures=150)

cluster_count = 6
total_dist_traveled = 0

vid_matches = cv2.VideoWriter('vid_matches.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, (int(horizontal_size+cap.get(3)), int(max([vertical_size, cap.get(4)]))))
ct = 0
# 5. Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()

    ct += 1
    #ret = cap.grab()
    if (ct >= 20):
        ct = 0
        # ret, frame = cap.read()
        if not ret:
            break

        # Drone vision edit
        drone_alt = 387

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 6. Detect keypoints and descriptors in the frame
        #kp_frame, des_frame = orb2.detectAndCompute(gray_frame, None)
        kpts_frame = orb.detect(gray_frame, None)
        kp_frame, des_frame = desc.compute(gray_frame, kpts_frame)

        frame_kps = cv2.drawKeypoints(gray_frame, kp_frame, None, color=(0,255,0), flags=0)

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
                        
                        good_matches_xy['frame_x_pt'].append(frame_pt[0])
                        good_matches_xy['frame_y_pt'].append(frame_pt[1])
                        good_matches_xy['still_x_pt'].append(still_pt[0])
                        good_matches_xy['still_y_pt'].append(still_pt[1])
                        
                        cv2.putText(still_kps, text=str(str((int(still_pt[0]), int(still_pt[1])))), org=(int(still_pt[0]), int(still_pt[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2, lineType=cv2.LINE_AA)
                        cv2.putText(frame_kps, text=str((int(frame_pt[0]), int(frame_pt[1]))), org=(int(frame_pt[0]), int(frame_pt[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2, lineType=cv2.LINE_AA)
                        good_matches.append(m[0])
            
            # print("good_matches: ", good_matches)
            dataset = pd.DataFrame(good_matches_xy)
            X = dataset.drop(columns=['frame_x_pt', 'frame_y_pt'])
            kmeans = KMeans(n_clusters=cluster_count, n_init=10)
            label = kmeans.fit_predict(X)
            dataset['cluster'] = kmeans.labels_
            count = Counter(kmeans.labels_)
            # print("count: ", count)
            print("cam_x"+str(cam_x))
            print("cam_y"+str(cam_y))


            plt.scatter(dataset['still_x_pt'], dataset['still_y_pt'], c=dataset['cluster'])
            plt.legend()
            plt.colorbar()
            plt.show()

            count_list = sorted(count.items(), key=itemgetter(1), reverse=True)
            print("count list: ")
            y = dataset[dataset['cluster'] ==  count_list[2][0]]
            for row in y.itertuples():
                cluster_matches.append(good_matches[row.Index])
            matched_img_1 = cv2.drawMatches(still_image, kp_still, gray_frame, kp_frame, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            cv2.imshow("matches", matched_img_1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            #print(good_matches_xy)
            # print(sorted(count.items(), key=itemgetter(1), reverse=True))
            # print(sorted(count.items(), key=itemgetter(1), reverse=True))

            # This finds the centroid of the image position and rotation
            centroid_gps_lat = 0
            centroid_gps_long = 0
            centroid_cluster_idx = 0
            medians = dataset.groupby('cluster').median()
            # print("medians: ", medians)

            # TODO: The med_dist_to_last_pt is too big. Get a better initial estimation
            for i in range(0, 4):
                largest_cluster_idx = count_list[i][0]
                print("largest cluster ", largest_cluster_idx)
                max_centroid = kmeans.cluster_centers_[largest_cluster_idx]
                max_median = (medians.at[largest_cluster_idx, 'still_x_pt'], medians.at[largest_cluster_idx, 'still_y_pt'])
                
                # print("max centroid ", max_centroid)
                # print("max median ",max_median)
                # print("still_image_dict", still_image_dict[1][1], " ", still_image_dict[1][2])
                centroid_gps_lat_temp = still_image_dict[1][1] - (max_centroid[1]*y_size / r_earth) * (180 / math.pi)
                centroid_gps_long_temp = still_image_dict[1][2] - ((max_centroid[0]*x_size / r_earth) * (180 / math.pi) / math.cos(still_image_dict[1][1]*math.pi/180))
                
                # print("still_image_dict", centroid_gps_lat_temp, " ", centroid_gps_long_temp)
                # print("still_image_dict", last_prediction_lat, " ", last_prediction_lon)
                
                
                dist_to_last_pt = get_distance_metres(centroid_gps_lat_temp, centroid_gps_long_temp, last_prediction_lat, last_prediction_lon)
                # print(dist_to_last_pt)
                median_gps_lat_temp = still_image_dict[1][1] - (max_median[1]*y_size / r_earth) * (180 / math.pi)
                median_gps_long_temp = still_image_dict[1][2] - ((max_median[0]*x_size / r_earth) * (180 / math.pi) / math.cos(still_image_dict[1][1]*math.pi/180))
                med_dist_to_last_pt = get_distance_metres(median_gps_lat_temp, median_gps_long_temp, last_prediction_lat, last_prediction_lon)

                print("dist", med_dist_to_last_pt)
                if med_dist_to_last_pt < 100:
                    # print("passed med_dist_to_last_point")
                    print((centroid_gps_lat_temp, centroid_gps_long_temp))
                    print((median_gps_lat_temp, median_gps_long_temp))
                    centroid_gps_lat = median_gps_lat_temp
                    centroid_gps_long = median_gps_long_temp
                    centroid_cluster_idx = largest_cluster_idx
                    break
            

            matched_img_1 = cv2.drawMatches(still_image, kp_still, gray_frame, kp_frame, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            cv2.imshow("matches", matched_img_1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if centroid_gps_lat == 0 or centroid_gps_long == 0:
                print("couldnt find centroid")
                continue


            cam_gps_long = 0
            cam_gps_lat = 0

            y = dataset[dataset['cluster'] ==  count_list[2][0]]
            best_match_idx = 0
            counter = 0
            cam_gps_lat_sum = 0
            cam_gps_long_sum = 0
            cluster_matches = []
            for row in y.itertuples():
                #print(row)
                x_lat = still_image_dict[1][1] - ((row.still_y_pt)*y_size / r_earth) * (180 / math.pi)
                x_long = still_image_dict[1][2] - (((row.still_x_pt)*x_size / r_earth) * (180 / math.pi) / math.cos(still_image_dict[1][1]*math.pi/180))
                dist_to_centroid = get_distance_metres(x_lat, x_long, centroid_gps_lat, centroid_gps_long)
                print("distance to centroid for ", row,  "  : ", dist_to_centroid)
                if dist_to_centroid < 10:
                    still_gps_lat = x_lat
                    still_gps_long = x_long
                    print("gps_coor", (x_lat, x_long))
                    print((((row.frame_y_pt) - cam_size[1]/2)*cam_y_size))
                    print((((row.frame_x_pt) - cam_size[0]/2)*cam_x_size))
                    best_match_idx = row.Index
                    cluster_matches.append(good_matches[row.Index])
                    cam_gps_lat = still_gps_lat - ((((row.frame_y_pt) - cam_size[1]/2)*cam_y_size)/ r_earth) * (180 / math.pi)
                    cam_gps_long = still_gps_long + ((((row.frame_x_pt) - cam_size[0]/2)*cam_x_size) / r_earth) * (180 / math.pi) / math.cos(still_gps_lat*math.pi/180)
                    cam_gps_lat_sum += cam_gps_lat
                    cam_gps_long_sum += cam_gps_long
                    counter+=1
                    #break
            # TODO: The error is showing at the end
            if cam_gps_lat == 0 or cam_gps_long == 0:
                print("exiting")
                continue

            # TODO: Change this so that the error is only shown at the end
            # Get the final and initial position from google maps
            # get the estimated position that results from the orb matching

            # cam_gps_lat = cam_gps_lat_sum / counter
            # cam_gps_long = cam_gps_long_sum / counter

            gps_dist_traveled = get_distance_metres(cam_gps_lat, cam_gps_long, last_prediction_lat, last_prediction_lon)

            print(gps_dist_traveled)
            total_dist_traveled += gps_dist_traveled



            # 9. Draw the matches
            matched_img = cv2.drawMatches(still_image, kp_still, gray_frame, kp_frame, cluster_matches, None, flags=cv2.DrawMatchesFlags_DEFAULT)

            # Show the matched image
            vid_matches.write(matched_img)
            last_prediction_lat = cam_gps_lat
            last_prediction_lon = cam_gps_long

            # print("hello")
            cv2.imshow("matches", matched_img)
            # Press 'q' to quit the video
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
print(total_dist_traveled)
# Release the video capture and close the window
vid_matches.release()
cap.release()
cv2.destroyAllWindows()
