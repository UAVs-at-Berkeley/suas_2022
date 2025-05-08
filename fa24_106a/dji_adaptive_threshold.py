import cv2
import math
import time
import statistics
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
from operator import itemgetter
from utils import *


#######****************************************************************##########

#######   Uses newer images                                             ##########

#######****************************************************************##########

still_image_dict = ('DJI_20250507160257_0026_D_2.png', 37.8714865, 122.3183067, 97, 128)


# 2. Detect keypoints and descriptors in the still image using ORB
orb = cv2.ORB_create(nfeatures=1000)
desc = cv2.xfeatures2d.BEBLID_create(0.75)


# 3. Initialize the FLANN-based matcher
index_params = dict(algorithm=6,  # FLANN_INDEX_LSH for ORB
                    table_number=6,  # number of hash tables
                    key_size=12,     # size of the hashed key
                    multi_probe_level=1)  # multi-probe level
search_params = dict(checks=50)  # number of checks (higher is more accurate)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# 4. Open the video file
video_path = 'DJI_20250507160257_0026_D.MP4'
cap = cv2.VideoCapture(video_path)

parsed = parse_metadata("DJI_20250507160257_0026_D.SRT")
df = pd.DataFrame(parsed)

r_earth = 6378000

initial_pos = (37.87161441993102, 122.31752593436721)

drone_alt = df['abs_alt'].iloc[0]
drone_lat = df['latitude'].iloc[0]
drone_lon = df['longitude'].iloc[0]
last_prediction_lat = drone_lat
last_prediction_lon = -1 * drone_lon
prev_angle = 0
h_fov = 71.5
d_fov = 79.5
cam_size = (1920, 1080)
cam_x = 2*(math.tan(h_fov*math.pi/2/180)*drone_alt)
print(cam_x)
cam_diag = 2*(math.tan(d_fov*math.pi/2/180)*drone_alt)
half_cam_diag = cam_diag/2
print(half_cam_diag)
cam_y = math.sqrt(4*((math.tan(79.5*math.pi/2/180))**2)*(drone_alt**2)-(cam_x**2))
print(cam_y)
cam_x_size = cam_x / cam_size[0]
print(cam_x_size)
cam_y_size = cam_y / cam_size[1]
print(cam_y_size)


# 1. Iterate through all images to find the best fit
still_image = cv2.imread(still_image_dict[0], cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(still_image, (5,5), 0)
# edges = cv2.Canny(blur, 50, 200)
# still_image = cv2.equalizeHist(still_image)

# mod_rec = drawRectangles(edges, 4, 20, 150, still_image)
horizontal_size = still_image.shape[:2][1]
# print(horizontal_size)
vertical_size = still_image.shape[:2][0]
# print(vertical_size)
x_size = still_image_dict[4] / horizontal_size
# print(x_size)
y_size = still_image_dict[3] / vertical_size
# print(y_size)

enhancement = truncated_adaptive_gamma(blur)
alpha = 1
beta = 0.3
G_mask = still_image - blur
i_unsharp = still_image + alpha * G_mask
gray_frame = blur + alpha*G_mask + beta * enhancement 
still_image = np.clip(gray_frame, 0, 255).astype(np.uint8)
# print("enhanced gray frame: ", gray_frame.shape)


kpts_still = orb.detect(still_image, None)
kp_still, des_still = desc.compute(still_image, kpts_still)
still_kps = cv2.drawKeypoints(still_image, kp_still, None, color=(0,255,0), flags=0)
# cv2.imshow("stil image", still_kps)

print(cap.get(3))
print(cap.get(4))


cluster_count = 6
total_dist_traveled = 0

vid_matches = cv2.VideoWriter('vid_matches_adaptive_threshold.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 20, (int(horizontal_size+cap.get(3)), int(max([vertical_size, cap.get(4)]))))
vid_location = cv2.VideoWriter('vid_location_adaptive_threshold.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 20, (still_image.shape[1], still_image.shape[0]))

ct = 0
i = 1
# 5. Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()

    ct += 1
    #ret = cap.grab()
    if (ct >= 5):
        ct = 0
        #ret, frame = cap.read()
        if not ret:
            break
        
        matches_xy_list = []
        matches_list = []
        angles_list = []

        cam_x = 2*(math.tan(h_fov*math.pi/2/180)*drone_alt)
        cam_diag = 2*(math.tan(d_fov*math.pi/2/180)*drone_alt)
        half_cam_diag = cam_diag/2
        cam_y = math.sqrt(4*((math.tan(79.5*math.pi/2/180))**2)*(drone_alt**2)-(cam_x**2))
        cam_x_size = cam_x / cam_size[0]
        cam_y_size = cam_y / cam_size[1]

        # lab= cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        # l_channel, a, b = cv2.split(lab)
        # clahe = cv2.createCLAHE(clipLimit=.75, tileGridSize=(10,10)) 
        # cl = clahe.apply(l_channel)
        # # merge the CLAHE enhanced L-channel with the a and b channel
        # limg = cv2.merge((cl,a,b))

        # # Converting image from LAB Color model to BGR color spcae
        # enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # # Stacking the original image with the enhanced image
        # frame = np.hstack((frame, enhanced_img))


        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Image preprocess
        blur = cv2.GaussianBlur(gray_frame, (13,13), 0)
        # edges = cv2.Canny(gray_frame, 50, 150)
        # gray_frame = cv2.equalizeHist(gray_frame)
        # drawRectangles(edges, 4, 20, 100, gray_frame)
        enhancement = truncated_adaptive_gamma(blur)
        alpha = 1
        beta = 0.3
        G_mask = gray_frame - blur
        i_unsharp = gray_frame + alpha * G_mask
        gray_frame = blur + alpha*G_mask + beta * enhancement 
        gray_frame = np.clip(gray_frame, 0, 255).astype(np.uint8)
        print("enhanced gray frame: ", gray_frame.shape)
       


        kpts_frame = orb.detect(gray_frame, None)
        kp_frame, des_frame = desc.compute(gray_frame, kpts_frame)
        frame_kps = cv2.drawKeypoints(gray_frame, kp_frame, None, color=(0,255,0), flags=0)

        # 7. Match descriptors using FLANN
        if des_frame is not None:
            matches = flann.knnMatch(des_still, des_frame, k=2)

            # 8. Apply the ratio test to filter matches (Lowe's ratio test)
            good_matches = []
            good_angles = []
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
    
                        
                        good_matches_xy['frame_x_pt'].append(frame_pt[0])
                        good_matches_xy['frame_y_pt'].append(frame_pt[1])
                        good_matches_xy['still_x_pt'].append(still_pt[0])
                        good_matches_xy['still_y_pt'].append(still_pt[1])
                        
                        cv2.putText(still_kps, text=str(str((int(still_pt[0]), int(still_pt[1])))), org=(int(still_pt[0]), int(still_pt[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2, lineType=cv2.LINE_AA)
                        cv2.putText(frame_kps, text=str((int(frame_pt[0]), int(frame_pt[1]))), org=(int(frame_pt[0]), int(frame_pt[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2, lineType=cv2.LINE_AA)
                        good_matches.append(m[0])

                        still_angle = kp_still[m[0].queryIdx].angle
                        frame_angle = kp_frame[m[0].trainIdx].angle
                        angle = still_angle - frame_angle
                        good_angles.append(angle)
            
            dataset = pd.DataFrame(good_matches_xy)
            X = dataset.drop(columns=['frame_x_pt', 'frame_y_pt'])
            kmeans = KMeans(n_clusters=cluster_count, n_init=10)
            label = kmeans.fit_predict(X)
            dataset['cluster'] = kmeans.labels_
            count = Counter(kmeans.labels_)
            # print("cam_x"+str(cam_x))
            # print("cam_y"+str(cam_y))
            #print(good_matches_xy)
            # print(sorted(count.items(), key=itemgetter(1), reverse=True))
            count_list = sorted(count.items(), key=itemgetter(1), reverse=True)
            centroid_gps_lat = 0
            centroid_gps_long = 0
            centroid_cluster_idx = 0
            medians = dataset.groupby('cluster').median()
            for i in range(0, 3):
                largest_cluster_idx = count_list[i][0]
                print(largest_cluster_idx)
                max_centroid = kmeans.cluster_centers_[largest_cluster_idx]
                max_median = (medians.at[largest_cluster_idx, 'still_x_pt'], medians.at[largest_cluster_idx, 'still_y_pt'])
                centroid_gps_lat_temp = still_image_dict[1] - (max_centroid[1]*y_size / r_earth) * (180 / math.pi)
                centroid_gps_long_temp = still_image_dict[2] - ((max_centroid[0]*x_size / r_earth) * (180 / math.pi) / math.cos(still_image_dict[1]*math.pi/180))
                dist_to_last_pt = get_distance_metres(centroid_gps_lat_temp, centroid_gps_long_temp, last_prediction_lat, last_prediction_lon)
                # print(dist_to_last_pt)
                median_gps_lat_temp = still_image_dict[1] - (max_median[1]*y_size / r_earth) * (180 / math.pi)
                median_gps_long_temp = still_image_dict[2] - ((max_median[0]*x_size / r_earth) * (180 / math.pi) / math.cos(still_image_dict[1]*math.pi/180))
                med_dist_to_last_pt = get_distance_metres(median_gps_lat_temp, median_gps_long_temp, last_prediction_lat, last_prediction_lon)
                print("centroid", (centroid_gps_lat_temp, centroid_gps_long_temp))
                print("median: ", (median_gps_lat_temp, median_gps_long_temp))
                print("last_prediction: ", (last_prediction_lat, last_prediction_lon))
                print("med_dist_to_last_pt: ", med_dist_to_last_pt)
                if med_dist_to_last_pt < 50:
                    print((centroid_gps_lat_temp, centroid_gps_long_temp))
                    print((median_gps_lat_temp, median_gps_long_temp))
                    centroid_gps_lat = median_gps_lat_temp
                    centroid_gps_long = median_gps_long_temp
                    centroid_cluster_idx = largest_cluster_idx
                    break
            if centroid_gps_lat == 0 or centroid_gps_long == 0:
                # last_prediction_lat = 
                print("exiting clusters")
                continue


            cam_gps_long = 0
            cam_gps_lat = 0

            y = dataset[dataset['cluster'] == largest_cluster_idx]
            best_match_idx = 0
            counter = 0
            cam_gps_lat_sum = 0
            cam_gps_long_sum = 0
            cluster_matches = []
            angle_sum = 0

            # Compare to the frame to the still image
            # print("dataset: ", dataset)
            for row in dataset.itertuples():
                x_lat = still_image_dict[1] - ((row.still_y_pt)*y_size / r_earth) * (180 / math.pi)
                x_long = still_image_dict[2] - (((row.still_x_pt)*x_size / r_earth) * (180 / math.pi) / math.cos(still_image_dict[1]*math.pi/180))
                dist_to_centroid = get_distance_metres(x_lat, x_long, centroid_gps_lat, centroid_gps_long)
                print(dist_to_centroid)
                if dist_to_centroid < 20:
                    still_gps_lat = x_lat
                    still_gps_long = x_long
                    # print((x_lat, x_long))
                    # print((((row.frame_y_pt) - cam_size[1]/2)*cam_y_size))
                    # print((((row.frame_x_pt) - cam_size[0]/2)*cam_x_size))
                    # assert frame.shape[0:2] == (cam_size[1], cam_size[0])
                    frame_y = row.frame_y_pt - cam_size[1]/2
                    frame_x = row.frame_x_pt - cam_size[0]/2
                    best_match_idx = row.Index
                    cluster_matches.append(good_matches[row.Index])
                    # finding angles and rotation
                    x_angles = good_angles[row.Index] # in degrees
                    x_angles = x_angles * np.pi/180
                    rotation_matrix = np.array([
                                        [np.cos(x_angles), -np.sin(x_angles)],
                                        [np.sin(x_angles), np.cos(x_angles)]
                                    ])
                    # put the coordinates relative to the center of the frame
                    rotation_angles = np.array([frame_x # - cam_size[1]/2
                                                , 
                                                frame_y # - cam_size[0]/2
                                                ])
                    # Rotate
                    rotated_coors = rotation_matrix @ rotation_angles
                    # reestablish point in the corner
                    # frame_y = (row.frame_y_pt) - cam_size[1]/2
                    # frame_x = (row.frame_x_pt) - cam_size[0]/2
                    # frame_y = rotated_coors[1]  - cam_size[1]/2
                    # frame_x = rotated_coors[0] - cam_size[0]/2
                    # print("angles: ", x_angles)
                    # print("stillx, stillk_y: ", (frame_x, frame_y))
                    # print("rotated coordinates: ", rotated_coors)
                    cam_gps_lat = still_gps_lat - ((frame_y*cam_y_size)/ r_earth) * (180 / math.pi)
                    cam_gps_long = still_gps_long + ((frame_x*cam_x_size) / r_earth) * (180 / math.pi) / math.cos(still_gps_lat*math.pi/180)
                    cam_gps_lat_sum += cam_gps_lat
                    cam_gps_long_sum += cam_gps_long
                    angle_sum += x_angles
                    counter+=1
                    #break
            if cam_gps_lat == 0 or cam_gps_long == 0:
                continue

            cam_gps_lat = cam_gps_lat_sum / counter
            cam_gps_long = cam_gps_long_sum / counter
            print("cam_gps: ", (cam_gps_lat, cam_gps_long))
            drone_lat = df['latitude'].iloc[i]
            drone_lon = df['longitude'].iloc[i]

            gps_error = get_distance_metres(cam_gps_lat, cam_gps_long, drone_lat, drone_lon)
            print("GPS error: "+str(gps_error))
            gps_dist_traveled = get_distance_metres(cam_gps_lat, cam_gps_long, last_prediction_lat, last_prediction_lon)
            print(gps_dist_traveled)
            total_dist_traveled += gps_dist_traveled

            #print((still_gps_lat, still_gps_lat))
            #print((cam_gps_long, cam_gps_lat))


            print("Estimated GPS: ("+str(cam_gps_lat)+","+str(cam_gps_long)+") Actual GPS: ("+str(drone_lat)+","+str(drone_lon)+") Error: "+str(gps_error)+"m")
            total_dist_traveled += gps_dist_traveled

            last_prediction_lat = cam_gps_lat
            last_prediction_lon = cam_gps_long

            # Location that the drone thinks it is in
            # //////////////////////////////////////////////////////////
            still_copy = still_image.copy()
            # y, x = get_vector_metres(still_image_dict[1][1], still_image_dict[1][2], cam_gps_lat, cam_gps_long)
            point_y = (still_image_dict[1] - cam_gps_lat) *  math.pi / 180 * r_earth /y_size # + cam_size[1]/2
            point_x = (still_image_dict[2] - cam_gps_long) * math.pi / 180 * math.cos(still_image_dict[1]*math.pi/180) * r_earth / x_size # + cam_size[0]/2
            print("pic drawing: ", (point_x, point_y))
            # print("Size: ",  still_copy.shape)
            cv2.circle(still_copy, (int(point_x), int(point_y)) , radius=10, color=(0, 0, 0), thickness=-1)  # Red dot
            # print("hello")
            # cv2.imshow("matches", matched_img)
            # still_copy = cv2.resize(still_copy, (0,0), fx=0.5, fy=0.5) 
            cv2.imshow("point", still_copy)
            if len(still_copy.shape) == 2:  # grayscale
                still_copy = cv2.cvtColor(still_copy, cv2.COLOR_GRAY2BGR)
            vid_location.write(still_copy)
            # //////////////////////////////////////////////////////////


            # Rotation Verification
            # //////////////////////////////////////////////////////////
            # img_copy = frame.copy()
            # (h, w) = img_copy.shape[:2]
            # center = (w / 2, h / 2)
            # # Get rotation matrix for the given angle
            # M = cv2.getRotationMatrix2D(center, np.abs(angle*180/np.pi) , 1.0)
            # # Perform the rotation
            # rotated = cv2.warpAffine(img_copy, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            # rotated = cv2.resize(rotated, (0,0), fx=0.5, fy=0.5) 
            # cv2.imshow("Rotated Point", rotated)
            # //////////////////////////////////////////////////////////


            # 9. Draw the matches
            #print(good_matches[best_match_idx])
            #print(good_matches)
            matched_img = cv2.drawMatches(still_image, kp_still, frame, kp_frame, good_matches, None, flags=cv2.DrawMatchesFlags_DEFAULT)
            vid_matches.write(matched_img)
            matched_img = cv2.resize(matched_img, (0,0), fx=0.5, fy=0.5) 

            # Show the matched image
            cv2.imshow("Matches", matched_img)
            #print(cv2.getWindowImageRect("Matches"))
            #cv2.imshow("Still image key points", still_kps)
            #cv2.imshow("Frame image key points", frame_kps)


            # Press 'q' to quit the video
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    i += 1
print(total_dist_traveled)

print("______________________________________________________")
print("initial_pos: ", initial_pos)
print("final coordinates: ", (cam_gps_lat, cam_gps_long))
print("total distance: ",  total_dist_traveled)
print("displacement: ", get_distance_metres(cam_gps_lat, cam_gps_long, initial_pos[0], initial_pos[1]))
print("______________________________________________________")

# Release the video capture and close the window
vid_matches.release()
vid_location.release()
cap.release()
cv2.destroyAllWindows()
