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


still_image_dict = {
    0:('37.872310N_122.322454W_231.23H_297.8W.png', 37.872310, 122.322454, 231.23, 297.8), 
    # 1:('37.872312N_122.319072W_235.56H_364.08W.png', 37.872312, 122.319072, 235.56, 364.08), 
    1:('dji_pic.png', 37.8719660, 122.3186288, 102, 150), 
    2:('37.874496H_122.322454W_242.73H_297.8W.png', 37.874496, 122.322454, 242.73, 297.8), 
    3:('37.874496N_122.319072W_242.73H_364.08W.png', 37.874496, 122.319072, 242.73, 364.08)
}


r_earth = 6378000
drone_alt = 50
drone_lat = 37.87119 #vehicle.home_location.lat
drone_lon = 122.3176 # vehicle.home_location.lon
# drone_lat = 37.8719660 #vehicle.home_location.lat
# drone_lon = 122.3186288 # vehicle.home_location.lon

last_prediction_lat = drone_lat
last_prediction_lon = drone_lon
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



def get_vector_metres(lat1, lon1, lat2, lon2):
    dlat = lat2 - lat1
    dlong = lon2 - lon1
    return dlat * 1.113195e5, dlong * 1.113195e5



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

def angle_bound(angle):
    if angle < 0.1 or angle > 6.18:
        return 0 # angle is negligible
    elif angle < 1:
        return 1 # angle is reasonable 
    else:
        return 2 #continue because the angle is an error
    

# 1. Load the still image and the video
still_image = cv2.imread(still_image_dict[1][0], cv2.IMREAD_GRAYSCALE)
video_path = 'dji_flight.MOV'
horizontal_size = still_image.shape[:2][1]
print(horizontal_size)
vertical_size = still_image.shape[:2][0]
print(vertical_size)
x_size = still_image_dict[1][4] / horizontal_size
print(x_size)
y_size = still_image_dict[1][3] / vertical_size
print(y_size)

# 2. Detect keypoints and descriptors in the still image using ORB
orb = cv2.ORB_create(nfeatures=500)
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
search_params = dict(checks=50)  # number of checks (higher is more accurate)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# 4. Open the video file
vid = "dji_fly.MOV"
cap = cv2.VideoCapture(vid)


print(cap.get(3))
print(cap.get(4))


cluster_count = 6
total_dist_traveled = 0

vid_matches = cv2.VideoWriter('vid_matches.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, (int(horizontal_size+cap.get(3)), int(max([vertical_size, cap.get(4)]))))
ct = 0
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

        cam_x = 2*(math.tan(h_fov*math.pi/2/180)*drone_alt)
        # print("camx: "+str(cam_x))
        cam_diag = 2*(math.tan(d_fov*math.pi/2/180)*drone_alt)
        half_cam_diag = cam_diag/2
        cam_y = math.sqrt(4*((math.tan(79.5*math.pi/2/180))**2)*(drone_alt**2)-(cam_x**2))
        cam_x_size = cam_x / cam_size[0]
        cam_y_size = cam_y / cam_size[1]
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 6. Detect keypoints and descriptors in the frame
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
                    if m[0].distance < 0.75 * m[1].distance:
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
           
            # print("matches: ", good_matches)

            cam_gps_long = 0
            cam_gps_lat = 0

            y = dataset
            best_match_idx = 0
            counter = 0
            cam_gps_lat_sum = []
            cam_gps_long_sum = []
            angle_sum = []



            # Compare to the frame to the still image
            # print("dataset: ", dataset)
            for row in dataset.itertuples():

                # Global Coordinates of the still frame
                x_lat = still_image_dict[1][1] - (row.still_y_pt*y_size / r_earth) * 180/math.pi
                x_long = still_image_dict[1][2] - ((row.still_x_pt*x_size / r_earth) * 180/math.pi / math.cos(still_image_dict[1][1]*math.pi/180)) 


                assert frame.shape[0:2] == (cam_size[1], cam_size[0])
                # move the origin coordinate to the top left
                # the video has a coordinate point that is based off of the left corner

                # frame_y = row.frame_y_pt - cam_size[1]/2
                # frame_x = row.frame_x_pt - cam_size[0]/2

                # local frame
                frame_y = row.frame_y_pt 
                frame_x = row.frame_x_pt

                # finding angles and rotation
                x_angles = good_angles[row.Index] # in degrees
                x_angles = x_angles * np.pi/180
                rotation_matrix = np.array([
                                    [np.cos(x_angles), -np.sin(x_angles)],
                                    [np.sin(x_angles), np.cos(x_angles)]
                                ])
                rotation_angles = np.array([frame_x #- cam_size[1]/2
                                            , 
                                            frame_y #- cam_size[0]/2
                                            ])
                rotated_coors = rotation_matrix @ rotation_angles
                frame_y = rotated_coors[1]
                frame_x = rotated_coors[0]
                # print("angles: ", x_angles)
                # print("stillx, stillk_y: ", (frame_x, frame_y))
                # print("rotated coordinates: ", rotated_coors)

                # Detect the difference between the 2 frames
                cam_gps_lat = x_lat + ((((frame_y))*cam_y_size)/ r_earth) * (180 / math.pi)
                cam_gps_long = x_long + (((frame_x)*cam_x_size) / r_earth) * (180 / math.pi) / math.cos(x_lat*math.pi/180)
                
                # cam_gps_lat = still_image_dict[1][1] - ((row.still_y_pt*y_size + ((row.frame_y_pt) - cam_size[1]/2))*cam_y_size / r_earth) * 180/math.pi
                # cam_gps_long = still_image_dict[1][2] - (((row.still_x_pt*x_size + ((row.frame_x_pt) - cam_size[0]/2)*cam_x_size)/ r_earth) * 180/math.pi / math.cos(cam_gps_lat*math.pi/180)) 


                # cam_gps_lat = still_image_dict[1][1] - ((row.still_y_pt*y_size + ((row.frame_y_pt) - cam_size[1]/2))*cam_y_size / r_earth) * 180/math.pi
                # cam_gps_long = still_image_dict[1][2] - (((row.still_x_pt*x_size + ((row.frame_x_pt) - cam_size[0]/2)*cam_x_size)/ r_earth) * 180/math.pi / math.cos(cam_gps_lat*math.pi/180)) 


                # print("x_coor change: ", ((row.still_y_pt*y_size + ((row.frame_y_pt) - cam_size[1]/2))*cam_y_size / r_earth) * 180/math.pi)

                cam_gps_lat_sum.append(cam_gps_lat)
                cam_gps_long_sum.append(cam_gps_long) 
                angle_sum.append(x_angles)
                counter+=1
                    #break
            # TODO: The error is showing at the end


            # median
            # print("lat, long: ", (cam_gps_lat_sum, cam_gps_long_sum))

            cam_gps_lat = statistics.median(cam_gps_lat_sum) 
            cam_gps_long = statistics.median(cam_gps_long_sum)
            angle = statistics.median(angle_sum)
            
            # cam_gps_lat = statistics.mean(cam_gps_lat_sum) 
            # cam_gps_long = statistics.mean(cam_gps_long_sum)

            gps_dist_traveled = get_distance_metres(cam_gps_lat, cam_gps_long, last_prediction_lat, last_prediction_lon)

            # print("difference in coordinates", difference_coor(cam_gps_lat, cam_gps_long, last_prediction_lat, last_prediction_lon))
            # print(np.array(x).dtype)  # for NumPy arrays

            # TODO: check the angle difference
            if gps_dist_traveled > 100:
                print("exiting distance")
                
                continue

            angle_check = angle_bound(np.abs(np.abs(angle) -  np.abs(prev_angle)))
            if angle_check == 0:
                angle = 0
            elif angle_check == 2:
                print("exiting angle")
                continue

            # print("last pos: ", (last_prediction_lat, last_prediction_lon))
            # print("cam gps: ", (cam_gps_lat, cam_gps_long))
            # print("distance traveled: ", gps_dist_traveled)
            total_dist_traveled += gps_dist_traveled

            last_prediction_lat = cam_gps_lat
            last_prediction_lon = cam_gps_long

            # Location that the drone thinks it is in
            # //////////////////////////////////////////////////////////
            still_copy = still_image.copy()
            # y, x = get_vector_metres(still_image_dict[1][1], still_image_dict[1][2], cam_gps_lat, cam_gps_long)
            point_y = (still_image_dict[1][1] - cam_gps_lat) *  math.pi / 180 * r_earth /y_size # + cam_size[1]/2
            point_x = (still_image_dict[1][2] - cam_gps_long) * math.pi / 180 * math.cos(still_image_dict[1][1]*math.pi/180) * r_earth / x_size # + cam_size[0]/2
            print("pic drawing: ", (point_x, point_y))
            # print("Size: ",  still_copy.shape)
            cv2.circle(still_copy, (int(point_x), int(point_y)) , radius=5, color=(255, 255, 255), thickness=-1)  # Red dot
            # print("hello")
            # cv2.imshow("matches", matched_img)
            still_copy = cv2.resize(still_copy, (0,0), fx=0.5, fy=0.5) 
            cv2.imshow("point", still_copy)
            # //////////////////////////////////////////////////////////


            # Rotation Verification
            # //////////////////////////////////////////////////////////
            img_copy = frame.copy()
            (h, w) = img_copy.shape[:2]
            center = (w / 2, h / 2)
            # Get rotation matrix for the given angle
            M = cv2.getRotationMatrix2D(center, np.abs(angle*180/np.pi) , 1.0)
            # Perform the rotation
            rotated = cv2.warpAffine(img_copy, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            rotated = cv2.resize(rotated, (0,0), fx=0.5, fy=0.5) 
            cv2.imshow("Rotated Point", rotated)
            # //////////////////////////////////////////////////////////


            # 9. Draw the matches
            #print(good_matches[best_match_idx])
            #print(good_matches)
            matched_img = cv2.drawMatches(still_image, kp_still, frame, kp_frame, good_matches, None, flags=cv2.DrawMatchesFlags_DEFAULT)
            matched_img = cv2.resize(matched_img, (0,0), fx=0.5, fy=0.5) 

            # Show the matched image
            vid_matches.write(matched_img)
            # cv2.imshow("Matches", matched_img)
            #print(cv2.getWindowImageRect("Matches"))
            #cv2.imshow("Still image key points", still_kps)
            #cv2.imshow("Frame image key points", frame_kps)


            # Press 'q' to quit the video
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
print(total_dist_traveled)
# Release the video capture and close the window
vid_matches.release()
cap.release()
cv2.destroyAllWindows()
