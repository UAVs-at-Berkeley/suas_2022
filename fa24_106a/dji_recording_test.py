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
    # 1:('37.872312N_122.319072W_235.56H_364.08W.png', 37.872312, 122.319072, 235.56, 364.08), 
    1:('dji_pic.png', 37.872312, 122.319072, 235.56, 364.08), 
    2:('37.874496H_122.322454W_242.73H_297.8W.png', 37.874496, 122.322454, 242.73, 297.8), 
    3:('37.874496N_122.319072W_242.73H_364.08W.png', 37.874496, 122.319072, 242.73, 364.08)
}


r_earth = 6378000
drone_alt = 30
drone_lat = 37.87119 #vehicle.home_location.lat
drone_lon = 122.3176 # vehicle.home_location.lon
last_prediction_lat = drone_lat
last_prediction_lon = drone_lon
h_fov = 71.5
d_fov = 79.5
cam_size = (1920, 1080)
droner_lat = 0
droner_lon = 0
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





def get_location_metres(original_location, dNorth, dEast):
    """
    Returns a LocationGlobal object containing the latitude/longitude `dNorth` and `dEast` metres from the
    specified `original_location`. The returned LocationGlobal has the same `alt` value
    as `original_location`.

    The function is useful when you want to move the vehicle around specifying locations relative to
    the current vehicle position.

    The algorithm is relatively accurate over small distances (10m within 1km) except close to the poles.
    """
    earth_radius=6378137.0 #Radius of "spherical" earth
    #Coordinate offsets in radians
    dLat = dNorth/earth_radius
    dLon = dEast/(earth_radius*math.cos(math.pi*original_location.lat/180))

    #New position in decimal degrees
    newlat = original_location.lat + (dLat * 180/math.pi)
    newlon = original_location.lon + (dLon * 180/math.pi)
    if type(original_location) is LocationGlobal:
        targetlocation=LocationGlobal(newlat, newlon,original_location.alt)
    elif type(original_location) is LocationGlobalRelative:
        targetlocation=LocationGlobalRelative(newlat, newlon,original_location.alt)
    else:
        raise Exception("Invalid Location object passed")

    return targetlocation

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
        print("camx: "+str(cam_x))
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
                        #print(m[0].queryIdx)
                        #print(m[1].queryIdx)
                        #print(m[0].trainIdx)
                        #print(m[1].trainIdx)
                        #query_idx = 0#m.queryIdx
                        #train_idx = 0#m.trainIdx 537,935
                        still_pt = kp_still[m[0].queryIdx].pt
                        frame_pt = kp_frame[m[0].trainIdx].pt
                        #print(still_pt)
                        #print(frame_pt)
                        
                        good_matches_xy['frame_x_pt'].append(frame_pt[0])
                        good_matches_xy['frame_y_pt'].append(frame_pt[1])
                        good_matches_xy['still_x_pt'].append(still_pt[0])
                        good_matches_xy['still_y_pt'].append(still_pt[1])
                        
                        cv2.putText(still_kps, text=str(str((int(still_pt[0]), int(still_pt[1])))), org=(int(still_pt[0]), int(still_pt[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2, lineType=cv2.LINE_AA)
                        cv2.putText(frame_kps, text=str((int(frame_pt[0]), int(frame_pt[1]))), org=(int(frame_pt[0]), int(frame_pt[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2, lineType=cv2.LINE_AA)
                        good_matches.append(m[0])
                        
            dataset = pd.DataFrame(good_matches_xy)
           








            # 9. Draw the matches
            #print(good_matches[best_match_idx])
            #print(good_matches)
            matched_img = cv2.drawMatches(still_image, kp_still, frame, kp_frame, good_matches, None, flags=cv2.DrawMatchesFlags_DEFAULT)

            # Show the matched image
            vid_matches.write(matched_img)
            cv2.imshow("Matches", matched_img)
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
