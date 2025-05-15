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
    0:('media/37.872310N_122.322454W_231.23H_297.8W.png', 37.872310, 122.322454, 231.23, 297.8), 
    # 1:('media/pair1.png', 37.872312, 122.319072, 170.3, 318), 
    # 1:('media/pair2.png', 37.8722765, 122.3193286, 279.09, 318), 
    1:('DJI_20250507160257_0026_D.png', 37.8714865, 122.3183067, 66, 117), 
    # 1:('pair3.png', 37.8714926, 122.3184300, 81.3, 160.08), 

    2:('media/37.874496H_122.322454W_242.73H_297.8W.png', 37.874496, 122.322454, 242.73, 297.8), 
    3:('media/37.874496N_122.319072W_242.73H_364.08W.png', 37.874496, 122.319072, 242.73, 364.08)
}

# vid = ('media/still_vid.mp4', 37.8722765, 122.3193286, 223.73, 300)






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


def get_vector_metres(lat1, lon1, lat2, lon2):
    dlat = lat2 - lat1
    dlong = lon2 - lon1
    return dlat * 1.113195e5, dlong * 1.113195e5


def difference_coor(lat1, lon1, lat2, lon2):
    dlat = lat2 - lat1
    dlong = lon2 - lon1
    return dlat, dlong

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


def isolateCurve(grey_image):
    # Edge detection
    blurred = cv2.GaussianBlur(grey_image, (11, 11), 0)
    edges = cv2.Canny(blurred, 100, 200)
    
    # Contour finding
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize variables to store the best contour and its area
    len_threshold = 2
    best_contour = None
    best_contours = []
    max_area = -1

    # Curved line isolation
    for contour in contours:
        # ===== isolate single best contours =====
        # # Calculate the area of the contour
        # area = cv2.contourArea(contour)
        
        # # Approximate the contour to simplify its shape
        # perimeter = cv2.arcLength(contour, True)
        # approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        
        # # Check if the contour is curved based on the number of vertices
        # if len(approx) > 5 and area > max_area:
        #    best_contour = contour
        #    max_area = area
        # ========================================
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        
        # Approximate the contour to simplify its shape
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) > len_threshold:
            best_contours.append(contour)


    # Masking and extraction
    mask = np.zeros_like(blurred)
    # cv2.drawContours(mask, [best_contour], -1, 255, cv2.FILLED)
    cv2.drawContours(mask, best_contours, -1, 255, cv2.FILLED)
    result = cv2.bitwise_and(grey_image, grey_image, mask=mask)    
    return result, mask

def hsv_filter(image, lower_hsv, upper_hsv):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result, mask

def remove_inner_contours(binary_image):
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    image_copy = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            # Check if the contour has a parent
            if hierarchy[0][i][3] != -1:
                # Draw the inner contour in green
                cv2.drawContours(image_copy, [contour], -1, (0, 255, 0), cv2.FILLED)
    green = np.array([0, 255, 0])
    green_mask = np.all(image_copy == green, axis=2)
    image_copy[green_mask] = [0, 0, 0]
    result = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    return result

def remove_smaller_contours(binary_image, max_contour_len):
    contours, _ = cv2.findContours(eroded_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) <= max_contour_len]

    image_copy = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_copy, filtered_contours, -1, (255, 0, 0), cv2.FILLED)
    blue = np.array([255, 0, 0])
    blue_mask = np.all(image_copy == blue, axis=2)
    image_copy[blue_mask] = [0, 0, 0]
    result = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    return result

vid = ('media/pair2.mp4', 37.8722765, 122.3193286, 279.09, 318)

# 4. Open the video file
video_path = 'DJI_20250507160257_0026_D.MP4'
cap = cv2.VideoCapture(video_path)

parsed = parse_metadata("DJI_20250507160257_0026_D.SRT")
df = pd.DataFrame(parsed)

r_earth = 6378000
# cap = cv2.VideoCapture(vid[0])
ret, frame = cap.read()
# # print("frame", frame.shape[1])


# TODO: Replace this part with google map video, fill out starting 
# starting:  
# ending: 
drone_alt = df['rel_alt'].iloc[0]
drone_lat = df['latitude'].iloc[0]
drone_lon = df['longitude'].iloc[0]
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
initial_pos = (37.871614, 122.317525)
last_prediction_lat = initial_pos[0] #vehicle.home_location.lat
last_prediction_lon = initial_pos[1] # vehicle.home_location.lon
last_prediction_lat_sum = [] 
last_prediction_lon_sum = [] 
last_x_lat_sum = [] 
last_x_long_sum = []
last_still_x = []
last_still_y = []

# h_fov = 71.5
# d_fov = 79.5
cam_size = (frame.shape[1], frame.shape[0])
droner_lat = 0
droner_lon = 0



# 0. parameters
# 400 is good without rectangle generation for contour-based path isolation
still_rect_spacing = 20
still_kp_ct = 400
still_blur_kernel = (21, 21)
vid_rect_spacing = 20
vid_kp_ct = 400
vid_blur_kernel = (21, 21)


# 1. Load the still image and the video
still_image = cv2.imread(still_image_dict[1][0], cv2.IMREAD_GRAYSCALE)
still_image_color = cv2.imread(still_image_dict[1][0], cv2.IMREAD_COLOR)
# video_path = 'google_movie.mp4'
horizontal_size = still_image.shape[:2][1]
# # print(horizontal_size)
vertical_size = still_image.shape[:2][0]
# # print(vertical_size)
x_size = still_image_dict[1][4] / horizontal_size
# # print(x_size)
y_size = still_image_dict[1][3] / vertical_size
# # print(y_size)

frame_x_size = vid[4] / cam_size[0]
frame_y_size = vid[3] / cam_size[1]

# 2. Detect keypoints and descriptors in the still image using ORB
orb = cv2.ORB_create(nfeatures=still_kp_ct)
orb2 = cv2.ORB_create(nfeatures=vid_kp_ct)
#kp_still, des_still = orb.detectAndCompute(still_image, None)


# alpha is 1.0 - 3.0 inclusive (gain/contrast)
alpha = 1.5
# beta is 0 - 100 inclusive (bias/brightness)
beta = 0
contrasty = cv2.convertScaleAbs(still_image, alpha=alpha, beta=beta)

# Image preprocess
bin_threshold = 110
blurred = cv2.GaussianBlur(contrasty, still_blur_kernel, 0)
ret, th1 = cv2.threshold(blurred,bin_threshold,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,41,2)
# interesting comment: using unblurred image generates a lot of non-connected contours

# ===== contouring strategy =====
image_copy = remove_inner_contours(th2)
erode_kernel = np.ones((5, 5), np.uint8)
eroded = cv2.erode(image_copy, erode_kernel)
eroded_copy = remove_inner_contours(eroded)
max_contour_len = 7000
final_contoured = remove_smaller_contours(eroded_copy, max_contour_len)
cv2.imshow("removed contours", final_contoured)
# ===============================

# (80, 20, 80), (130, 40, 120) original lower/upper bounds
grey_lower = (80, 20, 80)
grey_upper = (130, 40, 120)
# for some reason grey shows up as magenta/red here... let's pull magenta roughly
hsv_filtered, _ = hsv_filter(still_image_color, grey_lower, grey_upper)
# cv2.imshow("hsv for roughly grey", hsv_filtered)

hsv_filtered_grey = cv2.cvtColor(cv2.cvtColor(hsv_filtered, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
ret, th4 = cv2.threshold(hsv_filtered_grey,50,255,cv2.THRESH_BINARY)
cv2.imshow("hsv thresholded still", th4)


still_image = final_contoured

kpts_still = orb.detect(still_image, None)
desc = cv2.xfeatures2d.BEBLID_create(0.75)
kp_still, des_still = desc.compute(still_image, kpts_still)
still_kps = cv2.drawKeypoints(still_image, kp_still, None, color=(0,255,0), flags=0)

# 3. Initialize the FLANN-based matcher
index_params = dict(algorithm=6,  # FLANN_INDEX_LSH for ORB
                    table_number=6,  # number of hash tables
                    key_size=12,     # size of the hashed key
                    multi_probe_level=1)  # multi-probe level
search_params = dict(checks=1000)  # number of checks (higher is more accurate)

flann = cv2.FlannBasedMatcher(index_params, search_params)


##### CONSTANTS ##########
deg_to_rad = 180/math.pi
cv2.setNumThreads(1)
cluster_count = 6
total_dist_traveled = 0

# TODO: Everytime the image changes, look at the output frame
ct = 0
initial = 0

vid_matches = cv2.VideoWriter('vid_matches_voting_sun.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 20, (int(horizontal_size+cap.get(3)), int(max([vertical_size, cap.get(4)]))))
vid_location = cv2.VideoWriter('vid_location_voting_sun.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 20, (still_image.shape[1], still_image.shape[0]))

# 5. Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()

    ct += 1
    # starts every 20 frames
    if (ct >= 1):
        ct = 0
        if not ret:
            break

        # ===== edge detector pipeline =====
        # # Convert the frame to grayscale
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # edges, _ = isolateCurve(gray_frame)

        # # Image preprocess
        # blurred = cv2.GaussianBlur(gray_frame, vid_blur_kernel, 0)
        # edges = cv2.Canny(blurred, 100, 200)
        # rectangle_frame = edges
        # ==================================
        
        # ===== hsv grey pipeline =====
        # hsv_filtered, _ = hsv_filter(frame, grey_lower, grey_upper)
        # hsv_filtered_grey = cv2.cvtColor(cv2.cvtColor(hsv_filtered, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
        # ret, th4 = cv2.threshold(hsv_filtered_grey,50,255,cv2.THRESH_BINARY)
        # cv2.imshow("hsv thresholded frames", th4)
        # rectangle_frame = th4
        # =============================

        # ===== contouring pipeline =====
        contrasty = cv2.convertScaleAbs(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), alpha=alpha, beta=beta)

        # Image preprocess
        blurred = cv2.GaussianBlur(contrasty, still_blur_kernel, 0)
        ret, th1 = cv2.threshold(blurred,bin_threshold,255,cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,41,2)
        # interesting comment: using unblurred image generates a lot of non-connected contours

        # ===== contouring strategy =====
        image_copy = remove_inner_contours(th2)
        erode_kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(image_copy, erode_kernel)
        eroded_copy = remove_inner_contours(eroded)
        max_contour_len = 5000
        contoured_frame = remove_smaller_contours(eroded_copy, max_contour_len)
        # cv2.imshow("removed contours", contoured_frame)
        
        rectangle_frame = contoured_frame
        # ===============================

        # mod_rec = drawRectangles(rectangle_frame, 2, vid_rect_spacing, 150, rectangle_frame)

        # 6. Detect keypoints and descriptors in the frame
        # processed_frame = mod_rec
        # processed_frame = th4
        processed_frame = contoured_frame

        kpts_frame = orb2.detect(processed_frame, None)
        kp_frame, des_frame = desc.compute(processed_frame, kpts_frame)
        frame_kps = cv2.drawKeypoints(processed_frame, kp_frame, None, color=(0,255,0), flags=0)

        # 7. Match descriptors using FLANN
        if des_frame is not None:
            matches = flann.knnMatch(des_still, des_frame, k=2)
            # # print("matches: ", matches)

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
                        # Matched keypoints
                        still_pt = kp_still[m[0].queryIdx].pt # point on the image
                        frame_pt = kp_frame[m[0].trainIdx].pt # point on the video
                        
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
            
            # # print("good_matches: ", good_matches)
            dataset = pd.DataFrame(good_matches_xy)
            # This finds the centroid of the image position and rotation
            # # print("medians: ", medians)


            cam_gps_long = 0
            cam_gps_lat = 0

            y = dataset
            best_match_idx = 0
            counter = 0
            cam_gps_lat_sum = []
            cam_gps_long_sum = []
            cluster_matches = []
            angle_sum = []


            # Compare to the frame to the still image
            for row in dataset.itertuples():

                # Global Coordinates of the still frame
                x_lat = still_image_dict[1] - (row.still_y_pt*y_size / r_earth) * 180/math.pi
                x_long = still_image_dict[2] - ((row.still_x_pt*x_size / r_earth) * 180/math.pi / math.cos(still_image_dict[1]*math.pi/180)) 


                # assert frame.shape[0:2] == (cam_size[1], cam_size[0])
                frame_y = row.frame_y_pt - cam_size[1]/2
                frame_x = row.frame_x_pt - cam_size[0]/2

                # local frame
                # frame_y = row.frame_y_pt 
                # frame_x = row.frame_x_pt

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
                # frame_y = rotated_coors[1]  # + cam_size[1]/2
                # frame_x = rotated_coors[0]  # + cam_size[0]/2
                # print("angles: ", x_angles)
                # print("stillx, stillk_y: ", (frame_x, frame_y))
                # print("rotated coordinates: ", rotated_coors)

                # Detect the difference between the 2 frames
                cam_gps_lat = x_lat + ((((frame_y))*cam_y_size)/ r_earth) * (180 / math.pi)
                cam_gps_long = x_long + (((frame_x)*cam_x_size) / r_earth) * (180 / math.pi) / math.cos(cam_gps_lat*math.pi/180)
                
                # cam_gps_lat = still_image_dict[1][1] - ((row.still_y_pt*y_size + ((row.frame_y_pt) - cam_size[1]/2))*cam_y_size / r_earth) * 180/math.pi
                # cam_gps_long = still_image_dict[1][2] - (((row.still_x_pt*x_size + ((row.frame_x_pt) - cam_size[0]/2)*cam_x_size)/ r_earth) * 180/math.pi / math.cos(cam_gps_lat*math.pi/180)) 
                # print("x_coor change: ", ((row.still_y_pt*y_size + ((row.frame_y_pt) - cam_size[1]/2))*cam_y_size / r_earth) * 180/math.pi)

                cam_gps_lat_sum.append(cam_gps_lat)
                cam_gps_long_sum.append(cam_gps_long) 
                angle_sum.append(x_angles)
                counter+=1
                    #break
            # TODO: The error is showing at the end

            if len(cam_gps_lat_sum) == 0 or len(cam_gps_long_sum) == 0:
                # print("exiting")
                continue

            cam_gps_lat_sum = [round(val, 7) for val in cam_gps_lat_sum]
            cam_gps_long_sum = [round(val, 7) for val in cam_gps_long_sum]

            cam_gps_lat = statistics.median(cam_gps_lat_sum) 
            cam_gps_long = statistics.median(cam_gps_long_sum)
            
            if initial == 0:
                initial_pos = (last_prediction_lat, last_prediction_lon)
                initial += 1

            gps_dist_traveled = get_distance_metres(cam_gps_lat, cam_gps_long, last_prediction_lat, last_prediction_lon)

            # # print("difference in coordinates", difference_coor(cam_gps_lat, cam_gps_long, last_prediction_lat, last_prediction_lon))
            # # print(np.array(x).dtype)  # for NumPy arrays

            # if gps_dist_traveled > 100:
            #     continue

            # print("last pos: ", (last_prediction_lat, last_prediction_lon))
            # print("cam gps: ", (cam_gps_lat, cam_gps_long))
            # print("distance traveled: ", gps_dist_traveled)
            total_dist_traveled += gps_dist_traveled



            # 9. Draw the matches
            matched_img = cv2.drawMatches(still_image, kp_still, frame_kps, kp_frame, good_matches, None, flags=cv2.DrawMatchesFlags_DEFAULT)

            last_prediction_lat = cam_gps_lat
            last_prediction_lon = cam_gps_long

            still_copy = cv2.cvtColor(still_image, cv2.COLOR_GRAY2BGR)

            y, x = get_vector_metres(still_image_dict[1][1], still_image_dict[1][2], cam_gps_lat, cam_gps_long)
            point_y = (still_image_dict[1][1] - cam_gps_lat) *  math.pi / 180 * r_earth/y_size + cam_size[1]/2
            point_x = (still_image_dict[1][2] -cam_gps_long ) * math.pi / 180 * math.cos(still_image_dict[1][1]*math.pi/180) * r_earth / x_size + cam_size[0]/2

            cv2.circle(still_copy, (int(point_x), int(point_y)) , radius=4, color=(0, 255, 0), thickness=-1)  # Red dot
            
            cv2.imshow("point", still_copy)
            cv2.imshow("matches", matched_img)

            # time.sleep(0.2)

            # Press 'q' to quit the video
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


print("______________________________________________________")
print("initial_pos: ", initial_pos)
print("final coordinates: ", (cam_gps_lat, cam_gps_long))
print("total distance: ",  total_dist_traveled)
print("displacement: ", get_distance_metres(cam_gps_lat, cam_gps_long, initial_pos[0], initial_pos[1]))
print("average displacement error ", gps_error_sum/sample)
print("______________________________________________________")

# Release the video capture and close the window
# vid_matches.release()
cap.release()
cv2.destroyAllWindows()
