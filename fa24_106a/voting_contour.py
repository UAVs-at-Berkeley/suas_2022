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

def truncated_adaptive_gamma(image, tau=0.3, alpha=0.2):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Histogram: P_i
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    n = gray.size
    P_i = hist / n

    # Smooth min/max (ignore zero bins)
    nonzero = P_i[P_i > 0]
    P_min = np.min(nonzero)
    P_max = np.max(nonzero)

    # Weighted PDF: P_wi
    normalized = (P_i - P_min) / (P_max - P_min)
    normalized = np.clip(normalized, 0, 1)  # ensures all values in [0, 1]
    P_wi = P_max * (normalized ** alpha)
    P_wi[P_i <= P_min] = 0  # clamp negatives

    # Cumulative weighted distribution: C_wi
    C_wi = np.cumsum(P_wi) / np.sum(P_wi)

    # Adaptive gamma for each intensity
    gamma_i = 1.0 - C_wi
    gamma = np.maximum(tau, gamma_i)

    # Build gamma LUT
    lut = np.array([255 * ((i / 255.0) ** gamma[i]) for i in range(256)]).astype(np.uint8)

    # Apply LUT
    result = cv2.LUT(gray, lut)

    return result


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
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) <= max_contour_len]

    image_copy = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_copy, filtered_contours, -1, (255, 0, 0), cv2.FILLED)
    blue = np.array([255, 0, 0])
    blue_mask = np.all(image_copy == blue, axis=2)
    image_copy[blue_mask] = [0, 0, 0]
    result = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    return result
#######****************************************************************##########

#######   Uses newer images                                             ##########

#######****************************************************************##########

still_image_dict = ('DJI_20250507160257_0026_D.png', 37.8714865, 122.3183067, 66, 117)
# still_image_dict = ('dji_screenshot.png', 37.8714926, 122.3184300, 81.3, 160.08)
# still_image_dict = ('media/pair2.png', 37.8722765, 122.3193286, 279.09, 318)
# still_image_dict = ('media/pair3.png', 37.8714926, 122.3184300, 81.3, 160.08)
still_image_dict = ('media/dji_pic.png', 37.8179660, 122.3186288, 102, 150)
still_image_dict = ('media/pair1.png', 37.872312, 122.319072, 170.3, 318)

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

# 0. parameters
# 400 is good without rectangle generation for contour-based path isolation
still_rect_spacing = 20
still_kp_ct = 400
still_blur_kernel = (21, 21)
vid_rect_spacing = 20
vid_kp_ct = 400
vid_blur_kernel = (21, 21)

still_image = cv2.imread(still_image_dict[0], cv2.IMREAD_GRAYSCALE)
print("Image shape:", still_image.shape)
still_image_color = cv2.imread(still_image_dict[0], cv2.IMREAD_COLOR)

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
max_contour_len = 4000
final_contoured = remove_smaller_contours(eroded_copy, max_contour_len)
cv2.imshow("removed contours", final_contoured)
# ===============================

# (80, 20, 80), (130, 40, 120) original lower/upper bounds
# grey_lower = (80, 20, 80)
# grey_upper = (130, 40, 120)
grey_lower = (10, 10, 100)
grey_upper = (180, 60, 255)

# for some reason grey shows up as magenta/red here... let's pull magenta roughly
hsv_filtered, _ = hsv_filter(still_image_color, grey_lower, grey_upper)
# cv2.imshow("hsv for roughly grey", hsv_filtered)

hsv_filtered_grey = cv2.cvtColor(cv2.cvtColor(hsv_filtered, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
ret, th4 = cv2.threshold(hsv_filtered_grey,50,255,cv2.THRESH_BINARY)
cv2.imshow("hsv thresholded still", th4)


still_image = final_contoured
# still_image = th4

kpts_still = orb.detect(still_image, None)
desc = cv2.xfeatures2d.BEBLID_create(0.75)
kp_still, des_still = desc.compute(still_image, kpts_still)
still_kps = cv2.drawKeypoints(still_image, kp_still, None, color=(0,255,0), flags=0)

horizontal_size = still_image.shape[:2][1]
# print(horizontal_size)
vertical_size = still_image.shape[:2][0]
# print(vertical_size)
x_size = still_image_dict[4] / horizontal_size
# print(x_size)
y_size = still_image_dict[3] / vertical_size
# print(y_size)
kpts_still = orb.detect(still_image, None)
kp_still, des_still = desc.compute(still_image, kpts_still)
still_kps = cv2.drawKeypoints(still_image, kp_still, None, color=(0,255,0), flags=0)
# cv2.imshow("stil image", still_kps)

print(cap.get(3))
print(cap.get(4))


cluster_count = 6
total_dist_traveled = 0

# vid_matches = cv2.VideoWriter('vid_matches_voting_sun.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 20, (int(horizontal_size+cap.get(3)), int(max([vertical_size, cap.get(4)]))))
# vid_location = cv2.VideoWriter('vid_location_voting_sun.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 20, (still_image.shape[1], still_image.shape[0]))

ct = 0
sample = 0
gps_error_sum = 0
i  = 1

# 5. Process each frame of the video
print("starting algorithm")
while cap.isOpened():
    ret, frame = cap.read()

    ct += 1
    #ret = cap.grab()
    if (ct >= 5):
        ct = 0
        #ret, frame = cap.read()
        if not ret:
            break
        # print("frame red")
        
        matches_xy_list = []
        matches_list = []
        angles_list = []

        cam_x = 2*(math.tan(h_fov*math.pi/2/180)*drone_alt)
        cam_diag = 2*(math.tan(d_fov*math.pi/2/180)*drone_alt)
        half_cam_diag = cam_diag/2
        cam_y = math.sqrt(4*((math.tan(79.5*math.pi/2/180))**2)*(drone_alt**2)-(cam_x**2))
        cam_x_size = cam_x / cam_size[0]
        cam_y_size = cam_y / cam_size[1]

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
        cv2.imshow("removed contours", contoured_frame)
        
        rectangle_frame = contoured_frame
        # ===============================

        # mod_rec = drawRectangles(rectangle_frame, 2, vid_rect_spacing, 150, rectangle_frame)

        # for some reason grey shows up as magenta/red here... let's pull magenta roughly
        hsv_filtered, _ = hsv_filter(frame, grey_lower, grey_upper)
        # cv2.imshow("hsv for roughly grey", hsv_filtered)

        hsv_filtered_grey = cv2.cvtColor(cv2.cvtColor(hsv_filtered, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
        ret, th4 = cv2.threshold(hsv_filtered_grey,50,255,cv2.THRESH_BINARY)


        # 6. Detect keypoints and descriptors in the frame
        # processed_frame = mod_rec
        # processed_frame = th4
        processed_frame = contoured_frame

        kpts_frame = orb.detect(processed_frame, None)
        kp_frame, des_frame = desc.compute(processed_frame, kpts_frame)
        frame_kps = cv2.drawKeypoints(processed_frame, kp_frame, None, color=(0,255,0), flags=0)

        # 7. Match descriptors using FLANN
        # print("frame red")

        if des_frame is not None:

            matches = flann.knnMatch(des_still, des_frame, k=2)
            # print("match found")

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

        # print("GOod matches: ", good_matches)
    
            
        matched_img = cv2.drawMatches(still_image, kp_still, frame_kps, kp_frame, good_matches, None, flags=cv2.DrawMatchesFlags_DEFAULT)
        # matched_img = cv2.resize(matched_img, (0,0), fx=0.5, fy=0.5) 

        # Show the matched image
        # vid_matches.write(matched_img)
        cv2.imshow("Matches", matched_img)   
        dataset = pd.DataFrame(good_matches_xy)
        
        cam_gps_long = 0
        cam_gps_lat = 0

        y = dataset
        best_match_idx = 0
        counter = 0
        cam_gps_lat_sum = []
        cam_gps_long_sum = []
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


        # median
        # print("lat, long: ", (cam_gps_lat_sum, cam_gps_long_sum))
        if len(cam_gps_lat_sum) == 0 or len(cam_gps_long_sum) == 0 or len(angle_sum) == 0:
            print("breaking, cant find the match")
            continue
        cam_gps_lat = statistics.median(cam_gps_lat_sum) 
        cam_gps_long = statistics.median(cam_gps_long_sum)
        angle = statistics.median(angle_sum)
        
        # cam_gps_lat = statistics.mean(cam_gps_lat_sum) 
        # cam_gps_long = statistics.mean(cam_gps_long_sum)

        gps_dist_traveled = get_distance_metres(cam_gps_lat, cam_gps_long, last_prediction_lat, last_prediction_lon)

        # print("difference in coordinates", difference_coor(cam_gps_lat, cam_gps_long, last_prediction_lat, last_prediction_lon))
        # print(np.array(x).dtype)  # for NumPy arrays

        # TODO: check the angle difference
        # if gps_dist_traveled > 100:
        #     print("exiting distance")
            
        #     continue


        # angle_check = angle_bound(np.abs(np.abs(angle) -  np.abs(prev_angle)))
        # if angle_check == 0:
        #     angle = 0
        # elif angle_check == 2:
        #     print("exiting angle")
        #     continue
        drone_lat = df['latitude'].iloc[i]
        drone_lon = df['longitude'].iloc[i]
        gps_error = get_distance_metres(cam_gps_lat, cam_gps_long, drone_lat, -1 * drone_lon)
        # print("GPS error: "+str(gps_error))
        gps_error_sum += gps_error
        gps_dist_traveled = get_distance_metres(cam_gps_lat, cam_gps_long, last_prediction_lat, last_prediction_lon)
        print(gps_dist_traveled)
        total_dist_traveled += gps_dist_traveled

        #print((still_gps_lat, still_gps_lat))
        #print((cam_gps_long, cam_gps_lat))

        # filegps = open("comp_gps.txt", "a")
        # if filegps != None:
        #     filegps.write("Estimated GPS: ("+str(cam_gps_lat)+","+str(cam_gps_long)+") Actual GPS: ("+str(drone_lat)+","+str(drone_lon)+") Error: "+str(gps_error)+"m")
        # filegps.close()
        # print("Estimated GPS: ("+str(cam_gps_lat)+","+str(cam_gps_long)+") Actual GPS: ("+str(drone_lat)+","+str(drone_lon)+") Error: "+str(gps_error)+"m")
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
        cv2.circle(still_copy, (int(point_x), int(point_y)) , radius=10, color=(255, 255, 255), thickness=-1)  # Red dot
        # print("hello")
        # cv2.imshow("matches", matched_img)
        # still_copy = cv2.resize(still_copy, (0,0), fx=0.5, fy=0.5) 
        cv2.imshow("point", still_copy)
        if len(still_copy.shape) == 2:  # grayscale
            still_copy = cv2.cvtColor(still_copy, cv2.COLOR_GRAY2BGR)
        # vid_location.write(still_copy)
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
        matched_img = cv2.drawMatches(still_image, kp_still, frame_kps, kp_frame, good_matches, None, flags=cv2.DrawMatchesFlags_DEFAULT)
        matched_img = cv2.resize(matched_img, (0,0), fx=0.5, fy=0.5) 

        # Show the matched image
        # vid_matches.write(matched_img)
        cv2.imshow("Matches", matched_img)
        #print(cv2.getWindowImageRect("Matches"))
        #cv2.imshow("Still image key points", still_kps)
        #cv2.imshow("Frame image key points", frame_kps)


        # Press 'q' to quit the video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        sample += 1
    i += 1
print(total_dist_traveled)

print("______________________________________________________")
print("initial_pos: ", initial_pos)
print("final coordinates: ", (cam_gps_lat, cam_gps_long))
print("total distance: ",  total_dist_traveled)
print("displacement: ", get_distance_metres(cam_gps_lat, cam_gps_long, initial_pos[0], initial_pos[1]))
# print("displacement error: ", get_distance_metres(cam_gps_lat, cam_gps_long, drone_lat, drone_lon))
print("average displacement error ", gps_error_sum/sample)
print("______________________________________________________")


# vid_matches.release()
# vid_location.release()
cap.release()
cv2.destroyAllWindows()
# cv2.waitKey(0)  # waits for a key press indefinitely
# cv2.destroyAllWindows()  # closes all OpenCV windows
