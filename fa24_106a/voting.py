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
    # 1:('pair1.png', 37.872312, 122.319072, 170.3, 318), 
    1:('pair2.png', 37.8722765, 122.3193286, 279.09, 318), 
    # 1:('pair3.png', 37.8714926, 122.3184300, 81.3, 160.08), 

    2:('37.874496H_122.322454W_242.73H_297.8W.png', 37.874496, 122.322454, 242.73, 297.8), 
    3:('37.874496N_122.319072W_242.73H_364.08W.png', 37.874496, 122.319072, 242.73, 364.08)
}

vid = ('still_vid.mp4', 37.8722765, 122.3193286, 223.73, 300)
vid = ('pair2.mp4', 37.8722765, 122.3193286, 279.09, 318)
# vid = ('pair3.mp4', 37.8714926, 122.3184300, 60, 117)

r_earth = 6378000

# Video File
# video_path = "still_vid.mp4"

cap = cv2.VideoCapture(vid[0])
ret, frame = cap.read()
# print("frame", frame.shape[1])


# TODO: Replace this part with google map video, fill out starting 
# starting:  
# ending: 
drone_alt = 30
# last_prediction_lat = 37.872312
# last_prediction_lon = 122.319072
# initial_pos = (37.8714191, 122.3171289)
# initial_pos = (37.87147004386031, 122.31739616155451)
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

frame_x_size = vid[4] / cam_size[0]
frame_y_size = vid[3] / cam_size[1]

# 2. Detect keypoints and descriptors in the still image using ORB
orb = cv2.ORB_create(nfeatures=600)
#kp_still, des_still = orb.detectAndCompute(still_image, None)

# Image preprocess
blurred = cv2.GaussianBlur(still_image, (5,5), 0)
edges = cv2.Canny(blurred, 50, 200)
mod_rec = drawRectangles(edges, 2, 20, 150, still_image)

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

# 4. Open the video file
# rtsp_url = "rtsp://192.168.144.25:8554/main.264"



##### CONSTANTS ##########
deg_to_rad = 180/math.pi
cv2.setNumThreads(1)

print(cap.get(3))
print(cap.get(4))


cluster_count = 6
total_dist_traveled = 0

# TODO: Everytime the image changes, look at the output frame
vid_matches = cv2.VideoWriter('vid_matches.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (2084 , 975))
ct = 0
initial = 0
# 5. Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()

    ct += 1
    # starts every 20 frames
    if (ct >= 5):
        ct = 0
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Image preprocess
        blurred = cv2.GaussianBlur(gray_frame, (5,5), 0)
        edges = cv2.Canny(blurred, 50, 200)
        mod_rec = drawRectangles(edges, 2, 20, 150, gray_frame)

        # 6. Detect keypoints and descriptors in the frame
        #kp_frame, des_frame = orb2.detectAndCompute(gray_frame, None)
        kpts_frame = orb.detect(gray_frame, None)
        kp_frame, des_frame = desc.compute(gray_frame, kpts_frame)

        frame_kps = cv2.drawKeypoints(gray_frame, kp_frame, None, color=(0,255,0), flags=0)

        # 7. Match descriptors using FLANN
        if des_frame is not None:
            matches = flann.knnMatch(des_still, des_frame, k=2)
            # print("matches: ", matches)

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
            
            # print("good_matches: ", good_matches)
            dataset = pd.DataFrame(good_matches_xy)
            # This finds the centroid of the image position and rotation
            # print("medians: ", medians)


            cam_gps_long = 0
            cam_gps_lat = 0

            y = dataset
            best_match_idx = 0
            counter = 0
            cam_gps_lat_sum = []
            cam_gps_long_sum = []
            cluster_matches = []
            x_lat_sum = []
            x_long_sum = []
            x_still = []
            y_still = []

            # Compare to the frame to the still image
            # print("dataset")
            for row in dataset.itertuples():
                #print(row)
                # Moves the point down to where the match is to find the global coordinate of othe point
                # print("original image print: ", (row.still_x_pt, row.still_y_pt))
                # print("still_y coordinate: ", ((row.still_y_pt) *y_size / r_earth) * 180/math.pi)
                # print("still_x coordinate: ", (((row.still_x_pt)*x_size / r_earth) * 180/math.pi / math.cos(still_image_dict[1][1]*math.pi/180)) )
                x_lat = still_image_dict[1][1] - ((row.still_y_pt)*y_size / r_earth) * 180/math.pi
                x_long = still_image_dict[1][2] - (((row.still_x_pt)*x_size / r_earth) * 180/math.pi / math.cos(x_lat*math.pi/180)) 
                x_lat_sum.append(x_lat)
                x_long_sum.append(x_long)
                x_still.append(row.still_x_pt)
                y_still.append(row.still_y_pt)
    

                assert frame.shape[0:2] == (cam_size[1], cam_size[0])
                # move the origin coordinate to the top left
                # frame_y = (((row.frame_y_pt) - cam_size[1]/2))
                # frame_x = (((row.frame_x_pt) - cam_size[0]/2))

                # the video has a coordinate point that is based off of the left corner
                frame_y = row.frame_y_pt
                frame_x = row.frame_x_pt
                
                # Detect the difference between the 2 frames
                cam_gps_lat = x_lat - (frame_y * frame_y_size/ r_earth) * 180/math.pi
                cam_gps_long = x_long - (frame_x * frame_x_size / r_earth) * 180/math.pi / math.cos(cam_gps_lat*math.pi/180)
                
                cam_gps_lat = still_image_dict[1][1] - ((row.still_y_pt*y_size + frame_y * frame_y_size) / r_earth) * 180/math.pi
                cam_gps_long = still_image_dict[1][2] - (((row.still_x_pt*x_size +  frame_x * frame_x_size )/ r_earth) * 180/math.pi / math.cos(cam_gps_lat*math.pi/180)) 

                
                cam_gps_lat_sum.append(cam_gps_lat)
                cam_gps_long_sum.append(cam_gps_long) 
                counter+=1
                    #break
            # TODO: The error is showing at the end

            if len(cam_gps_lat_sum) == 0 or len(cam_gps_long_sum) == 0:
                print("exiting")
                continue


            # print("still image diff check ")
            # if len(last_x_lat_sum) > 0:
            #     for i in range(min(len(last_x_lat_sum), len(x_lat_sum))):
            #         print("still: ", x_still[i] - last_still_x[i])
            #         print("x_lat: ", last_x_lat_sum[i] - x_lat_sum[i])
            #         # assert last_x_lat_sum[i] == x_lat_sum[i]
            # last_x_lat_sum = x_lat_sum[:]
            # last_x_long_sum = x_long_sum[:]
            # last_still_x = x_still[:]
            # last_still_y = y_still[:]




            # median
            # fig = plt.figure()
            # ax1 = fig.add_subplot(211)
            # ax2 = fig.add_subplot(212)
            
            # ax1.scatter(cam_gps_lat_sum, np.zeros(len(cam_gps_lat_sum)), marker='x')
            # ax2.scatter(cam_gps_long_sum, np.zeros(len(cam_gps_long_sum)), marker='o')
            # plt.show()
            # print((cam_gps_long_sum, cam_gps_lat_sum))
            # print((last_prediction_lon_sum, last_prediction_lat_sum))
            cam_gps_lat_sum = [round(val, 7) for val in cam_gps_lat_sum]
            cam_gps_long_sum = [round(val, 7) for val in cam_gps_long_sum]

            cam_gps_lat = statistics.median(cam_gps_lat_sum) 
            cam_gps_long = statistics.median(cam_gps_long_sum)
            
            # cam_gps_lat = statistics.mean(cam_gps_lat_sum) 
            # cam_gps_long = statistics.mean(cam_gps_long_sum)


            # scatter1 = plt.scatter(
            #     last_prediction_lat_sum, last_prediction_lon_sum, cmap='tab10', label='Still Image Matches', marker='o'
            # )

            # # Plot the frame keypoints with the same cluster-based colors but a different marker
            # scatter2 = plt.scatter(
            #     cam_gps_lat_sum, cam_gps_long_sum, cmap='Set3', label='Frame Matches', marker='x'
            # )

            # last_prediction_lat_sum = cam_gps_lat_sum[:]
            # last_prediction_lon_sum = cam_gps_long_sum[:]
            # Create a legend
            # plt.legend(loc='upper right')

            # # Optional: Add colorbar to reflect cluster info
            # plt.colorbar(scatter1, label='Cluster ID (Still)')

            # # Add axis labels and show
            # plt.xlabel('X Coordinate')
            # plt.ylabel('Y Coordinate')
            # plt.title('Clustered Keypoint Matches: Still vs. Frame')
            # # plt.grid(True)
            # plt.tight_layout()
            # plt.show()

            if initial == 0:
                initial_pos = (last_prediction_lat, last_prediction_lon)
                initial += 1

            gps_dist_traveled = get_distance_metres(cam_gps_lat, cam_gps_long, last_prediction_lat, last_prediction_lon)

            # print("difference in coordinates", difference_coor(cam_gps_lat, cam_gps_long, last_prediction_lat, last_prediction_lon))
            # print(np.array(x).dtype)  # for NumPy arrays

            # if gps_dist_traveled > 100:
            #     continue

            print("last pos: ", (last_prediction_lat, last_prediction_lon))
            print("cam gps: ", (cam_gps_lat, cam_gps_long))
            print("distance traveled: ", gps_dist_traveled)
            total_dist_traveled += gps_dist_traveled



            # 9. Draw the matches
            matched_img = cv2.drawMatches(still_image, kp_still, gray_frame, kp_frame, good_matches, None, flags=cv2.DrawMatchesFlags_DEFAULT)



            # Show the matched image
            # print("output frame shape ", matched_img.shape[1], matched_img.shape[0]  )
            # matched_img = cv2.cvtColor(matched_img, cv2.COLOR_GRAY2BGR)

            vid_matches.write(matched_img)
            last_prediction_lat = cam_gps_lat
            last_prediction_lon = cam_gps_long

            still_copy = still_image.copy()

            # y, x = get_vector_metres(still_image_dict[1][3], still_image_dict[1][4], cam_gps_lat, cam_gps_long)
            # point_y = (still_image_dict[1][1] - cam_gps_lat) *  math.pi / 180 * r_earth / y_size 
            # point_x = (still_image_dict[1][2] - cam_gps_long) * math.pi / 180 * math.cos(still_image_dict[1][1]*math.pi/180) * r_earth / x_size
            # print("pic drawing: ", (point_x, point_y))
            # print("Size: ",  still_copy.shape)

            # cv2.circle(still_copy, (int(point_y), int(point_x)) , radius=4, color=(0, 0, 255), thickness=-1)  # Red dot
            

            # print("hello")
            # cv2.imshow("matches", matched_img)
            # cv2.imshow("point", still_copy)
            # Press 'q' to quit the video
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


print("______________________________________________________")
print("initial_pos: ", initial_pos)
print("final coordinates: ", (cam_gps_lat, cam_gps_long))
print("total distance: ",  total_dist_traveled)
print("displacement: ", get_distance_metres(cam_gps_lat, cam_gps_long, initial_pos[0], initial_pos[1]))
print("______________________________________________________")

# Release the video capture and close the window
vid_matches.release()
cap.release()
cv2.destroyAllWindows()
