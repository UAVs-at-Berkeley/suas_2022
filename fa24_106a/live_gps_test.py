import cv2
import math
import time

still_image_dict = {
    0:('37.872310N_122.322454W_231.23H_297.8W.png', 37.872310, 122.322454, 231.23, 297.8), 
    1:('37.872312N_122.319072W_235.56H_364.08W.png', 37.872312, 122.319072, 235.56, 364.08), 
    2:('37.874496H_122.322454W_242.73H_297.8W.png', 37.874496, 122.322454, 242.73, 297.8), 
    3:('37.874496N_122.319072W_242.73H_364.08W.png', 37.874496, 122.319072, 242.73, 364.08)
}
r_earth = 6378000
drone_alt = 50
h_fov = 71.5
d_fov = 79.5
cam_size = (2560, 1440)
cam_x = 2*(math.tan(h_fov*math.pi/2/180)*drone_alt)
print(cam_x)
cam_y = math.sqrt(4*((math.tan(79.5*math.pi/2/180))**2)*(drone_alt**2)-(cam_x**2))
print(cam_y)

cam_x_size = cam_x / cam_size[0]
print(cam_x_size)
cam_y_size = cam_y / cam_size[1]
print(cam_y_size)

# 1. Load the still image and the video
still_image = cv2.imread(still_image_dict[0][0], cv2.IMREAD_GRAYSCALE)
video_path = 'lowaltflight.mp4'
horizontal_size = still_image.shape[:2][1]
print(horizontal_size)
vertical_size = still_image.shape[:2][0]
print(vertical_size)
x_size = still_image_dict[0][4] / horizontal_size
print(x_size)
y_size = still_image_dict[0][3] / vertical_size
print(y_size)

# 2. Detect keypoints and descriptors in the still image using ORB
orb = cv2.ORB_create(nfeatures=50)
kp_still, des_still = orb.detectAndCompute(still_image, None)

still_kps = cv2.drawKeypoints(still_image, kp_still, None, color=(0,255,0), flags=0)


# 3. Initialize the FLANN-based matcher
index_params = dict(algorithm=6,  # FLANN_INDEX_LSH for ORB
                    table_number=6,  # number of hash tables
                    key_size=12,     # size of the hashed key
                    multi_probe_level=1)  # multi-probe level
search_params = dict(checks=50)  # number of checks (higher is more accurate)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# 4. Open the video file
cap = cv2.VideoCapture(video_path)

orb2 = cv2.ORB_create(nfeatures=50)

# 5. Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    cv2.resize(frame, (100, 100))
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 6. Detect keypoints and descriptors in the frame
    kp_frame, des_frame = orb2.detectAndCompute(gray_frame, None)
    frame_kps = cv2.drawKeypoints(gray_frame, kp_frame, None, color=(0,255,0), flags=0)

    # 7. Match descriptors using FLANN
    if des_frame is not None:
        matches = flann.knnMatch(des_still, des_frame, k=2)

        # 8. Apply the ratio test to filter matches (Lowe's ratio test)
        good_matches = []
        for m in matches:
            if len(m) == 2:  # Ensure that we have two matches
                # Apply Lowe's ratio test
                if m[0].distance < 0.7 * m[1].distance:
                    #print(m[0].queryIdx)
                    #print(m[1].queryIdx)
                    #print(m[0].trainIdx)
                    #print(m[1].trainIdx)
                    #query_idx = 0#m.queryIdx
                    #train_idx = 0#m.trainIdx 537,935
                    still_pt = kp_still[m[0].queryIdx].pt
                    frame_pt = kp_frame[m[1].trainIdx].pt
                    print(still_pt)
                    print(frame_pt)
                    
                    print(still_pt[1]*y_size)
                    print(still_pt[0]*x_size)
                    print((still_pt[1]*y_size / r_earth) * (180 / math.pi))
                    still_gps_lat = still_image_dict[1][1] - (still_pt[1]*y_size / r_earth) * (180 / math.pi)
                    still_gps_long = still_image_dict[1][2] - ((still_pt[0]*x_size / r_earth) * (180 / math.pi) / math.cos(still_image_dict[1][1]*math.pi/180))
                    
                    print(((frame_pt[1] - cam_size[1]/2)*cam_y_size))
                    print(((frame_pt[0] - cam_size[0]/2)*cam_x_size))
                    print((((frame_pt[1] - cam_size[1]/2)*cam_y_size)/ r_earth) * (180 / math.pi))
                    
                    cam_gps_lat = still_gps_lat - (((frame_pt[1] - cam_size[1]/2)*cam_y_size)/ r_earth) * (180 / math.pi)
                    cam_gps_long = still_gps_long + (((frame_pt[0] - cam_size[0]/2)*cam_x_size) / r_earth) * (180 / math.pi) / math.cos(still_gps_lat*math.pi/180)

                    print((still_gps_long, still_gps_lat))
                    print((cam_gps_long, cam_gps_lat))
                    cv2.putText(still_kps, text=str((still_gps_long, still_gps_lat)) + " " + str((int(still_pt[0]), int(still_pt[1]))), org=(int(still_pt[0]), int(still_pt[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2, lineType=cv2.LINE_AA)
                    cv2.putText(frame_kps, text=str((cam_gps_long, cam_gps_lat)), org=(int(frame_pt[0]), int(frame_pt[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2, lineType=cv2.LINE_AA)
                    good_matches.append(m[0])
                    

        # 9. Draw the matches
        matched_img = cv2.drawMatches(still_image, kp_still, frame, kp_frame, good_matches, None, flags=cv2.DrawMatchesFlags_DEFAULT)

        # Show the matched image
        cv2.imshow('Matches', matched_img)
        cv2.imshow("Still image key points", still_kps)
        cv2.imshow("Frame image key points", frame_kps)

        # Press 'q' to quit the video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
