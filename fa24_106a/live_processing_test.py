import cv2

# 1. Load the still image and the video
still_image = cv2.imread('google_earth.png', cv2.IMREAD_GRAYSCALE)
video_path = 'google_earth.mov'

# 2. Detect keypoints and descriptors in the still image using ORB
orb = cv2.ORB_create(nfeatures=50)
kp_still, des_still = orb.detectAndCompute(still_image, None)

still_kps = cv2.drawKeypoints(still_image, kp_still, None, color=(0,255,0), flags=0)

cv2.imshow("Still image key points", still_kps)

# 3. Initialize the FLANN-based matcher
index_params = dict(algorithm=6,  # FLANN_INDEX_LSH for ORB
                    table_number=6,  # number of hash tables
                    key_size=12,     # size of the hashed key
                    multi_probe_level=1)  # multi-probe level
search_params = dict(checks=50)  # number of checks (higher is more accurate)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# 4. Open the video file
cap = cv2.VideoCapture(video_path)

orb2 = cv2.ORB_create(nfeatures=150)

# 5. Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 6. Detect keypoints and descriptors in the frame
    kp_frame, des_frame = orb2.detectAndCompute(gray_frame, None)

    # 7. Match descriptors using FLANN
    if des_frame is not None:
        matches = flann.knnMatch(des_still, des_frame, k=2)

        # 8. Apply the ratio test to filter matches (Lowe's ratio test)
        good_matches = []
        for m in matches:
            if len(m) == 2:  # Ensure that we have two matches
                # Apply Lowe's ratio test
                if m[0].distance < 0.7 * m[1].distance:
                    good_matches.append(m[0])

        # 9. Draw the matches
        matched_img = cv2.drawMatches(still_image, kp_still, frame, kp_frame, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Show the matched image
        cv2.imshow('Matches', matched_img)

        # Press 'q' to quit the video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
