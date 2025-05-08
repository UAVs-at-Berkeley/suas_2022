#!/usr/bin/env python3

import cv2
import time

# Initialize the SIFT detector
orb = cv2.ORB_create()
#"rtspsrc location='rtsp://192.168.144.25:8554/main.264' protocols=tcp ! rtph265depay ! avdec_h265 ! nvvideoconvert ! appsink"
# gst-launch-1.0 rtspsrc location=rtsp://192.168.144.25:8554/main.264 protocols=tcp latency=0 ! rtph265depay ! h265parse ! nvv4l2decoder ! nvvideoconvert ! queue ! autovideosink
# RTSP stream URL decodebin ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! fakesink
gst_pipeline = "rtspsrc location=rtsp://192.168.144.25:8554/main.264 protocols=tcp latency=0 ! queue ! rtph265depay ! h265parse ! nvv4l2decoder ! nvvidconv ! queue ! appsink"


rtsp_url = "rtsp://192.168.144.25:8554/main.264"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)


if not cap.isOpened():
    print("Error: Unable to open the video stream.")
    exit()

ct = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # if not ret:
    #     print("Error: Failed to read frame.")
    #     break
    ct += 1

    #ret = cap.grab()
    if (ct >= 30):
        ct = 0
        #ret, frame = cap.retrieve()
        if not ret:
            print("error: failed to read frame :\(")
            break

        # Convert the frame to grayscale (ORB works on grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        # Draw keypoints on the frame
        frame_with_keypoints = cv2.drawKeypoints(gray, keypoints, frame)

        # Display the resulting frame
        cv2.imshow('ORB Keypoints', frame_with_keypoints)
            
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
