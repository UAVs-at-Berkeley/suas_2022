#!/usr/bin/env python3

import cv2
import time

# Initialize the SIFT detector

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

vid_matches = cv2.VideoWriter('mapping2.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, (int(cap.get(3)), int(cap.get(4))))
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
        

        # Detect keypoints and compute descriptors
        

        # Draw keypoints on the frame
       

        # Display the resulting frame
        vid_matches.write(frame)
        
            
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
