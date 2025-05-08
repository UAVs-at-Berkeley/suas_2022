import cv2
import os
import numpy as np

PATH_OF_SCRIPT = os.path.dirname(os.path.abspath(__file__)) #local directory, NOT the working directory
CAMERA_SERVER_IP = "192.168.144.25"
CAMERA_PORT = 37260
RSTP_URL = "rtsp://192.168.144.25:8554/main.264"

def capture_image_and_save(existing_video_capture, coordinates = None):
    """
    Captures an image from the provided video capture via RTSP and saves to the directory of this file with the provided coordinates
    """
    ret, frame = existing_video_capture.read()
    cv2.imwrite(f'{PATH_OF_SCRIPT}/{str(coordinates[0])}_{str(coordinates[1])}.png', frame)
