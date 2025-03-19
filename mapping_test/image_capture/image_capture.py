import cv2
import os
import numpy as np
from time import sleep
import threading

# from stream import SIYIRTSP
# from siyi_sdk import SIYISDK
from new_mapping_library import VideoCaptureWrapper

PATH_OF_SCRIPT = os.path.dirname(os.path.abspath(__file__)) #local directory, NOT the working directory
DUMMY_IMAGE = np.full((100, 100, 3), 255, dtype=np.uint8)

CAMERA_SERVER_IP = "192.168.144.25"
CAMERA_PORT = 37260
RSTP_URL = "rtsp://192.168.144.25:8554/main.264"

# def capture_image_and_save(stream = None, dimensions = (None, None)):    
#     i = 0

#     # cam = SIYISDK(server_ip=CAMERA_SERVER_IP, port=CAMERA_PORT)
    
#     # if not cam.connected():
#     #     print("No connection ")
#     #     exit(1)
#     ret, frame = stream.read()
#     # Get camera name
#     # cam_str = cam.getCameraTypeString()
#     # cam.disconnect()
        
#     # rtsp = SIYIRTSP(rtsp_url=RSTP_URL, debug=False, cam_name="")
#     # if dimensions[0]:
#     #     rtsp.resize(dimensions[0], dimensions[1])

#     # frame = rtsp.getFrame()

#     while os.path.exists(f"{PATH_OF_SCRIPT}/{i}.png"):
#         i += 1
#     cv2.imwrite(f'{PATH_OF_SCRIPT}/{i}.png', frame)


# def capture_image_and_save(dimensions = (None, None)):    
#     i = 0

#     cam = SIYISDK(server_ip=CAMERA_SERVER_IP, port=CAMERA_PORT)
#     cam.connect()

#     if not cam.isConnected():
#         print("No connection ")
#         exit(1)

#     # Get camera name
#     cam_str = cam.getCameraTypeString()
#     cam.disconnect()
        
#     rtsp = SIYIRTSP(rtsp_url=RSTP_URL, debug=False, cam_name=cam_str)
#     if dimensions[0]:
#         rtsp.resize(dimensions[0], dimensions[1])

#     rtsp.start()
#     frame = rtsp.getFrame()

#     while os.path.exists(f"{PATH_OF_SCRIPT}/{i}.png"):
#         i += 1
#     cv2.imwrite(f'{PATH_OF_SCRIPT}/{i}.png', frame)


def capture_image_and_save(existing_video_capture, coordinates = None):
    """
    Captures an image from the provided video capture via RTSP and saves to the directory of this file \n
    If no coordinates are provided, saves the image with the next available integer as the file name
    """
    ret, frame = existing_video_capture.read()

    if coordinates:
        cv2.imwrite(f'{PATH_OF_SCRIPT}/{str(coordinates)}.png', frame)
    else:
        i = 0
        while os.path.exists(f'{PATH_OF_SCRIPT}/{i}.png'):
            i += 1
        cv2.imwrite(f'{PATH_OF_SCRIPT}/{i}.png', frame)

if __name__ == '__main__':
    cap = cv2.VideoCapture(RSTP_URL)
    sleep(3)
    capture_image_and_save(cap)
    capture_image_and_save(cap)
    capture_image_and_save(cap)

