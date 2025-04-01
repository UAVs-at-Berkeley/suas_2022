from image_capture_modified import capture_image_and_save
from image_stitcher import stich_all
from random import choice
import cv2
import os
import numpy as np

PATH_OF_SCRIPT = os.path.dirname(os.path.abspath(__file__)) #local directory, NOT the working directory
RSTP_URL = "rtsp://192.168.144.25:8554/main.264"

def test():
    video_capture = cv2.VideoCapture(RSTP_URL)
    ensure_capture_not_gray(video_capture)
    capture_image_and_save(video_capture, (0, 0))
    capture_image_and_save(video_capture, (0, 1))
    capture_image_and_save(video_capture, (0, 2))
    capture_image_and_save(video_capture, (1, 0))
    capture_image_and_save(video_capture, (1, 1))
    capture_image_and_save(video_capture, (1, 2))
    capture_image_and_save(video_capture, (2, 0))
    capture_image_and_save(video_capture, (2, 1))
    capture_image_and_save(video_capture, (2, 2))
    video_capture.release()

def test_dummy_photos_1():
    for row in range(3):
        for col in range(3):
            frame = cv2.imread(f'{PATH_OF_SCRIPT}/3x3_runway/({row}, {col}).png')
            cv2.imwrite(f'{PATH_OF_SCRIPT}/({row}, {col}).png', frame)

def test_dummy_photos_2():
    for row in range(4):
        for col in range(3):
            frame = cv2.imread(f'{PATH_OF_SCRIPT}/4x3_field/({row}, {col}).png')
            cv2.imwrite(f'{PATH_OF_SCRIPT}/({row}, {col}).png', frame)

def ensure_capture_not_gray(video_capture):
    ret, frame = video_capture.read()
    while choice(choice(frame)) == np.array([129, 129, 129]) and choice(choice(frame)) == np.array([129, 129, 129]) and choice(choice(frame)) == np.array([129, 129, 129]):
        pass

if __name__ == '__main__':
    test_dummy_photos_1()
    stich_all(3, 3)

    #test_dummy_photos_2()
    #stich_all(3, 3)

