from video_maker import VideoMaker
import cv2
from time import sleep

RSTP_URL = "rtsp://192.168.144.25:8554/main.264"

if __name__ == '__main__':
    cap = cv2.VideoCapture(RSTP_URL)
    maker = VideoMaker(cap)

    maker.start() # start recording
    sleep(2) # woo epic fight hell yeah
    sleep(1) # images are being taken or smthing
    sleep(2) # more epic flight also mathew's other drone has entered low-earth orbit
    sleep(1) # landing
    maker.stop() # stop recording, video should be saved to the local directory

    cap.release()
