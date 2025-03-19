import cv2
import threading

class VideoStream:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.stream = None
        self.frame = None

    def get_frame(self):
        ret, frame = self.stream.read()
        return frame

    def start_stream(self): # start thread
        self.stream = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

    def end_stream(self):
        self.stream.release()
    

