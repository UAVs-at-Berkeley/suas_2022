import cv2

class VideoCaptureWrapper:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.video_capture = None
        self.frame = None

    def get_frame(self):
        ret, frame = self.stream.read()
        return frame

    def start_new_capture(self): 
        self.video_capture = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
    
    def add_video_capture_reference(self, stream):
        self.video_capture = stream

    def end_video_capture(self):
        self.video_capture.release()
