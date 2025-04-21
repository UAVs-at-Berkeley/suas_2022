from time import sleep
from threading import Thread
import cv2
import os

PATH_OF_SCRIPT = os.path.dirname(os.path.abspath(__file__)) #local directory, NOT the working directory
RSTP_URL = "rtsp://192.168.144.25:8554/main.264"
FOURCC = cv2.VideoWriter_fourcc(*'H264')

class VideoMaker:
    '''
    A class that creates mp4 videos via a videocapture object \n
    Constructors: \n
        VideoCapture stream: a opencv videocaputre object 
        str parallelism(default = "threading"): a string corresponding to the type of parallelism to use, if a string other than "multiprocessing" is passed, uses Threading
    '''
    def __init__(self, stream):
        self.recording_flag = True
        self.stream = stream
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

        frame_width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.stream.get(cv2.CAP_PROP_FPS)
        fourcc2 = self.stream.get(CAP_PROP_FOURCC)

        i = 0
        while os.path.exists(f'{PATH_OF_SCRIPT}/video_{i}.avi'):
            i += 1

        self.data = cv2.VideoWriter(f'{PATH_OF_SCRIPT}/video_{i}.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_width, frame_height))

    def recording_process(self):
        while self.recording_flag:
            (self.grabbed, self.frame) = self.stream.read()
            if self.grabbed:
                self.data.write(self.frame)

    def start(self):
        Thread(target=self.recording_process, args=()).start()
        return self
        
    def stop(self):
        self.recording_flag = False
        self.data.release()

def basic_sanity_test():
    '''
    takes a 3 second video from the camera
    '''
    capture = cv2.VideoCapture(RSTP_URL)
    video_maker = VideoMaker(capture)
    video_maker.start()
    time.sleep(180)
    video_maker.stop()
    capture.release()

if __name__ == '__main__':
    basic_sanity_test()
