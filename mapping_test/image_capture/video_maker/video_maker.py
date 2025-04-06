from time import sleep
from multiprocessing import Process
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
        VideoCapture capture: a opencv videocaputre object 
        str parallelism(default = "threading"): a string corresponding to the type of parallelism to use, if a string other than "multiprocessing" is passed, uses Threading
    '''
    def __init__(self, capture, parallelism = "threading"):
        self.recording_flag = True
        self.parallelism = parallelism
        self.capture = capture
        self.parallel_process = None

        frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.capture.get(cv2.CAP_PROP_FPS)

        i = 0
        while os.path.exists(f'{PATH_OF_SCRIPT}/video_{i}.avi'):
            i += 1

        self.data = cv2.VideoWriter(f'{PATH_OF_SCRIPT}/video_{i}.avi', FOURCC, fps, (frame_width, frame_height))

    def recording_process(self):
        while self.recording_flag:
            ret, frame = self.capture.read()
            self.data.write(frame)

    def start(self):
        if self.parallel_process != None:
            raise Exception("Cannot start video recording when it has already started")
        if self.parallelism == "multiprocessing":
            writing_loop = Process(target= self.recording_process)
        else: 
            writing_loop = Thread(target= self.recording_process)

        self.parallel_process = writing_loop
        writing_loop.start()

    def stop(self):
        self.recording_flag = False
        if self.parallel_process is not None:
            self.parallel_process.join()
        self.data.release()

def basic_sanity_test():
    '''
    takes a 3 second video from the camera
    '''
    capture = cv2.VideoCapture(RSTP_URL)
    video_maker = VideoMaker(capture)
    video_maker.start()
    sleep(3)
    video_maker.stop()
    capture.release()

if __name__ == '__main__':
    basic_sanity_test()
