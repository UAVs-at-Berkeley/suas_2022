import cv2
import multiprocessing as mp
from multiprocessing import set_start_method, Queue, Event

def main():

    set_start_method("spawn")
    frames_queue = mp.Queue()
    stop_switch = mp.Event()

    reader = mp.Process(target=Reader, args=(frames_list,), daemon=True)
    consumer = mp.Process(target=Consumer, args=(frames_list, stop_switch), daemon=True)

    reader.start()
    consumer.start()

    stop_switch.wait()

import cv2

def Reader(thing):
    cap = cv2.VideoCapture('rtsp_address')

    while True:
        ret, frame = cap.read()
        if ret:
            try:
                # discard possible previous (unprocessed) frame
                frames_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                frames_queue.put(cv2.resize(frame, (1080, 720)), block=False)
            except:
                pass

def Consumer(frames_queue, stop_switch):

    while True:
        frame = frames_queue.get()     ## get current camera frame from queue
        ## do something computationally intensive on frame
        cv2.imshow('output', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        ## stop system when pressing 'q' key
        key = cv2.waitKey(1)
        if key==ord('q'):
            stop_switch.set()
            break

if __name__ == "__main__":
    main()

def write(stack, cam, top: int) -> None:
    """
         :param cam: camera parameters
         :param stack: Manager.list object
         :param top: buffer stack capacity
    :return: None
    """
    print('Process to write: %s' % os.getpid())
    cap = cv2.VideoCapture(cam)
    while True:
        _, img = cap.read()
        if _:
            stack.append(img)
            # Clear the buffer stack every time it reaches a certain capacity
            # Use the gc library to manually clean up memory garbage to prevent memory overflow
            if len(stack) >= top:
                del stack[:]
                gc.collect()


# Read data in the buffer stack:
def read(stack) -> None:
    print('Process to read: %s' % os.getpid())
    while True:
        if len(stack) != 0:
            value = stack.pop()
            cv2.imshow("img", value)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

if __name__ == '__main__':
    # The parent process creates a buffer stack and passes it to each child process:
    q = Manager().list()
    pw = Process(target=write, args=(q, "rtsp://xxx:xxx@192.168.1.102:554", 100))
    pr = Process(target=read, args=(q,))
    # Start the child process pw, write:
    pw.start()
    # Start the child process pr, read:
    pr.start()

    # Wait for pr to end:
    pr.join()

    # pw Process is an infinite loop, can not wait for its end, can only be forced to terminate:
    pw.terminate()