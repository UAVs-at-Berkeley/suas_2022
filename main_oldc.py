import os
import zoom_image
import cv2
import matplotlib
import argparse

#Take a photo with the camera attached
def take_photo():
    cap = cv2.VideoCapture()
    ret, image = cap.read()
    cv2.write(" ", image)
    cap.release()

#Run yolov5 on image from camera to detect possible objects
def yolo():
    os.system("python detect.py --weights yolov5s.pt --img 416 --conf 0.4 --source 0 --save-crop --save-txt")

#Run zoom on all of the cropped images of possible objects

def main():
    zoom_image.zoom()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--img_path', type=str, help='The path of the image you want to zoom.')
    parser.add_argument('-n', '--img_name', type=str, help='The name of the new zoomed image.')
    args = parser.parse_args()
    main(arguments)