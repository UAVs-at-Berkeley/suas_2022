import os
import zoom_image
import find_contour
import cv2
import matplotlib
import argparse
import color_detection

#Take a photo with the camera attached
def take_photo():
    cap = cv2.VideoCapture()
    ret, image = cap.read()
    cv2.write(" ", image)
    cap.release()

#Run yolov5 on image from camera to detect possible objects
def yolo(source_path):
    os.system(f"python detect.py --weights yolov5s.pt --img 416 --conf 0.4 --source \"{source_path}\" --save-crop --save-txt")

#Run zoom on all of the cropped images of possible objects

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--img_path', type=str, help='The path of the image you want to zoom.')
    parser.add_argument('-n', '--img_name', type=str, help='The name of the new zoomed image.')
    args = parser.parse_args()
    detected = yolo(args.img_path)
    if detected > 0:
        directory = 
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            img = zoom_image.zoom(f"{directory}/{filename}.jpg", filename)
            contour = find_contour.find_largest_contour(img)
            colors = {'color_shape': 'Gold', 'color_alphanumeric': 'Gold'}
            color_max = color_detection.color_det(img)
            colors['color_shape'] = color_max[0]
            colors['color_alphanumeric'] = color_max[1]
            cv2.imwrite(path1, color_max[2])
            cv2.imwrite(path2, color_max[23])
            shape_det = yolo(path1)
            alpha_det = yolo(path2)
