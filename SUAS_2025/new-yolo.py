import cv2
import time
import time
from dronekit import connect, VehicleMode, LocationGlobalRelative, mavutil, Command
import numpy
import math
import vehicle_state as vs
import utils
import cv2
import image_capture_modified as imcap
import RTMP
from video_maker import VideoMaker
import os
from ultralytics import YOLO

PATH_OF_SCRIPT = os.path.dirname(os.path.abspath(__file__))

# Initialize the SIFT detector

#"rtspsrc location='rtsp://192.168.144.25:8554/main.264' protocols=tcp ! rtph265depay ! avdec_h265 ! nvvideoconvert ! appsink"
# gst-launch-1.0 rtspsrc location=rtsp://192.168.144.25:8554/main.264 protocols=tcp latency=0 ! rtph265depay ! h265parse ! nvv4l2decoder ! nvvideoconvert ! queue ! autovideosink
# RTSP stream URL decodebin ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! fakesink
gst_pipeline = "rtspsrc location=rtsp://192.168.144.25:8554/main.264 protocols=tcp latency=0 ! queue ! rtph265depay ! h265parse ! nvv4l2decoder ! nvvidconv ! queue ! appsink"

# Set up option parsing to get connection string
import argparse
parser = argparse.ArgumentParser(description='Commands vehicle using vehicle.simple_goto.')
parser.add_argument('-c', '--connect', nargs='?', const="/dev/ttyACM0", type=str, default="/dev/ttyACM0",
                    help="Vehicle connection target string. If not specified, SITL automatically started and used.")
parser.add_argument('-v', '--verbose', action="store_true",
                    help="Verbose flag prints out all vehicle state parameters upon connection to autopilot.")
parser.add_argument('-s', '--stream', action="store_true",
                    help="Set up RTMP livestream of camera feed")
parser.add_argument('-sg', '--stopgo', action="store_true",
                    help="If used, drone will stop at waypoints taking mapping photos")
parser.add_argument('-vid', '--video', action="store_true",
                    help="Used to determine if recording style is video stream (true/include flag) or singular images (false/do not include flag)")
parser.add_argument('-rts', '--rtsp', nargs='?', const="rtsp://192.168.144.25:8554/main.264", type=str, default="rtsp://192.168.144.25:8554/main.264",
                   help="RTSP connection string. By default rtsp://192.168.144.25:8554/main.264 is used")
parser.add_argument('-rtm', '--rtmp', nargs='?', const="rtmp://127.0.0.1:1935/live/webcam", type=str, default="rtmp://127.0.0.1:1935/live/webcam",
                   help="RTMP connection string. By default rtmp://127.0.0.1:1935/live/webcam is used")
args = parser.parse_args()

connection_string = args.connect
verbose = args.verbose
show_stream = args.stream
stop_go_mapping = args.stopgo
vid_mapping = args.video
rtsp_url = args.rtsp
rtmp_url = args.rtmp

sitl = None
cap = None
rtmp = None
video_maker = None

output_file = "geo.txt"

model = YOLO("best.onnx")
#model = YOLO("yolo11n.pt")

#if no connection string start sitl
if not connection_string:
    import dronekit_sitl
    sitl = dronekit_sitl.start_default()
    connection_string = sitl.connection_string()

print('Connecting to vehicle on: %s' % connection_string)
vehicle = connect(ip=connection_string, wait_ready=True, timeout=30, heartbeat_timeout=60, baud=115200)
# wait_ready: If ``True`` wait until all default attributes have downloaded before the method returns (default is ``None``).
#             The default attributes to wait on are: :py:attr:`parameters`, :py:attr:`gps_0`, :py:attr:`armed`, :py:attr:`mode`, and :py:attr:`attitude`.
# timeout: timeout in seconds for wait_ready, aka time to wait for attributes to download from autopilot before throwing exception
# heartbeat_timeout: time to wait in seconds for heartbeat connection with autopilot

if verbose:
    vs.print_vehicle_state(vehicle)

cmds = utils.downloadCommands(vehicle)

lastwaypoint = len(cmds)

print(lastwaypoint)

rtsp_url = "rtsp://192.168.144.25:8554/main.264"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Unable to open the video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    if frame.mean() > 10:  # Image is probably dark/gray
        print("Frame is now active")
        break
    print("Waiting for valid camera frame")

#if vid_mapping:
#    vid_matches = cv2.VideoWriter('mapping6.mp4', cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(3)), int(cap.get(4))))

#Wait until vehicle is armable
counter = 0
while not vehicle.is_armable:
    # If cannot acheive armable in 120 seconds, reboot the autopilot
    if counter == 1200:
        vehicle.reboot()
    if counter == 1000:
        print("Waiting for vehicle to initialise...")
        counter = 0
    counter += 1
counter = 0
#while vehicle.mode != VehicleMode("AUTO"):
#    counter += 1
#    if counter == 50000:
#        counter = 0
#        print("Currently in manual mode... Waiting for pilot to switch to AUTO")

ct = 0
nextct = 0
vehicle.gimbal.rotate(-90, 0, 0)
nextwaypoint=vehicle.commands.next
while nextwaypoint < 8:

    waypoint = utils.getCurrentWaypoint(vehicle)
    nextct += 1
    if nextwaypoint == lastwaypoint:
        break
    #if nextwaypoint >= 1:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # if not ret:
    #     print("Error: Failed to read frame.")
    #     break
    ct += 1

    #ret = cap.grab()
    #if (ct >= 5):
    #    ct = 0
        #ret, frame = cap.retrieve()
    #    if not ret:
    #        print("error: failed to read frame :\(")
    #        break

        # Convert the frame to grayscale (ORB works on grayscale images)
        

        # Detect keypoints and compute descriptors
        

        # Draw keypoints on the frame
       

        # Display the resulting frame
    #    vid_matches.write(frame)
    if nextct == 20:
        print('Distance to waypoint (%s): %s' % (nextwaypoint, utils.distance_to_current_waypoint(vehicle)))
        nextct = 0
    nextwaypoint=vehicle.commands.next        
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
ct = 0
nextct = 0
framect = 0
relaydrop = 0
with open(output_file, "w") as file:
    file.write("EPSG:4326\n")
vid_matches = cv2.VideoWriter('mappin9.mp4', cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(3)), int(cap.get(4))))
yolo_matches = cv2.VideoWriter('yolo_match9.mp4', cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(3)), int(cap.get(4))))
while nextwaypoint < lastwaypoint-1:
    ret, frame = cap.read()
    ct += 1
    nextct += 1
    if nextwaypoint == lastwaypoint:
        break
    waypoint = utils.getCurrentWaypoint(vehicle)
    if (ct >= 10):
        ct = 0
        if not ret:
            print("error: failed to read frame :\(")
            continue

        # Convert the frame to grayscale (ORB works on grayscale images)


        # Detect keypoints and compute descriptors


        # Draw keypoints on the frame
        
        with open(output_file, "a") as f:
            f.write(f"frame_{framect}.png {str(waypoint.lon)} {str(waypoint.lat)} {str(waypoint.alt)}\n")


        framect += 1
        # Display the resulting frame
        vid_matches.write(frame)

        if nextwaypoint < 15:
            results = model(frame)
            annotated_frame = results[0].plot()

            yolo_matches.write(annotated_frame)

            for result in results[0].boxes:
                #x1, y1, x2, y2 = results.ayay[0].tolist()
                conf = result.conf[0]
                class_index = result.cls[0].item()
                class_name = results[0].names[int(class_index)]
                if class_name == "Person-Mannequin" and relaydrop < 2 and conf > 0.5:
                    utils.setRelay(vehicle=vehicle, num=relaydrop, state=1)
                    relaydrop+=1
    nextwaypoint=vehicle.commands.next



while vehicle.armed:
    print("Returning to land. Will terminate once landed.")
    time.sleep(3)

# Release the video capture object and close all OpenCV windows
cap.release()
vid_matches.release()
yolo_matches.release()
cv2.destroyAllWindows()

print("Close vehicle object")
vehicle.close()
