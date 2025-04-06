import time
from dronekit import connect, VehicleMode, LocationGlobalRelative, mavutil, Command
import numpy
import math
import vehicle_state as vs
import utils
import cv2

# Set up option parsing to get connection string
import argparse
parser = argparse.ArgumentParser(description='Commands vehicle using vehicle.simple_goto.')
parser.add_argument('-c', '--connect', nargs='?', const="/dev/ttyACM0", type=str, default="/dev/ttyACM0",
                    help="Vehicle connection target string. If not specified, SITL automatically started and used.")
parser.add_argument('-v', '--verbose', action="store_true",
                    help="Verbose flag prints out all vehicle state parameters upon connection to autopilot.")
parser.add_argument('-s', '--stream', action="store_true",
                    help="Set up RTMP livestream of camera feed")
parser.add_argument('-rts', '--rtsp', nargs='?', const="rtsp://192.168.144.25:8554/main.264", type=str, default="rtsp://192.168.144.25:8554/main.264",
                   help="RTSP connection string. By default rtsp://192.168.144.25:8554/main.264 is used")
parser.add_argument('-rtm', '--rtmp', nargs='?', const="rtmp://127.0.0.1:1935/live/webcam", type=str, default="rtmp://127.0.0.1:1935/live/webcam",
                   help="RTMP connection string. By default rtmp://127.0.0.1:1935/live/webcam is used")
args = parser.parse_args()

connection_string = args.connect
verbose = args.verbose
show_stream = args.stream
rtsp_url = args.rtsp
rtmp_url = args.rtmp

sitl = None
rtmp = None

#if no connection string start sitl
if not connection_string:
    import dronekit_sitl
    sitl = dronekit_sitl.start_default()
    connection_string = sitl.connection_string()

# Connect to the Vehicle
print('Connecting to vehicle on: %s' % connection_string)
vehicle = connect(ip=connection_string, wait_ready=True, timeout=30, heartbeat_timeout=60, baud=115200)
# wait_ready: If ``True`` wait until all default attributes have downloaded before the method returns (default is ``None``).
#             The default attributes to wait on are: :py:attr:`parameters`, :py:attr:`gps_0`, :py:attr:`armed`, :py:attr:`mode`, and :py:attr:`attitude`.
# timeout: timeout in seconds for wait_ready, aka time to wait for attributes to download from autopilot before throwing exception
# heartbeat_timeout: time to wait in seconds for heartbeat connection with autopilot

if verbose:
    vs.print_vehicle_state(vehicle)

cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("No connection")
    exit(1)
    
if show_stream:
    rtmp = RTMPSender(rtmp_url)
    rtmp.start()

#Wait until vehicle is armable
counter = 0
while not vehicle.is_armable:
    # If cannot acheive armable in 120 seconds, reboot the autopilot
    if counter == 120:
        vehicle.reboot()
    print("Waiting for vehicle to initialise...")
    counter += 1
    time.sleep(1)

rtmp.stop()
cap.release()
cv2.destroyAllWindows()

print("Close vehicle object")
vehicle.close()

if sitl is not None:
    sitl.stop()