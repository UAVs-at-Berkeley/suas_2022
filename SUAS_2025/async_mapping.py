import asyncio
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

lastwaypoint=13

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
mission_term = True

async def drone_control():
    # if no connection string start sitl
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

    cmds = utils.downloadCommands(vehicle)

    #Wait until vehicle is armable
    counter = 0
    while not vehicle.is_armable:
        # If cannot acheive armable in 120 seconds, reboot the autopilot
        if counter == 1200:
            vehicle.reboot()
        print("Waiting for vehicle to initialise...")
        counter += 1

        if KeyboardInterrupt:
            if show_stream:
                rtmp.stop()
            if vid_mapping:
                video.stop()
            cap.release()
            cv2.destroyAllWindows()
            # quit
            exit(0)
        asyncio.sleep(1)
    
    while vehicle.mode != VehicleMode("AUTO"):
        print("Currently in manual mode... Waiting for pilot to switch to AUTO")
        asyncio.sleep(3)
    
    print("Entered AUTO mode")
    vehicle.gimbal.rotate(-90, 0, 0)
    try:
    while True:
        nextwaypoint=vehicle.commands.next
        if not nextwaypoint:
            break
        print('Distance to waypoint (%s): %s' % (nextwaypoint, distance_to_current_waypoint()))

        asyncio.sleep(3)
    
    utils.RTL(vehicle)

    vehicle.gimbal.rotate(0, 0, 0)

    while vehicle.armed:
        print("Returning to land. Will terminate once landed.")
        asyncio.sleep(3)
    
    mission_term = False
    print("Close vehicle object")
    vehicle.close()

    if sitl is not None:
        sitl.stop()

async def camera():
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("No connection")
        exit(1)

    if show_stream:
        rtmp = RTMP.RTMPSender(rtmp_url)
        rtmp.start()

    if vid_mapping:
        video = VideoMaker(cap)
        video.start()

    if show_stream:
        ret, frame = cap.read()
        rtmp.setFrame(frame)
        imcap.image_save(frame)
    try:
        while True:
            if not mission_term:
                break
            frame = None
            if show_stream:
                ret, frame = cap.read()
                rtmp.setFrame(frame)
            await asyncio.sleep(0)
    except KeyboardInterrupt:
        if show_stream:
            rtmp.stop()
        if vid_mapping:
            video.stop()
        cap.release()
        cv2.destroyAllWindows()
        # quit
        exit(0)

    if show_stream:
        rtmp.stop()

    if vid_mapping:
        video.stop()

    cap.release()
    cv2.destroyAllWindows()
    

async def yolo():
    for letter in 'abcdefghij':
        print(f"Letter: {letter}")
        await asyncio.sleep(1)  # Non-blocking sleep


async def main():
    camera = asyncio.create_task(camera())
    drone = asyncio.create_task(drone_control())  # Run coroutines concurrently

    await drone
    await camera

# Run the main coroutine
asyncio.run(main())
