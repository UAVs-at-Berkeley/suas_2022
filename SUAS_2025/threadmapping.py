from dronekit import connect, VehicleMode
import threading
import cv2
import time
import datetime
import utils

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

#NW_corner = LocationGlobalRelative(37.8724990, -122.3190522, 61)
#NE_corner = LocationGlobalRelative(37.8728674, -122.3177487, 61)
#SE_corner = LocationGlobalRelative(37.8712964, -122.3165900, 61)
#SW_corner = LocationGlobalRelative(37.8709873, -122.3179793, 61)


#cmds1 = scan_mission(NW_corner, NE_corner, SE_corner, SW_corner, 61)
#utils.write_missionlist("mapping.txt", cmds1)

sitl = None
cap = None
rtmp = None
video_maker = None

#if no connection string start sitl
if not connection_string:
    import dronekit_sitl
    sitl = dronekit_sitl.start_default()
    connection_string = sitl.connection_string()

# Connect to the Vehicle
print('Connecting to vehicle on: %s' % connection_string)
vehicle = connect(ip=connection_string, wait_ready=True, timeout=30, heartbeat_timeout=60, baud=115200)

# Thread stop signal
stop_event = threading.Event()


cmds = utils.downloadCommands(vehicle)

# Waypoint indices for which video should be saved (inclusive)
SAVE_FROM_WAYPOINT = 2
SAVE_TO_WAYPOINT = 7

def should_save_frames():
    """
    Return True if the current waypoint is in the save range.
    """
    current_wp = vehicle.commands.next
    #return SAVE_FROM_WAYPOINT <= current_wp <= SAVE_TO_WAYPOINT
    return True

# === DroneKit Parameter Monitor ===
def vehicle_param_monitor():

    while not stop_event.is_set():
        try:
            if vehicle.commands.next <= vehicle.commands.count-1:
                distance = utils.distance_to_current_waypoint(vehicle)
                print(f"[DroneKit] Distance to next waypoint: {distance:.2f} meters")
        except Exception as e:
            print(f"[Monitor] Error calculating distance: {e}")

        stop_event.wait(1.0)  # Wait 1 second, but allow immediate stop

# === RTSP Reader and Conditional MP4 Writer ===
def rtsp_reader_and_saver():
    stream_url = "rtsp://your_rtsp_stream"
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("[RTSP] Failed to open stream.")
        stop_event.set()
        return

    # Get frame size and FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Fallback

    print(f"[RTSP] Frame size: {width}x{height}, FPS: {fps}")

    # Initially no VideoWriter
    out = None
    saving = False

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        # Check whether we should be saving
        if should_save_frames():
            if not saving:
                # Start recording
                out = cv2.VideoWriter(f'output_wp_{SAVE_FROM_WAYPOINT}_to_{SAVE_TO_WAYPOINT}.mp4',
                                      cv2.VideoWriter_fourcc(*'mp4v'),
                                      fps,
                                      (width, height))
                saving = True
                print(f"[RTSP] Started saving video at waypoint {vehicle.commands.next}")
        else:
            if saving:
                # Stop recording
                out.release()
                out = None
                saving = False
                print(f"[RTSP] Stopped saving video at waypoint {vehicle.commands.next}")

        # If saving, write frame
        if saving and out is not None:
            out.write(frame)

    cap.release()
    if out is not None:
        out.release()
    print("[RTSP] Stream and file writer stopped.")

# === Main Control ===
if __name__ == '__main__':
    try:
        # Start threads
        video_thread = threading.Thread(target=rtsp_reader_and_saver)
        vehicle_thread = threading.Thread(target=vehicle_param_monitor)

        video_thread.start()
        vehicle_thread.start()

        # Mission monitoring loop
        while not stop_event.is_set():
            if (vehicle.mode.name == "AUTO" and
                vehicle.commands.next >= vehicle.commands.count and not vehicle.armed):
                print("[Mission] Completed.")
                stop_event.set()
            stop_event.wait(1)

    except KeyboardInterrupt:
        print("[Main] Interrupted by user.")
        stop_event.set()

    finally:
        video_thread.join()
        vehicle_thread.join()
        vehicle.close()
        print("[Main] All threads stopped and vehicle disconnected.")
