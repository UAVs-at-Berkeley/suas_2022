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

#Given a 4-coordinate box, generates out equally spaced waypoint mission for photos to map area
# Give coordinates in order: Northwest corner, Northeast Corner, Southeast Corner, Southwest Corner
# delay is how long to wait at each coordinate to take a photo, 0 is when it is live footage or waiting isn't desired
# In deciding which corner is which, give priority to the vertical direction. This algorithm is configured so that 
# the two highest latitude coordinates are always the North coordinates
#TODO: Check coordinates myself to make sure NW, NE, SW, SE are correctly chosen and that altitude is same as drone_alt
def scan_mission(NW_coord, NE_coord, SE_coord, SW_coord, drone_alt, cam_h_fov=71.5, cam_d_fov=79.5, delay=10):

    cam_x, cam_y = utils.cam_params(alt=drone_alt, h_fov=cam_h_fov, d_fov=cam_d_fov)
    print(f"cam_x is {cam_x}m")
    print(f"cam_y is {cam_y}m")
    #cam_x will always be > cam_y
    NS_height = max(utils.get_distance_metres(NW_coord, SW_coord), utils.get_distance_metres(NE_coord, SE_coord))
    EW_width = max(utils.get_distance_metres(NW_coord, NE_coord), utils.get_distance_metres(SW_coord, SE_coord))
    NSEW_diag = max(utils.get_distance_metres(NW_coord, SE_coord), utils.get_distance_metres(NE_coord, SW_coord))

    # configure yaw to be locked in direction that minimizes the number of pictures
    # yaw should be perpendicular to the longest side since the horizontal/x field of view is wider and we want it parallel to the longest side 
    # while we want the front of the drone (vertical/y field of view) perpendicular to the longest side


    if NS_height >= EW_width:
        recc_yaw = utils.get_yaw_degrees(NW_coord, NE_coord)
        num_NS_pic = NS_height // cam_x + 1
        num_EW_pic = EW_width // cam_y + 1
        NS_image_size = NS_height / num_NS_pic
        print(f"The original NS_image_size is {NS_image_size}m")
        EW_image_size = EW_width / num_EW_pic
        print(f"The original EW_image_size is {EW_image_size}m")
        #Check percentage overlap is more than 10% on each side
        while NS_image_size > (0.95*cam_x):
            num_NS_pic += 1
            NS_image_size = NS_height / num_NS_pic
        while EW_image_size > (0.95*cam_y):
            num_EW_pic += 1
            EW_image_size = EW_width / num_EW_pic

    else:
        recc_yaw = utils.get_yaw_degrees(NW_coord, SW_coord)
        num_NS_pic = NS_height // cam_y + 1
        num_EW_pic = EW_width // cam_x + 1
        NS_image_size = NS_height / num_NS_pic
        print(f"The original NS_image_size is {NS_image_size}m")
        EW_image_size = EW_width / num_EW_pic
        print(f"The original EW_image_size is {EW_image_size}m")
        #Check percentage overlap is more than 10% on each side
        while NS_image_size > (0.95*cam_y):
            num_NS_pic += 1
            NS_image_size = NS_height / num_NS_pic
        while EW_image_size > (0.95*cam_x):
            num_EW_pic += 1
            EW_image_size = EW_width / num_EW_pic

    
    tot_pic_num = num_NS_pic * num_EW_pic
    print(f"The final NS_image_size is {NS_image_size}m")
    print(f"The final EW_image_size is {EW_image_size}m")
    print(f"The total number of pictures at this altitude for this area is {tot_pic_num} pictures")
    print(f"The recommended yaw while taking pictures is {recc_yaw} degrees")

    #make yaw relative to 90 degree angle and calculate multiplier on distance
    relative_yaw = recc_yaw
    angle_mult = 1
    if relative_yaw >= 360:
        relative_yaw = relative_yaw % 360
    if 45 >= relative_yaw >= 0:
        angle_mult = abs(math.cos(relative_yaw))
    elif 90 >= relative_yaw > 45:
        angle_mult = abs(math.cos(90-relative_yaw))
    elif 135 > relative_yaw > 90:
        angle_mult = abs(math.cos(relative_yaw-90))
    elif 180 >= relative_yaw >= 135:
        angle_mult = abs(math.cos(180-relative_yaw))
    elif 225 >= relative_yaw > 180:
        angle_mult = abs(math.cos(relative_yaw-180))
    elif 270 >= relative_yaw > 225:
        angle_mult = abs(math.cos(270 - relative_yaw))
    elif 315 > relative_yaw > 270:
        angle_mult = abs(math.cos(relative_yaw - 270))
    elif 360 >= relative_yaw >= 315:
        angle_mult = abs(math.cos(360 - relative_yaw))

    #d_North and d_East can be negative if the change is to the South or West respectively
    d_North = NS_image_size*angle_mult / 2
    d_East = EW_image_size*angle_mult / 2

    #based on the where the Northwestern corner is relative to the other corners (determine by relative yaw) d_North and d_East may need -1 multiplier
    if 90 >= relative_yaw >= 0:
        d_North = d_North*-1
    elif 180 > relative_yaw > 90:
        d_North = d_North*-1
        d_East = d_East*-1
    elif 270 >= relative_yaw >= 180:
        d_North = d_North*-1
    elif 360 > relative_yaw > 270:
        d_North = d_North*-1
        d_East = d_East*-1
    
    print(f"The d_North for half of a photo is {d_North}m")
    print(f"The d_East for half of a photo is {d_East}m")

    cmds = []

    #add yaw command
    cmds.append(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0, 1, recc_yaw, 0, 0, 0, 0, 0, 0))

    # Create a list of commands that sets the yaw and directs drone list of coordinates to take photos
    # off of NW corner of box add only 1*d_North, for every subsequent distance at (2*d_Nmult + 1)*d_North to account for drone being centered in photo
    # starting in NW corner of box
    for d_Nmult in range(int(num_NS_pic)):
        if d_Nmult % 2 == 0:
            for d_Emult in range(int(num_EW_pic)):
                coord = utils.get_location_metres(NW_coord, d_North*(2*d_Nmult+1), d_East*(2*d_Emult+1))
                cmds.append(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 1, delay, 0, 0, 0, coord.lat, coord.lon, coord.alt))
        else:
            for d_Emult in range(int(num_EW_pic)):
                coord = utils.get_location_metres(NW_coord, d_North*(2*d_Nmult+1), d_East*(2*(num_EW_pic-d_Emult)-1))
                cmds.append(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 1, delay, 0, 0, 0, coord.lat, coord.lon, coord.alt))
    return cmds

def distance_to_current_waypoint():
    """
    Gets distance in metres to the current waypoint.
    It returns None for the first waypoint (Home location).
    """
    nextwaypoint = vehicle.commands.next
    if nextwaypoint==0:
        return None
    missionitem=vehicle.commands[nextwaypoint-1] #commands are zero indexed
    lat = missionitem.x
    lon = missionitem.y
    alt = missionitem.z
    targetWaypointLocation = LocationGlobalRelative(lat,lon,alt)
    distancetopoint = utils.get_distance_metres(vehicle.location.global_frame, targetWaypointLocation)
    return distancetopoint

def getCurrentWaypoint():
    nextwaypoint = vehicle.commands.next
    if nextwaypoint==0:
        return None
    missionitem=vehicle.commands[nextwaypoint-1]
    lat = missionitem.x
    lon = missionitem.y
    alt = missionitem.z
    targetWaypointLocation = LocationGlobalRelative(lat,lon,alt)
    return targetWaypointLocation

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

NW_corner = LocationGlobalRelative(37.6571885, -121.8872434, 61)
NE_corner = LocationGlobalRelative(37.6571885, -121.8858862, 61)
SE_corner = LocationGlobalRelative(37.6549716, -121.8858862, 61)
SW_corner = LocationGlobalRelative(37.6549716, -121.8872434, 61)


cmds1 = scan_mission(NW_corner, NE_corner, SE_corner, SW_corner, 61)
utils.write_missionlist("mapping.txt", cmds1)

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
# wait_ready: If ``True`` wait until all default attributes have downloaded before the method returns (default is ``None``).
#             The default attributes to wait on are: :py:attr:`parameters`, :py:attr:`gps_0`, :py:attr:`armed`, :py:attr:`mode`, and :py:attr:`attitude`.
# timeout: timeout in seconds for wait_ready, aka time to wait for attributes to download from autopilot before throwing exception
# heartbeat_timeout: time to wait in seconds for heartbeat connection with autopilot

if verbose:
    vs.print_vehicle_state(vehicle)

cmds = vehicle.commands
cmds.download()
cmds.wait_ready()
if not vehicle.home_location:
    print("Waiting for home location ...")

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

#Wait until vehicle is armable
counter = 0
while not vehicle.is_armable:
    # If cannot acheive armable in 120 seconds, reboot the autopilot
    if counter == 120:
        vehicle.reboot()
    print("Waiting for vehicle to initialise...")
    counter += 1
    time.sleep(1)

while vehicle.mode != VehicleMode("AUTO"):
    #print("Currently in manual mode... Waiting for pilot to switch to AUTO mode")
    time.sleep(1)

print("Entered AUTO mode")

vehicle.gimbal.rotate(-90, 0, 0)
utils.setYaw(vehicle, 90)
while True:
    nextwaypoint=vehicle.commands.next

    if not nextwaypoint:
        break

    if distance_to_current_waypoint() < 1 and vehicle.groundspeed < 0.5:
        waypoint = getCurrentWaypoint()
        time.sleep(1)
        imcap.capture_image_and_save(cap, coordinates = (waypoint.lat, waypoint.lon))
        time.sleep(1)

    print('Distance to waypoint (%s): %s' % (nextwaypoint, distance_to_current_waypoint()))

    time.sleep(3)

utils.RTL(vehicle)

vehicle.gimbal.rotate(0, 0, 0)

while vehicle.armed:
    print("Returning to land. Will terminate once landed.")
    time.sleep(3)

if show_stream:
    rtmp.stop()

if vid_mapping:
    video.stop()

cap.release()
cv2.destroyAllWindows()

print("Close vehicle object")
vehicle.close()

if sitl is not None:
    sitl.stop()
