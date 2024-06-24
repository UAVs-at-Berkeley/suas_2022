from __future__ import print_function

from dronekit import connect, VehicleMode, LocationGlobalRelative, LocationGlobal, Command
import time
import math
from pymavlink import mavutil
import cv2
from ultralytics import YOLO

import argparse  
parser = argparse.ArgumentParser(description='Demonstrates basic mission operations.')
parser.add_argument('--connect', 
                   help="vehicle connection target string. If not specified, SITL automatically started and used.")
args = parser.parse_args()

connection_string = args.connect

mission_ALT = 25
airfield_MSL = 43.2816

last_searched = 11

lastwaypoint = 45

drop_count = 0

model = YOLO("yolov8n.pt")

classNames = [
    "emergent",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "circle",
    "triangle",
    "rectangle",
    "star",
    "cross",
    "half-circle",
    "quarter-circle",
    "pentagon",

]

# Connect to the Vehicle
print('Connecting to vehicle on: %s' % connection_string)
vehicle = connect(connection_string, wait_ready=True)

vehicle.airspeed = 5

def get_location_metres(original_location, dNorth, dEast):
    """
    Returns a LocationGlobal object containing the latitude/longitude `dNorth` and `dEast` metres from the 
    specified `original_location`. The returned Location has the same `alt` value
    as `original_location`.

    The function is useful when you want to move the vehicle around specifying locations relative to 
    the current vehicle position.
    The algorithm is relatively accurate over small distances (10m within 1km) except close to the poles.
    For more information see:
    http://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters
    """
    earth_radius=6378137.0 #Radius of "spherical" earth
    #Coordinate offsets in radians
    dLat = dNorth/earth_radius
    dLon = dEast/(earth_radius*math.cos(math.pi*original_location.lat/180))

    #New position in decimal degrees
    newlat = original_location.lat + (dLat * 180/math.pi)
    newlon = original_location.lon + (dLon * 180/math.pi)
    return LocationGlobal(newlat, newlon, original_location.alt)


def get_distance_metres(aLocation1, aLocation2):
    """
    Returns the ground distance in metres between two LocationGlobal objects.

    This method is an approximation, and will not be accurate over large distances and close to the 
    earth's poles. It comes from the ArduPilot test code: 
    https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
    """
    dlat = aLocation2.lat - aLocation1.lat
    dlong = aLocation2.lon - aLocation1.lon
    return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5



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
    distancetopoint = get_distance_metres(vehicle.location.global_frame, targetWaypointLocation)
    return distancetopoint

def download_mission():
    """
    Download the current mission from the vehicle.
    """
    cmds = vehicle.commands
    cmds.download()
    cmds.wait_ready() # wait until download is complete.

def adds_wypt_mission():
    """
    Adds a takeoff command and four waypoint commands to the current mission. 
    The waypoints are positioned to form a square of side length 2*aSize around the specified LocationGlobal (aLocation).

    The function assumes vehicle.commands matches the vehicle mission state 
    (you must have called download at least once in the session and after clearing the mission)
    """	

    curr = vehicle.location.global_frame

    print("\nCurrent Location: %s" % curr)

    cmds = vehicle.commands
    cmds.download()
    cmds.wait_ready()
    if not vehicle.home_location:
        print(" Waiting for home location ...")

    # We have a home location.
    print("\n Home location: %s" % vehicle.home_location)

    print(" Clear any existing commands")
    cmds.clear() 

    vehicle.home_location = LocationGlobal(38.315339, -76.548108, airfield_MSL)
    
    print(" Define/add new commands.")
    # Add new commands. The meaning/order of the parameters is documented in the Command class. 
     
    #Add MAV_CMD_NAV_TAKEOFF command. This is ignored if the vehicle is already in the air.
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 0, mission_ALT))

    #Define the ten MAV_CMD_NAV_WAYPOINT locations and add the commands
    point1 = LocationGlobalRelative(newlat, newlon, mission_ALT)
    point2 = LocationGlobalRelative(newlat, newlon, mission_ALT)
    point3 = LocationGlobalRelative(newlat, newlon, mission_ALT)
    point4 = LocationGlobalRelative(newlat, newlon, mission_ALT)
    point5 = LocationGlobalRelative(newlat, newlon, mission_ALT)
    point6 = LocationGlobalRelative(newlat, newlon, mission_ALT)
    point7 = LocationGlobalRelative(newlat, newlon, mission_ALT)
    point8 = LocationGlobalRelative(newlat, newlon, mission_ALT)
    point9 = LocationGlobalRelative(newlat, newlon, mission_ALT)
    point10 = LocationGlobalRelative(newlat, newlon, mission_ALT)
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 15, 5, 0, 0, point1.lat, point1.lon, mission_ALT))
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 15, 5, 0, 0, point2.lat, point2.lon, mission_ALT))
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 15, 5, 0, 0, point3.lat, point3.lon, mission_ALT))
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 15, 5, 0, 0, point4.lat, point4.lon, mission_ALT))
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 15, 5, 0, 0, point5.lat, point5.lon, mission_ALT))
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 15, 5, 0, 0, point6.lat, point6.lon, mission_ALT))
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 15, 5, 0, 0, point7.lat, point7.lon, mission_ALT))
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 15, 5, 0, 0, point8.lat, point8.lon, mission_ALT))
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 15, 5, 0, 0, point9.lat, point9.lon, mission_ALT))
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 15, 5, 0, 0, point10.lat, point10.lon, mission_ALT))
    #add dummy waypoint "11" at point 10 (lets us know when have reached destination)
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 15, 5, 0, 0, point10.lat, point10.lon, mission_ALT))    

    print(" Upload new commands to vehicle")
    cmds.upload()

def arm_and_takeoff(aTargetAltitude):
    """
    Arms vehicle and fly to aTargetAltitude.
    """

    print("Basic pre-arm checks")
    # Don't let the user try to arm until autopilot is ready
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

        
    print("Arming motors")
    # Copter should arm in GUIDED mode
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:      
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude) # Take off to target altitude

    # Wait until the vehicle reaches a safe height before processing the goto (otherwise the command 
    #  after Vehicle.simple_takeoff will execute immediately).
    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)      
        if vehicle.location.global_relative_frame.alt>=aTargetAltitude*0.95: #Trigger just below target alt.
            print("Reached target altitude")
            break
        time.sleep(1)

print('Create a new mission (for current location)')

adds_wypt_mission()

print("Starting mission")
# Reset mission set to first (0) waypoint
vehicle.commands.next=0

# Set mode to AUTO to start mission
vehicle.mode = VehicleMode("AUTO")


# Monitor mission. 
# Demonstrates getting and setting the command number 
# Uses distance_to_current_waypoint(), a convenience function for finding the 
#   distance to the next waypoint.

while True:
    nextwaypoint=vehicle.commands.next
    if nextwaypoint>10:
        camSet = 'nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=3840,height=2160,framerate=29/1,format=NV12 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,width=1920,height=1080,format=BGR ! queue ! appsink'
        video_capture = cv2.VideoCapture(camSet, cv2.CAP_GSTREAMER)
        if video_capture.isOpened():
            ret_val, frame = video_capture.read()
            results = model(frame, stream=True)
            for r in results:
                boxes = r.boxes

                for box in boxes:
                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    )  # convert to int values

                    # put box in cam
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    roi = frame[y1:y2, x1:x2]

                    roi = cv2.resize(roi, (300, 300))

                    # class name
                    cls = int(box.cls[0])
                    print("Class name -->", model.names[cls])
                    cv2.imwrite(roi, "Image_"+ str(model.names[cls]) + ".jpg")

                    # object details
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2

                    cv2.putText(
                        frame, classNames[cls], org, font, fontScale, color, thickness
                    )
            # Check to see if the user closed the window
            # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
            # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
            #cv2.imshow(window_title, frame)
            if model.names[cls]=="person":
                    last_searched = nextwaypoint
                    vehicle.mode = VehicleMode("GUIDED")
                    object_point = LocationGlobalRelative(, , mission_ALT)
                    vehicle.simple_goto(object_point)
                    x=0
                    while(get_distance_metres(vehicle.location.global_frame, object_point) > 1):
                        x += 1
                    msg = vehicle.message_factory.command_long_encode(0, 0, mavutil.mavlink.MAV_CMD_CONDITION_DELAY, 0, 10, 0, 0, 0, 0, 0, 0) #Pause command
                    vehicle.send_mavlink(msg)
                    msg = vehicle.message_factory.command_long_encode(0, 0, mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0, 9, 2600, 0, 0, 0, 0, 0)
                    vehicle.send_mavlink(msg)
                    msg = vehicle.message_factory.command_long_encode(0, 0, mavutil.mavlink.MAV_CMD_CONDITION_DELAY, 0, 2, 0, 0, 0, 0, 0, 0) #Pause command for 2 seconds
                    vehicle.send_mavlink(msg)
                    drop_count += 1
                    vehicle.commands.next = 0
                    vehicle.mode = VehicleMode("AUTO")


        else:
            print("Error: Unable to open camera")

    print('Distance to waypoint (%s): %s' % (nextwaypoint, distance_to_current_waypoint()))
  
    if drop_count > 0 and nextwaypoint == 10:
        vehicle.commands.next = last_searched
    if nextwaypoint == lastwaypoint:
        break
    time.sleep(1)

print('Return to launch')
vehicle.mode = VehicleMode("RTL")
video_capture.release()


#Close vehicle object before exiting script
print("Close vehicle object")
vehicle.close()