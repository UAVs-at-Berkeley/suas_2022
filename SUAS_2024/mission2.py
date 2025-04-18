#!/usr/bin/env python
# -*- coding: utf-8 -*-

# MIT License

# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import cv2

import time
import math

import geopy
import color_detection
import geopy.distance
import numpy as np

from ultralytics import YOLO
from pymavlink import mavutil
from dronekit import connect, VehicleMode, LocationGlobalRelative, LocationGlobal, Command

""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080 displayd in a 1/4 size window
"""

import argparse  
parser = argparse.ArgumentParser(description='Demonstrates basic mission operations.')
parser.add_argument('--connect', 
                   help="vehicle connection target string. If not specified, SITL automatically started and used.")
args = parser.parse_args()

connection_string = args.connect

mission_ALT = 25
airfield_MSL = 43.2816

last_searched = 12

lastwaypoint = 131

drop_count = 0

model = YOLO("best.pt")

given_targets = ["triangle", "pentagon", "cross","J", "E", "4"]

given_colors = ["green", "red", "blue", "purple", "yellow", "white"]

servo = {
  9: ("triangle", "J", "green", "purple"),
  10: ("pentagon", "E", "red", "yellow"),
  13: ("cross", "4", "blue", "white")
}

confidence = {}

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
    "semicircle",
    "quartercircle",
    "pentagon",

]
window_title = "CSI Camera"

print('Connecting to vehicle on: %s' % connection_string)
vehicle = connect(connection_string, wait_ready=True)

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
    
def localization(lon, lat, theta, h, x, y, h_delta, v_delta):
    dist_x = h * np.tan(h_delta * (x - 0.5))
    dist_y = h * np.tan(v_delta * (0.5 - y))
    dist_vector = np.array([[dist_x],[dist_y]])
    rotation_matrix = np.array([[np.cos(-theta), -np.sin(-theta)],[np.sin(-theta), np.cos(-theta)]])
    rotated_vector = np.dot(rotation_matrix, dist_vector)

    origin = geopy.Point(lat, lon)
    distance = (rotated_vector[0][0] ** 2 + rotated_vector[1][0] ** 2) ** 0.5 # in meters
    direction = 90 - np.arctan(rotated_vector[0][0]/rotated_vector[1][0])
    d = geopy.distance.distance(meters = distance)
    final = d.destination(origin, direction)
    return final.latitude, final.longitude

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
    
    newlat = 38.315339
    newlon = -76.548108
     
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

print('Create a new mission (for current location)')

#adds_wypt_mission()

cmds = vehicle.commands
cmds.download()
cmds.wait_ready()
if not vehicle.home_location:
    print("Waiting for home location ...")

print("Starting mission")
# Reset mission set to first (0) waypoint
vehicle.commands.next=0

# Set mode to AUTO to start mission
#vehicle.mode = VehicleMode("AUTO")

# Monitor mission. 
# Demonstrates getting and setting the command number 
# Uses distance_to_current_waypoint(), a convenience function for finding the 
#   distance to the next waypoint.
while True:
    nextwaypoint=vehicle.commands.next
    if drop_count > 0 and nextwaypoint == 12:
        vehicle.commands.next = last_searched
    if nextwaypoint>11:
        vehicle.airspeed = 3
        camSet = 'nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=3840,height=2160,framerate=29/1,format=NV12 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,width=1920,height=1080,format=BGR ! queue ! appsink'
        video_capture = cv2.VideoCapture(camSet, cv2.CAP_GSTREAMER)
        if video_capture.isOpened():
            try:
                
                while True:
                    ret_val, frame = video_capture.read()
                    results = model(frame, stream=True)
                    inframe = []
                    colors_found = []
                    x_loc = 0
                    y_loc = 0
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
                            
                            if len(inframe) == 0:
                                x_loc = x2+x1 // 2
                                y_loc = y2+y1 // 2                            

                            # put box in cam
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                            # class name
                            roi = frame[y1:y2, x1:x2]
                            roi = cv2.resize(roi, (300, 300))
                            
                            colors = color_detection.color_det(roi)

                            if colors[0] in given_colors:
                                colors_found.append(colors[0])
                                main_color = colors[0]
                            if colors[1] in given_colors:
                                colors_found.append(colors[1])
                                sec_color = colors[1]
                            
                            #class name
                            cls = int(box.cls[0])
                            if model.names[cls] in given_targets:
                                inframe.append(model.names[cls])
                            print("Class name -->", model.names[cls])
                            cv2.imwrite("Image_"+ model.names[cls] + ".jpg", roi)
                            
                            print(inframe)
                            print(colors_found)

                            # object details
                            org = [x1, y1]
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            fontScale = 1
                            color = (255, 0, 0)
                            thickness = 2

                            cv2.putText(
                                frame, model.names[cls], org, font, fontScale, color, thickness
                            )
                    # Check to see if the user closed the window
                    # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                    # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                    #cv2.imshow(window_title, frame)

                    for tar in inframe:
                        print(inframe[0])
                        print(confidence)
                        if (tar in given_targets):
                            if not tar in list(confidence.keys()):
                                vehicle.mode = VehicleMode("LOITER")
                                while not vehicle.mode.name=='LOITER':
                                    time.sleep(1)
                                    print("waiting for mode change...")
                                print(vehicle.mode)
                                confidence[tar] = 0
                            confidence[tar] += 1
                            if confidence[tar] == 3:
                                last_searched = nextwaypoint
                                confidence[tar] += 1
                                vehicle.mode = VehicleMode("GUIDED")
                                while not vehicle.mode.name=='GUIDED':
                                    time.sleep(1)
                                    print("waiting for mode change...")
                                print(vehicle.mode)
                                object_point = localization(vehicle.location.global_relative_frame.lon, vehicle.location.global_relative_frame.lat, vehicle.heading, vehicle.location.global_relative_frame.alt, x_loc, y_loc, 0.235969, 0.1308997)
                                point1 = LocationGlobalRelative(object_point[0], object_point[1], vehicle.location.global_relative_frame.alt)
                                print(point1)
                                print(vehicle.location.global_relative_frame)
                                if (get_distance_metres(vehicle.location.global_frame, point1) < 10):
                                    vehicle.simple_goto(point1)
                                    x=0
                                    while(get_distance_metres(vehicle.location.global_frame, point1) > 3):
                                        time.sleep(1)
                                        print("Approaching Target")
                                        
                                    msg = vehicle.message_factory.command_long_encode(0, 0, mavutil.mavlink.MAV_CMD_CONDITION_DELAY, 0, 10, 0, 0, 0, 0, 0, 0) #Pause command
                                    vehicle.send_mavlink(msg)
                                    time.sleep(10)
                                    for val in servo.values():
                                        if tar in val:
                                            drop_servo = list(servo.keys())[list(servo.values()).index(val)]
                                    msg = vehicle.message_factory.command_long_encode(0, 0, mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0, drop_servo, 2600, 0, 0, 0, 0, 0)
                                    vehicle.send_mavlink(msg)
                                    msg = vehicle.message_factory.command_long_encode(0, 0, mavutil.mavlink.MAV_CMD_CONDITION_DELAY, 0, 2, 0, 0, 0, 0, 0, 0) #Pause command for 2 seconds
                                    vehicle.send_mavlink(msg)
                                    time.sleep(2)
                                    drop_count += 1
                                    vehicle.commands.next = 0
                                    vehicle.mode = VehicleMode("AUTO")
                                    while not vehicle.mode.name=='AUTO':
                                        time.sleep(1)
                                        print("Waiting for mode change back to AUTO...")
                                    break
                                else:
                                    print("Bad Location")
                                    msg = vehicle.message_factory.command_long_encode(0, 0, mavutil.mavlink.MAV_CMD_CONDITION_DELAY, 0, 10, 0, 0, 0, 0, 0, 0) #Pause command
                                    vehicle.send_mavlink(msg)
                                    time.sleep(10)
                                    for val in servo.values():
                                        if tar in val:
                                            drop_servo = list(servo.keys())[list(servo.values()).index(val)]
                                    msg = vehicle.message_factory.command_long_encode(0, 0, mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0, drop_servo, 2600, 0, 0, 0, 0, 0)
                                    vehicle.send_mavlink(msg)
                                    msg = vehicle.message_factory.command_long_encode(0, 0, mavutil.mavlink.MAV_CMD_CONDITION_DELAY, 0, 2, 0, 0, 0, 0, 0, 0) #Pause command for 2 seconds
                                    vehicle.send_mavlink(msg)
                                    time.sleep(2)
                                    drop_count += 1
                                    vehicle.commands.next = 0
                                    vehicle.mode = VehicleMode("AUTO")
                                    while not vehicle.mode.name=='AUTO':
                                        time.sleep(1)
                                        print("Waiting for mode change back to AUTO...")
                                    break
                        
                    
                    
                    #keyCode = cv2.waitKey(30) & 0xFF
                    # Stop the program on the ESC key or 'q'
                    #if keyCode == 27 or keyCode == ord('q'):
                    #    break
                    
                    time.sleep(1)
            finally:
                video_capture.release()
                #cv2.destroyAllWindows()
        else:
            print("Error: Unable to open camera")
            
    print('Distance to waypoint (%s): %s' % (nextwaypoint, distance_to_current_waypoint()))
  
    if nextwaypoint == lastwaypoint:
        break
    time.sleep(10)
    
print('Return to launch')
vehicle.mode = VehicleMode("RTL")
video_capture.release()


#Close vehicle object before exiting script
print("Close vehicle object")
vehicle.close()
