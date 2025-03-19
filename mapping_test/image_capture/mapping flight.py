#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
© Copyright 2015-2016, 3D Robotics.
mission_basic.py: Example demonstrating basic mission operations including creating, clearing and monitoring missions.

Full documentation is provided at https://dronekit-python.readthedocs.io/en/latest/examples/mission_basic.html
"""
# MIT License
# Copyright (c) 2019-2022 JetsonHacks

# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import cv2

import time
import math
import numpy as np
import image_capture

from pymavlink import mavutil
from dronekit import connect, VehicleMode, LocationGlobalRelative, LocationGlobal, Command


import argparse  
# parser = argparse.ArgumentParser(description='Demonstrates basic mission operations.')
# parser.add_argument('--connect', 
#                    help="vehicle connection target string. If not specified, SITL automatically started and used.")
# args = parser.parse_args()

# connection_string = args.connect

# mission_ALT = 61

# lastwaypoint = 8

# drop_count = 0

# print('Connecting to vehicle on: %s' % connection_string)
# vehicle = connect(connection_string, wait_ready=True)

# vehicle.airspeed = 7

# def get_distance_metres(aLocation1, aLocation2):
#     """
#     Returns the ground distance in metres between two LocationGlobal objects.

#     This method is an approximation, and will not be accurate over large distances and close to the 
#     earth's poles. It comes from the ArduPilot test code: 
#     https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
#     """
#     dlat = aLocation2.lat - aLocation1.lat
#     dlong = aLocation2.lon - aLocation1.lon
#     return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5

# def distance_to_current_waypoint():
#     """
#     Gets distance in metres to the current waypoint. 
#     It returns None for the first waypoint (Home location).
#     """
#     nextwaypoint = vehicle.commands.next
#     if nextwaypoint==0:
#         return None
#     missionitem=vehicle.commands[nextwaypoint-1] #commands are zero indexed
#     lat = missionitem.x
#     lon = missionitem.y
#     alt = missionitem.z
#     targetWaypointLocation = LocationGlobalRelative(lat,lon,alt)
#     distancetopoint = get_distance_metres(vehicle.location.global_frame, targetWaypointLocation)
#     return distancetopoint

# cmds = vehicle.commands
# cmds.download()
# cmds.wait_ready()
# if not vehicle.home_location:
#     print("Waiting for home location ...")

# print("Starting mission")
# # Reset mission set to first (0) waypoint
# vehicle.commands.next=0

# Set mode to AUTO to start mission
#vehicle.mode = VehicleMode("AUTO")

# Monitor mission. 
# Demonstrates getting and setting the command number 
# Uses distance_to_current_waypoint(), a convenience function for finding the 
#   distance to the next waypoint.

RSTP_URL = "rtsp://192.168.144.25:8554/main.264"
CAP = cv2.VideoCapture(RSTP_URL)
time.sleep(5)

# while cap.isOpened():
#     image_capture.capture_image_and_save(stream=cap)

image_capture.capture_image_and_save(exisitng_video_capture= RSTP_URL, coordinates= (0, 0))



# while True:
#     nextwaypoint=vehicle.commands.next
#     if nextwaypoint>1 and nextwaypoint < 7:

#         if distance_to_current_waypoint() < 1:
#             vehicle.gimbal.rotate(-90, 0, 0)
#             time.sleep(1)
#             image_capture.capture_image_and_save(stream=cap)
#             time.sleep(1)
#             vehicle.gimbal.rotate(0, 0, 0)
#             time.sleep(1)
                                  


#     print('Distance to waypoint (%s): %s' % (nextwaypoint, distance_to_current_waypoint()))

#     if nextwaypoint == lastwaypoint:
#         break
#     time.sleep(3)


# #Close vehicle object before exiting script
# while vehicle.armed:
#     time.sleep(1)
#     print("Waiting for vehicle to disarm...")
# print("Close vehicle object")
# vehicle.close()
