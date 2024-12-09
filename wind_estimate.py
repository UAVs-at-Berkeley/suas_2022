#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import math

import numpy as np

from pymavlink import mavutil
from dronekit import connect, VehicleMode, LocationGlobalRelative, LocationGlobal, Command

import argparse  
parser = argparse.ArgumentParser(description='Demonstrates basic mission operations.')
parser.add_argument('--connect', 
                   help="vehicle connection target string. If not specified, SITL automatically started and used.")
args = parser.parse_args()

connection_string = args.connect

print('Connecting to vehicle on: %s' % connection_string)
vehicle = connect(connection_string, wait_ready=True)

cmds = vehicle.commands
cmds.download()
cmds.wait_ready()
if not vehicle.home_location:
    print("Waiting for home location ...")

mission_ALT = 15
vehicle.airspeed = 7

@vehicle.on_message('WIND')
def listener(self, name, message):
    wind_speed = message.speed
    wind_dir = message.direction
    wind_speed_z = message.speed_z
    filewind = open("wind.txt", "a")
    if filewind != None:
        filewind.write("Wind speed: "+str(wind_speed)+"m/s \n Wind Direction: "+str(wind_dir)+ "deg")
    filewind.close()
    print("Wind speed: %s m/s \n Wind Direction: %s deg", wind_speed, wind_dir)

def arm_and_takeoff(aTargetAltitude):
    """
    Arms vehicle and fly to aTargetAltitude.
    """

    print("Basic pre-arm checks")
    # Don't try to arm until autopilot is ready
    # while not vehicle.is_armable:
    #     print(" Waiting for vehicle to initialise...")
    #     time.sleep(1)

    #Setting mode
    print("\nSet Vehicle.mode = GUIDED (currently: %s)" % vehicle.mode.name) 
    vehicle.mode = VehicleMode("GUIDED")
    # while not vehicle.mode.name=='GUIDED':  #Wait until mode has changed
    #     print(" Waiting for mode change ...")
    #     time.sleep(1)
    print("Arming motors")
    print("Arm Status", vehicle.armed)
    #vehicle.armed = True

    # Confirm vehicle armed before attempting to take off
    # while not vehicle.armed:
    #     print(" Waiting for arming...")
    #     print("Arm Status", vehicle.armed)
    #     time.sleep(1)
    print("Arm Status", vehicle.armed)
    print("Taking off!")
    #vehicle.simple_takeoff(aTargetAltitude)  # Take off to target altitude

    # Wait until the vehicle reaches a safe height before processing the goto
    #  (otherwise the command after Vehicle.simple_takeoff will execute
    #   immediately).
    # while True:
    #     print(" Altitude: ", vehicle.location.global_relative_frame.alt)
    #     # Break and return from function just below target altitude.
    #     if vehicle.location.global_relative_frame.alt >= aTargetAltitude * 0.95:
    #         print("Reached target altitude")
    #         break
    #     time.sleep(1)

#arm_and_takeoff(mission_ALT)

#print("Going towards first point for 30 seconds ...")
#point1 = LocationGlobalRelative(37.772900, -122.302000, 15)
#vehicle.simple_goto(point1)

# sleep so we can see the change in map
#time.sleep(30)

#print("Going towards second point for 30 seconds (groundspeed set to 10 m/s) ...")
#point2 = LocationGlobalRelative(37.772950, -122.301980, 15)
#vehicle.simple_goto(point2, groundspeed=3)

# sleep so we can see the change in map
#time.sleep(30)
while (1):
    True
# print('Return to launch')
# vehicle.mode = VehicleMode("RTL")
# while not vehicle.mode.name=='RTL':
#     time.sleep(1)
#     print("Waiting for mode change to RTL...")

#Close vehicle object before exiting script
while vehicle.armed:
    time.sleep(1)
    print("Waiting for vehicle to disarm...")
print("Close vehicle object")
vehicle.close()
