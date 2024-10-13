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

@vehicle.on_message('WIND_COV')
def listener(self, name, message):
    wind_x = message.wind_x
    wind_y = message.wind_y
    wind_z = message.wind_z
    wind_alt = message.wind_alt

