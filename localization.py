"""
Python Script for Localization of imagines

use YOLOV5 boxing,
find center of object box and then turn coordinate value into distance relative 
to angle of gimbal of camera, GPS coordinates of drone and altitude being plugged 
in as parameters.

method localization pass in parameters
camera_angle (due to the front of the drone): the field of the camera (default angle is pointing downward),
(up/down: alpha, left/right: beta, rotating clockwise/counterclockwise: gamma)
height: the height of the drone
latitude, longtitude: information from GPS
direction: direction of the drone (angle due north)

returns the target latitude, longtitude
"""

# lon: longtitude
# lat: latitude
# theta: the direction of drone due north
# h: height of the drone relative to ground (could be converted to altitude later on)
# x: x value of the image
# y: y value of the image
# delta: camera angle of field
# all angles in radius

import geopy
from geopy.distance import VincentyDistance
import numpy as np
def localization(lon, lat, theta, h, x, y, delta):
    dist_x = h * np.tan(delta * (x - 0.5))
    dist_y = h * np.tan(delta * (0.5 - y))
    dist_vector = np.array([[dist_x],[dist_y]])
    rotation_matrix = np.array([[np.cos(-theta), -np.sin(-theta)],[np.sin(-theta), np.cos(-theta)]])
    rotated_vector = np.dot(rotation_matrix, dist_vector)

    origin = geopy.Point(lat, lon)
    distance = (rotated_vector[0][0] ** 2 + rotated_vector[1][0] ** 2) ** 0.5 # in meters
    direction = 90 - np.arctan(rotated_vector[0][0]/rotated_vector[1][0])
    dest_lat, dest_lon = VincentyDistance(meters = distance).destination(origin, direction)
    return dest_lat, dest_lon