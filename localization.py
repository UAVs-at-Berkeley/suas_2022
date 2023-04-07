"""
Python Script for Localization of imagines

use YOLOV5 boxing,
find center of object box and then turn coordinate value into distance relative 
to angle of gimbal of camera, GPS coordinates of drone and altitude being plugged 
in as parameters.

method localization pass in parameters
camera_angle (due to the front of the drone): the field of the camera (default angle is pointing downward),
(up/down, left/right, rotating clockwise/counterclockwise)
height: the height of the drone
latitude, longtitude: information from GPS
direction: direction of the drone (angle due north)

returns the target latitude, longtitude
"""
def localization(camera_angle, height, latitude, longtitude, direction):

    return []