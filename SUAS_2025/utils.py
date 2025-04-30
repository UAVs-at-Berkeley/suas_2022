from dronekit import connect, VehicleMode, LocationGlobalRelative, mavutil, Command
import math
import time

# For setting custom mavlink command can either encode with vehicle.message_factory.command_long_encode() and send with vehicle.send_mavlink()
# or can encode and send at once with vehicle.command_long_send()

def getHomeLocation(vehicle):
    # Get Vehicle Home location - will be `None` until first set by autopilot
    while not vehicle.home_location:
        cmds = vehicle.commands
        cmds.download()
        cmds.wait_ready()
        if not vehicle.home_location:
            print(" Waiting for home location ...")
    # We have a home location, so print it!        
    print("\n Home location: %s" % vehicle.home_location)
    return vehicle.home_location

def downloadCommands(vehicle):
    cmds = vehicle.commands
    cmds.download()
    cmds.wait_ready()
    return cmds

def goto_position_target_global_int(vehicle, aLocation):
    """
    Send SET_POSITION_TARGET_GLOBAL_INT command to request the vehicle fly to a specified LocationGlobal.

    For more information see: https://pixhawk.ethz.ch/mavlink/#SET_POSITION_TARGET_GLOBAL_INT

    See the above link for information on the type_mask (0=enable, 1=ignore). 
    At time of writing, acceleration and yaw bits are ignored.
    """
    msg = vehicle.message_factory.set_position_target_global_int_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, # frame
        0b0000111111111000, # type_mask (only speeds enabled)
        aLocation.lat*1e7, # lat_int - X Position in WGS84 frame in 1e7 * meters
        aLocation.lon*1e7, # lon_int - Y Position in WGS84 frame in 1e7 * meters
        aLocation.alt, # alt - Altitude in meters in AMSL altitude, not WGS84 if absolute or relative, above terrain if GLOBAL_TERRAIN_ALT_INT
        0, # X velocity in NED frame in m/s
        0, # Y velocity in NED frame in m/s
        0, # Z velocity in NED frame in m/s
        0, 0, 0, # afx, afy, afz acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink) 
    # send command to vehicle
    vehicle.send_mavlink(msg)

def goto(vehicle, dNorth, dEast, gotoFunction=None):
    """
    Moves the vehicle to a position dNorth metres North and dEast metres East of the current position.

    The method takes a function pointer argument with a single `dronekit.lib.LocationGlobal` parameter for 
    the target position. This allows it to be called with different position-setting commands. 
    By default it uses the standard method: dronekit.lib.Vehicle.simple_goto().

    The method reports the distance to target every two seconds.
    """
    if not gotoFunction:
        gotoFunction = vehicle.simple_goto
    currentLocation = vehicle.location.global_relative_frame
    targetLocation = get_location_metres(currentLocation, dNorth, dEast)
    targetDistance = get_distance_metres(currentLocation, targetLocation)
    gotoFunction(targetLocation)
    
    #print "DEBUG: targetLocation: %s" % targetLocation
    #print "DEBUG: targetLocation: %s" % targetDistance

    while vehicle.mode.name=="GUIDED": #Stop action if we are no longer in guided mode.
        #print "DEBUG: mode: %s" % vehicle.mode.name
        remainingDistance=get_distance_metres(vehicle.location.global_relative_frame, targetLocation)
        print("Distance to target: ", remainingDistance)
        if remainingDistance<=targetDistance*0.01: #Just below target, in case of undershoot.
            print("Reached target")
            break
        time.sleep(2)

def setVehicleMode(vehicle, mode='STABILIZE'):
    print("\nSet Vehicle.mode = %s (currently: %s)",mode, vehicle.mode.name) 
    vehicle.mode = VehicleMode(mode)
    while not vehicle.mode.name==mode:  #Wait until mode has changed
        print(" Waiting for mode change ...")
        time.sleep(1)
    print("Vehicle mode is now %s" % vehicle.mode.name)

def armVehicle(vehicle):
    print("\nSet Vehicle.armed=True (currently: %s)" % vehicle.armed)
    vehicle.armed = True
    while not vehicle.armed:
        print("Waiting for arming...")
        time.sleep(1)
    print("Vehicle is armed: %s" % vehicle.armed)

def setAirspeed(vehicle, speed):
    vehicle.airspeed = speed
    print("Vehicle airspeed is now: %f m/s" % speed)

def setGroundspeed(vehicle, speed):
    vehicle.groundspeed = speed
    print("Vehicle ground speed is now: %f m/s" % speed)

def takeoff(vehicle, altitude):
    print("Taking off")
    if alt is not None:
        altitude = float(alt)
        if math.isnan(altitude) or math.isinf(altitude):
            raise ValueError("Altitude was NaN or Infinity. Please provide a real number")
        vehicle.command_long_send(0, 0, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                                               0, 0, 0, 0, 0, 0, 0, altitude)


def RTL(vehicle):
    print("Returning to Launch")
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH, #command
        0, #confirmation
        0, 0, 0, 0, #params 1-4
        0, #param 5
        0, #param 6
        0 #param 7
        )
    # send command to vehicle
    vehicle.send_mavlink(msg)
    while not vehicle.mode.name=="RTL":  #Wait until mode has changed
        print("Waiting for mode change ...")
        time.sleep(1)

def readMission(vehicle, aFileName):
    """
    Load a mission from a file into a list. The mission definition is in the Waypoint file
    format (http://qgroundcontrol.org/mavlink/waypoint_protocol#waypoint_file_format).

    This function is used by upload_mission().
    """
    print("\nReading mission from file: %s" % aFileName)
    cmds = vehicle.commands
    missionlist=[]
    with open(aFileName) as f:
        for i, line in enumerate(f):
            if i==0:
                if not line.startswith('QGC WPL 110'):
                    raise Exception('File is not supported WP version')
            else:
                linearray=line.split('\t')
                ln_index=int(linearray[0])
                ln_currentwp=int(linearray[1])
                ln_frame=int(linearray[2])
                ln_command=int(linearray[3])
                ln_param1=float(linearray[4])
                ln_param2=float(linearray[5])
                ln_param3=float(linearray[6])
                ln_param4=float(linearray[7])
                ln_param5=float(linearray[8])
                ln_param6=float(linearray[9])
                ln_param7=float(linearray[10])
                ln_autocontinue=int(linearray[11].strip())
                cmd = Command( 0, 0, 0, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4, ln_param5, ln_param6, ln_param7)
                missionlist.append(cmd)
    return missionlist


def upload_mission(vehicle, aFileName):
    """
    Upload a mission from a file. 
    """
    #Read mission from file
    missionlist = readMission(vehicle, aFileName)
    
    print("\nUpload mission from a file: %s" % aFileName)
    #Clear existing mission from vehicle
    print(' Clear mission')
    cmds = vehicle.commands
    cmds.clear()
    #Add new mission to vehicle
    for command in missionlist:
        cmds.add(command)
    print(' Upload mission')
    vehicle.commands.upload()


def download_mission():
    """
    Downloads the current mission and returns it in a list.
    It is used in save_mission() to get the file information to save.
    """
    print(" Download mission from vehicle")
    missionlist=[]
    cmds = vehicle.commands
    cmds.download()
    cmds.wait_ready()
    for cmd in cmds:
        missionlist.append(cmd)
    return missionlist

def save_mission(vehicle, aFileName):
    """
    Save a mission in the Waypoint file format 
    (http://qgroundcontrol.org/mavlink/waypoint_protocol#waypoint_file_format).
    """
    print("\nSave mission from Vehicle to file: %s" % aFileName)    
    #Download mission from vehicle
    missionlist = download_mission()
    #Add file-format information
    output='QGC WPL 110\n'
    #Add home location as 0th waypoint
    home = vehicle.home_location
    output+="%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (0,1,0,16,0,0,0,0,home.lat,home.lon,home.alt,1)
    #Add commands
    for cmd in missionlist:
        commandline="%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (cmd.seq,cmd.current,cmd.frame,cmd.command,cmd.param1,cmd.param2,cmd.param3,cmd.param4,cmd.x,cmd.y,cmd.z,cmd.autocontinue)
        output+=commandline
    with open(aFileName, 'w') as file_:
        print(" Write mission to file")
        file_.write(output)

def write_missionlist(aFileName, cmds):
    """
    Save a mission in the Waypoint file format 
    (http://qgroundcontrol.org/mavlink/waypoint_protocol#waypoint_file_format).
    """
    print("\nSave mission from list to file: %s" % aFileName)    
    #Download mission from vehicle
    missionlist = cmds
    #Add file-format information
    output='QGC WPL 110\n'

    #Add commands
    for cmd in missionlist:
        commandline="%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (cmd.seq,cmd.current,cmd.frame,cmd.command,cmd.param1,cmd.param2,cmd.param3,cmd.param4,cmd.x,cmd.y,cmd.z,cmd.autocontinue)
        output+=commandline
    with open(aFileName, 'w') as file_:
        print(" Write mission to file")
        file_.write(output)

# Given a list of GPS coordinates and an altitude and hold time, generates a mission command list of, optionally write to file
def createMissionNavPts(coords, alt, hold_time=0, aFileName=None):
    cmds = []
    for coord in coords:
        cmds.append(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 1, hold_time, 0, 0, 0, coord.lat, coord.lon, alt))
    
    if aFileName is None:
        return cmds
    else:
        write_missionlist(aFileName, cmds)
        return cmds

# Given a list of GPS coordinates with altitude and hold time, generates a mission command list of, optionally write to file
def varAlt_createMissionNavPts(coords, hold_time=0, aFileName=None):
    cmds = []
    for coord in coords:
        cmds.append(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 1, hold_time, 0, 0, 0, coord.lat, coord.lon, coord.alt))
    
    if aFileName is None:
        return cmds
    else:
        write_missionlist(aFileName, cmds)
        return cmds
        
        
def printfile(aFileName):
    """
    Print a mission file to demonstrate "round trip"
    """
    print("\nMission file: %s" % aFileName)
    with open(aFileName) as f:
        for line in f:
            print(' %s' % line.strip())

def get_yaw_degrees(aLocation1, aLocation2):
    """Alternate code:
    off_x = aLocation2.lon - aLocation1.lon
    off_y = aLocation2.lat - aLocation1.lat
    bearing = 90.00 + math.atan2(-off_y, off_x) * 57.2957795
    if bearing < 0:
        bearing += 360.00
    return bearing;
    """
    dlon = math.pi*(aLocation2.lon - aLocation1.lon)/180
    y = math.sin(dlon) * math.cos(math.radians(aLocation2.lat))
    x = math.cos(math.radians(aLocation1.lat)) * math.sin(math.radians(aLocation2.lat)) - math.sin(math.radians(aLocation1.lat)) * math.cos(math.radians(aLocation2.lat)) * math.cos(dlon)
    bearing = math.atan2(y, x)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360
    # round down to 2 decimal places
    bearing = round(bearing, 3)
    return bearing

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
    return LocationGlobalRelative(newlat, newlon, original_location.alt)

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

def distance_to_current_waypoint(vehicle):
    """
    Gets distance in metres to the current waypoint. 
    It returns None for the first waypoint (Home location).
    """
    nextwaypoint = vehicle.commands.next
    if nextwaypoint==0:
        return None

    missionitem=vehicle.commands[nextwaypoint] #commands are zero indexed
    lat = missionitem.x
    lon = missionitem.y
    alt = missionitem.z
    targetWaypointLocation = LocationGlobalRelative(lat,lon,alt)
    distancetopoint = get_distance_metres(vehicle.location.global_frame, targetWaypointLocation)
    return distancetopoint

def getCurrentWaypoint(vehicle):
    nextwaypoint = vehicle.commands.next
    if nextwaypoint==0:
        return None
    missionitem=vehicle.commands[nextwaypoint-1]
    lat = missionitem.x
    lon = missionitem.y
    alt = missionitem.z
    targetWaypointLocation = LocationGlobalRelative(lat,lon,alt)
    return targetWaypointLocation

def clearCmds(vehicle):
    print(" Clear any existing commands")
    vehicle.commands.clear()

# Here is an Example of the waypoint command format in text file (CURRENT WP==1 for first waypoint because that sets the current waypoint. Should only be true for one waypoint at a time)
# More details here: https://mavlink.io/en/file_formats/
# <INDEX> <CURRENT WP> <COORD FRAME> <COMMAND> <PARAM1> <PARAM2> <PARAM3> <PARAM4> <PARAM5/X/LATITUDE> <PARAM6/Y/LONGITUDE> <PARAM7/Z/ALTITUDE> <AUTOCONTINUE>
#    0	       1	        0	         16	  0.14999999	0	     0	      0	        8.54800000	      47.375999999999	       550	              1
#

# Here is an example and the command format for the mavlink message below described here https://mavlink.io/en/messages/common.html#MISSION_ITEM_INT
#     <SYSTEM ID> <COMPONENT ID> <WAYPOINT ID/INDEX> <COORD FRAME> <COMMAND> <CURRENT WP> <AUTOCONTINUE> <PARAM1> <PARAM2> <PARAM3> <PARAM4> <PARAM5/X/LATITUDE> <PARAM6/Y/LONGITUDE> <PARAM7/Z/ALTITUDE>
# Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0(change to 1 only if want to be current waypoint), 1, hold_time, 0, 0, 0, lat, lon, alt)

def addWaypoint(cmds, lat, lon, alt, hold_time=0):
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 1, hold_time, 0, 0, 0, lat, lon, alt))

def addCustomMissionCommand(cmds, command, param1, param2, param3, param4, param5, param6, param7):
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, command, 0, 1, param1, param2, param3, param4, param5, param6, param7))

def addWaitWaypoint(cmds, lat, lon, alt, loiter_time):
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_LOITER_TIME, 0, 1, loiter_time, 0, 0, 0, lat, lon, alt))

def addSetYawCommand(cmds, yaw_angle, yaw_speed=0, direction=0, relative=0):
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0, 1, yaw_angle, yaw_speed, direction, relative, 0, 0, 0))


# Here is an example and the command format for the mavlink message that is immediate, not being added to a mission list, we are using the set servo described here: https://mavlink.io/en/messages/common.html#MAV_CMD_DO_SET_SERVO
#               <SYSTEM ID> <COMPONENT ID> <COMMAND> <CONFIRMATION> <PARAM1> <PARAM2> <PARAM3> <PARAM4> <PARAM5> <PARAM6> <PARAM7>
# command_long_encode(0, 0, mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0, num, state, 0, 0, 0, 0, 0)
def setServo(vehicle, num, state):
    msg = vehicle.message_factory.command_long_encode(0, 0, mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0, num, state, 0, 0, 0, 0, 0)
    vehicle.send_mavlink(msg)

def setYaw(vehicle, heading, relative=False):
    #Send MAV_CMD_CONDITION_YAW message to point vehicle at a specified heading (in degrees).
    if relative:
        is_relative = 1 #yaw relative to direction of travel
    else:
        is_relative = 0 #yaw is an absolute angle
    # create the CONDITION_YAW command using command_long_encode()
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_CONDITION_YAW, #command
        0, #confirmation
        heading,    # param 1, yaw in degrees
        0,          # param 2, yaw speed deg/s
        1,          # param 3, direction -1 ccw, 1 cw
        is_relative, # param 4, relative offset 1, absolute angle 0
        0, 0, 0)    # param 5 ~ 7 not used
    # send command to vehicle
    vehicle.send_mavlink(msg)

def simple_goto(vehicle, location, airspeed=None, groundspeed=None):
        '''
        Go to a specified global location (:py:class:`LocationGlobal` or :py:class:`LocationGlobalRelative`).

        There is no mechanism for notification when the target location is reached, and if another command arrives
        before that point that will be executed immediately.

        You can optionally set the desired airspeed or groundspeed (this is identical to setting
        :py:attr:`airspeed` or :py:attr:`groundspeed`). The vehicle will determine what speed to
        use if the values are not set or if they are both set.

        The method will change the :py:class:`VehicleMode` to ``GUIDED`` if necessary.

        .. code:: python

            # Set mode to guided - this is optional as the simple_goto method will change the mode if needed.
            vehicle.mode = VehicleMode("GUIDED")

            # Set the LocationGlobal to head towards
            a_location = LocationGlobal(-34.364114, 149.166022, 30)
            vehicle.simple_goto(a_location)

        :param location: The target location (:py:class:`LocationGlobal` or :py:class:`LocationGlobalRelative`).
        :param airspeed: Target airspeed in m/s (optional).
        :param groundspeed: Target groundspeed in m/s (optional).
        '''
        if isinstance(location, LocationGlobalRelative):
            frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
            alt = location.alt
        elif isinstance(location, LocationGlobal):
            # This should be the proper code:
            # frame = mavutil.mavlink.MAV_FRAME_GLOBAL
            # However, APM discards information about the relative frame
            # and treats any alt value as relative. So we compensate here.
            frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
            if not self.home_location:
                self.commands.download()
                self.commands.wait_ready()
            alt = location.alt - self.home_location.alt
        else:
            raise ValueError('Expecting location to be LocationGlobal or LocationGlobalRelative.')

        self._master.mav.mission_item_send(0, 0, 0, frame,
                                           mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 2, 0, 0,
                                           0, 0, 0, location.lat, location.lon,
                                           alt)

        if airspeed is not None:
            self.airspeed = airspeed
        if groundspeed is not None:
            self.groundspeed = groundspeed


# Here is the documentation on sending a LONG command: https://mavlink.io/en/messages/common.html#COMMAND_LONG
def sendCustomCommandLONG(vehicle, command, param1, param2, param3, param4, param5, param6, param7):
    msg = vehicle.message_factory.command_long_encode(0, 0, command, 0, param1, param2, param3, param4, param5, param6, param7)
    vehicle.send_mavlink(msg)

# Here is the documentation on sending a INT command: https://mavlink.io/en/messages/common.html#COMMAND_INT
# <SYSTEM ID> <COMPONENT ID> <COORD FRAME> <COMMAND> <CURRENT WP (NOT USED)> <AUTOCONTINUE (NOT USED)> <PARAM1> <PARAM2> <PARAM3> <PARAM4> <PARAM5> <PARAM6> <PARAM7>
def sendCustomCommandINT(vehicle, command, param1, param2, param3, param4, param5, param6, param7):
    msg = vehicle.message_factory.command_int_encode(0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, command, 0, 0, param1, param2, param3, param4, param5, param6, param7)
    vehicle.send_mavlink(msg)

# You can create message encodings with the message factory for all mavlink common.xml (https://mavlink.io/en/messages/common.html#messages) and ardupilotmega.xml (https://mavlink.io/en/messages/ardupilotmega.html) messages
# instructions are described here: https://dronekit.netlify.app/guide/copter/guided_mode#guided-mode-how-to-send-commands

# Returns size of camera image in meters given camera params
def cam_params(alt, h_fov=71.5, d_fov=79.5):
    r_earth = 6378000
    cam_x = 2*(math.tan(h_fov*math.pi/2/180)*alt)
    #print(cam_x)
    cam_diag = 2*(math.tan(d_fov*math.pi/2/180)*alt)
    half_cam_diag = cam_diag/2
    #print(half_cam_diag)
    cam_y = math.sqrt(4*((math.tan(79.5*math.pi/2/180))**2)*(alt**2)-(cam_x**2))
    #print(cam_y)
    return cam_x, cam_y

# Returns size of image pixels in meters given image size
def image_pixel_sizes(cam_x, cam_y, image_size = (1920, 1080)):
    cam_x_size = cam_x / image_size[0]
    print(cam_x_size)
    cam_y_size = cam_y / image_size[1]
    print(cam_y_size)
    return cam_x_size, cam_y_size

def set_roi(vehicle, location):
    """
    Send MAV_CMD_DO_SET_ROI message to point camera gimbal at a 
    specified region of interest (LocationGlobal).
    The vehicle may also turn to face the ROI.

    For more information see: 
    http://copter.ardupilot.com/common-mavlink-mission-command-messages-mav_cmd/#mav_cmd_do_set_roi
    """
    # create the MAV_CMD_DO_SET_ROI command
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_DO_SET_ROI, #command
        0, #confirmation
        0, 0, 0, 0, #params 1-4
        location.lat,
        location.lon,
        location.alt
        )
    # send command to vehicle
    vehicle.send_mavlink(msg)

# Set zoom to high or low using AUX function, Camera Zoom is code 167 as described here by RCx_OPTIONS https://ardupilot.org/sub/docs/common-auxiliary-functions.html#common-auxiliary-functions
def setZoom(vehicle, level=0):
    
    # create the MAV_CMD_DO_SET_ROI command
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_DO_AUX_FUNCTION, #command
        0, #confirmation
        167, level, 0, 0, #params 1-4
        0,
        0,
        0
        )
    # send command to vehicle
    vehicle.send_mavlink(msg)

# Set autofocus to high or low using AUX function, Camera Autofocus is code 169 as described here by RCx_OPTIONS https://ardupilot.org/sub/docs/common-auxiliary-functions.html#common-auxiliary-functions
def setAutoFocus(vehicle, level):
    # create the MAV_CMD_DO_SET_ROI command
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_DO_AUX_FUNCTION, #command
        0, #confirmation
        169, level, 0, 0, #params 1-4
        0,
        0,
        0
        )
    # send command to vehicle
    vehicle.send_mavlink(msg)

# Set aux function to high or low, pass in RCx_OPTIONs code to auxfn (Codes: as described here by RCx_OPTIONS https://ardupilot.org/sub/docs/common-auxiliary-functions.html#common-auxiliary-functions)
def setAuxFunction(vehicle, auxfn, level):
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_DO_AUX_FUNCTION, #command
        0, #confirmation
        auxfn, level, 0, 0, #params 1-4
        0,
        0,
        0
        )
    # send command to vehicle
    vehicle.send_mavlink(msg)

def resetGimbal(vehicle):
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_GIMBAL_RESET, #command
        0, #confirmation
        0, 0, 0, 0, #params 1-4
        0, #param 5
        0, #param 6
        0 #param 7
        )
    # send command to vehicle
    vehicle.send_mavlink(msg)

def setAlt(vehicle, altitude, rate = 0.5):
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_CONDITION_CHANGE_ALT, #command
        0, #confirmation
        rate, 0, 0, 0, #params 1-4
        0, #param 5
        0, #param 6
        altitude #param 7
        )
    # send command to vehicle
    vehicle.send_mavlink(msg)

# https://mavlink.io/en/messages/common.html#MAV_CMD_NAV_CONTINUE_AND_CHANGE_ALT
def setNavAlt(vehicle, altitude, climb_dir=0):
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_NAV_CONTINUE_AND_CHANGE_ALT, #command
        0, #confirmation
        climb_dir, 0, 0, 0, #params 1-4
        0, #param 5
        0, #param 6
        altitude #param 7
        )
    # send command to vehicle
    vehicle.send_mavlink(msg)

# https://mavlink.io/en/messages/common.html#MAV_CMD_DO_CHANGE_ALTITUDE
def setAltFrame(vehicle, altitude, frame = 3):
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_DO_CHANGE_ALTITUDE, #command
        0, #confirmation
        altitude, frame, 0, 0, #params 1-4
        0, #param 5
        0, #param 6
        0 #param 7
        )
    # send command to vehicle
    vehicle.send_mavlink(msg)

def doJump(vehicle, command_num, repeat = 0):
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_DO_JUMP, #command
        0, #confirmation
        command_num, repeat, 0, 0, #params 1-4
        0, #param 5
        0, #param 6
        0 #param 7
        )
    # send command to vehicle
    vehicle.send_mavlink(msg)

def setSpeed(vehicle, speed_type, speed = -1, throttle=-1):
    # for speed_type: 0=AIRSPEED, 1=GROUNDSPEED, 2=CLIMBSPEED, 3=DESCENTSPEED
    # for speed in m/s and throttle as %, value of -2 returns value to vehicle default, -1 means no change
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_DO_JUMP, #command
        0, #confirmation
        speed_type, speed, throttle, 0, #params 1-4
        0, #param 5
        0, #param 6
        0 #param 7
        )
    # send command to vehicle
    vehicle.send_mavlink(msg)
