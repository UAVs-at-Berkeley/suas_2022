from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import urllib 

"""lower = {'red':(166, 84, 141), 'green':(66, 122, 129), 'blue':(97, 100, 117), 'yellow':(23, 59, 119), 'orange':(0, 50, 80)} 
upper = {'red':(186,255,255), 'green':(86,255,255), 'blue':(117,255,255), 'yellow':(54,255,255), 'orange':(20,255,255)}
 

colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0), 'yellow':(0, 255, 217), 'orange':(0,140,255)}
 
def color_det(path):
    camera = cv2.VideoCapture(0)

    while True:

        (grabbed, frame) = camera.read()

        if not grabbed:
            break

        image = cv2.imread(path)
        
        frame = imutils.resize(frame, width=900)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)

        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        for key, value in upper.items():
        
            kernel = np.ones((9,9),np.uint8)
            mask = cv2.inRange(hsv, lower[key], upper[key])
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
        
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)[-2]
            center = None
        


            if len(cnts) > 0:
                
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        

                if radius > 0.5:
                
                    cv2.circle(frame, (int(x), int(y)), int(radius), colors[key], 5)
                    cv2.putText(frame, key + "color", (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[key],2)
        
        cv2.imshow("Image", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            camera.release()
            cv2.destroyAllWindows()
            break"""

def color_det(img_in):

    # Capturing video through webcam
    webcam = cv2.VideoCapture(0)
    
    max_contours = dict.fromkeys(['max1', 'max2'], {'contour': None, 'color': 'Black', 'area': 0})

    # Start a while loop
    #while(1):
        
    # Reading the video from the
    # webcam in image frames
    #_, imageFrame = webcam.read()

    #img = cv2.imread(path)
    img = imutils.resize(img_in, width=700)
    # Convert the imageFrame in 
    # BGR(RGB color space) to 
    # HSV(hue-saturation-value)
    # color space
    hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    """{'black': [[180, 255, 30], [0, 0, 0]],
            'white': [[180, 18, 255], [0, 0, 231]],
            'brown': [[22, 255, 200], [10, 70, 20]],
            'red1': [[180, 255, 255], [159, 50, 70]],
            'red2': [[9, 255, 255], [0, 50, 70]],
            'green': [[89, 255, 255], [36, 50, 70]],
            'blue': [[128, 255, 255], [90, 50, 70]],
            'yellow': [[35, 255, 255], [25, 50, 70]],
            'purple': [[158, 255, 255], [129, 50, 70]],
            'orange': [[24, 255, 255], [10, 50, 70]],
            'gray': [[180, 18, 230], [0, 0, 40]]}"""

    # Set range for black color and 
    # define mask
    black_lower = np.array([90, 0, 0], np.uint8)
    black_upper = np.array([180, 255, 108], np.uint8)
    black_mask = cv2.inRange(hsvFrame, black_lower, black_upper)

    # Set range for white color and 
    # define mask
    white_lower = np.array([0, 0, 221], np.uint8)
    white_upper = np.array([180, 18, 255], np.uint8)
    white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)

    # Set range for brown color and
    # define mask
    brown_lower = np.array([15, 70, 50], np.uint8)
    brown_upper = np.array([30, 255, 230], np.uint8)
    brown_mask = cv2.inRange(hsvFrame, brown_lower, brown_upper)

    # Set range for yellow color and 
    # define mask
    yellow_lower = np.array([26, 70, 90], np.uint8)
    yellow_upper = np.array([28, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

    # Set range for purple color and 
    # define mask
    purple_lower = np.array([121, 50, 70], np.uint8)
    purple_upper = np.array([158, 255, 255], np.uint8)
    purple_mask = cv2.inRange(hsvFrame, purple_lower, purple_upper)

    # Set range for gray color and 
    # define mask

    # Set range for red color and 
    # define mask
    red_lower = np.array([0, 50, 70], np.uint8)
    red_upper = np.array([9, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    # Set range for green color and 
    # define mask
    green_lower = np.array([50, 50, 70], np.uint8)
    green_upper = np.array([80, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    # Set range for blue color and
    # define mask
    blue_lower = np.array([90, 50, 70], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
    
    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernal = np.ones((5, 5), "uint8")
    
    # For red color
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(img, img, 
                            mask = red_mask)
    
    # For green color
    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(img, img,
                                mask = green_mask)
    
    # For blue color
    blue_mask = cv2.dilate(blue_mask, kernal)
    res_blue = cv2.bitwise_and(img, img,
                            mask = blue_mask)
    
    # For black color
    black_mask = cv2.dilate(black_mask, kernal)
    res_black = cv2.bitwise_and(img, img, 
                            mask = black_mask)
    
    # For white color
    white_mask = cv2.dilate(white_mask, kernal)
    res_white = cv2.bitwise_and(img, img,
                                mask = white_mask)
    
    # For brown color
    brown_mask = cv2.dilate(brown_mask, kernal)
    res_brown = cv2.bitwise_and(img, img,
                            mask = brown_mask)
    
    # For yellow color
    yellow_mask = cv2.dilate(yellow_mask, kernal)
    res_yellow = cv2.bitwise_and(img, img, 
                            mask = yellow_mask)

    
    # For purple color
    purple_mask = cv2.dilate(purple_mask, kernal)
    res_purple = cv2.bitwise_and(img, img,
                            mask = purple_mask)
    
    
    color_masks = {'Red': red_mask, 'Green': green_mask, 'Blue': blue_mask, 'Black': black_mask, 'White': white_mask, 'Brown': brown_mask, 'Yellow': yellow_mask, 'Purple': purple_mask}
    
    # Creating contour to track black color
    contours, hierarchy = cv2.findContours(black_mask,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
    
    max_area_black = (None, 0)
    
    for contour in list(contours):
        area = cv2.contourArea(contour)
        if(area > max_area_black[1]):
            max_area_black = (contour, area)
            
    if max_area_black[1] > max_contours['max1']['area']:
        max_contours['max2'] = max_contours['max1']
        max_contours['max1'] = {'contour': max_area_black[0], 'color': 'Black', 'area': max_area_black[1]}
    elif max_area_black[1] > max_contours['max2']['area']:
        max_contours['max2'] = {'contour': max_area_black[0], 'color': 'Black', 'area': max_area_black[1]}
            
    # Creating contour to track white color
    contours, hierarchy = cv2.findContours(white_mask,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
    
    max_area_white = (None, 0)
    
    for contour in list(contours):
        area = cv2.contourArea(contour)
        if(area > max_area_white[1]):
            max_area_white = (contour, area) 
            
    if max_area_white[1] > max_contours['max1']['area']:
        max_contours['max2'] = max_contours['max1']
        max_contours['max1'] = {'contour': max_area_white[0], 'color': 'White', 'area': max_area_white[1]}
    elif max_area_white[1] > max_contours['max2']['area']:
        max_contours['max2'] = {'contour': max_area_white[0], 'color': 'White', 'area': max_area_white[1]}

    # Creating contour to track brown color
    contours, hierarchy = cv2.findContours(brown_mask,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
    
    max_area_brown = (None, 0)
    
    for contour in list(contours):
        area = cv2.contourArea(contour)
        if(area > max_area_brown[1]):
            max_area_brown = (contour, area)
            
    if max_area_brown[1] > max_contours['max1']['area']:
        max_contours['max2'] = max_contours['max1']
        max_contours['max1'] = {'contour': max_area_brown[0], 'color': 'Brown', 'area': max_area_brown[1]}
    elif max_area_brown[1] > max_contours['max2']['area']:
        max_contours['max2'] = {'contour': max_area_brown[0], 'color': 'Brown', 'area': max_area_brown[1]}
            
    # Creating contour to track yellow color
    contours, hierarchy = cv2.findContours(yellow_mask,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
    
    max_area_yellow = (None, 0)

    for contour in list(contours):
        area = cv2.contourArea(contour)
        if(area > max_area_yellow[1]):
            max_area_yellow = (contour, area)
            
    if max_area_yellow[1] > max_contours['max1']['area']:
        max_contours['max2'] = max_contours['max1']
        max_contours['max1'] = {'contour': max_area_yellow[0], 'color': 'Yellow', 'area': max_area_yellow[1]}
    elif max_area_yellow[1] > max_contours['max2']['area']:
        max_contours['max2'] = {'contour': max_area_yellow[0], 'color': 'Yellow', 'area': max_area_yellow[1]}
            
    # Creating contour to track purple color
    contours, hierarchy = cv2.findContours(purple_mask,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
    
    max_area_purple = (None, 0)

    for contour in list(contours):
        area = cv2.contourArea(contour)
        if(area > max_area_purple[1]):
            max_area_purple = (contour, area)
            
    if max_area_purple[1] > max_contours['max1']['area']:
        max_contours['max2'] = max_contours['max1']
        max_contours['max1'] = {'contour': max_area_purple[0], 'color': 'Purple', 'area': max_area_purple[1]}
    elif max_area_purple[1] > max_contours['max2']['area']:
        max_contours['max2'] = {'contour': max_area_purple[0], 'color': 'Purple', 'area': max_area_purple[1]}

    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(red_mask,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
    
    max_area_red = (None, 0)
    
    for contour in list(contours):
        area = cv2.contourArea(contour)
        if(area > max_area_red[1]):
            max_area_red = (contour, area)
            
    if max_area_red[1] > max_contours['max1']['area']:
        max_contours['max2'] = max_contours['max1']
        max_contours['max1'] = {'contour': max_area_red[0], 'color': 'Red', 'area': max_area_red[1]}
    elif max_area_red[1] > max_contours['max2']['area']:
        max_contours['max2'] = {'contour': max_area_red[0], 'color': 'Red', 'area': max_area_red[1]}

    # Creating contour to track green color
    contours, hierarchy = cv2.findContours(green_mask,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
    
    max_area_green = (None, 0)
    
    for contour in list(contours):
        area = cv2.contourArea(contour)
        if(area > max_area_green[1]):
            max_area_green = (contour, area)
            
    if max_area_green[1] > max_contours['max1']['area']:
        max_contours['max2'] = max_contours['max1']
        max_contours['max1'] = {'contour': max_area_green[0], 'color': 'Green', 'area': max_area_green[1]}
    elif max_area_green[1] > max_contours['max2']['area']:
        max_contours['max2'] = {'contour': max_area_green[0], 'color': 'Green', 'area': max_area_green[1]}

    # Creating contour to track blue color
    contours, hierarchy = cv2.findContours(blue_mask,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
    
    max_area_blue = (None, 0)

    for contour in list(contours):
        area = cv2.contourArea(contour)
        if(area > max_area_blue[1]):
            max_area_blue = (contour, area)
            
    if max_area_blue[1] > max_contours['max1']['area']:
        max_contours['max2'] = max_contours['max1']
        max_contours['max1'] = {'contour': max_area_blue[0], 'color': 'Blue', 'area': max_area_blue[1]}
    elif max_area_blue[1] > max_contours['max2']['area']:
        max_contours['max2'] = {'contour': max_area_blue[0], 'color': 'Blue', 'area': max_area_blue[1]}


    x, y, w, h = cv2.boundingRect(max_contours['max1']['contour'])
    img = cv2.rectangle(img, (x, y),
                            (x + w, y + h),
                            (0, 0, 0), 2)
    
    cv2.putText(img, f"{max_contours['max1']['color']} Colour", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0))
    
    x, y, w, h = cv2.boundingRect(max_contours['max2']['contour'])
    img = cv2.rectangle(img, (x, y),
                            (x + w, y + h),
                            (0, 0, 0), 2)
    
    cv2.putText(img, f"{max_contours['max2']['color']} Colour", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0))
            
    # Program Termination
    #cv2.imshow("Multiple Color Detection in Real-Time", img)
    #cv2.imshow(f"{max_contours['max1']['color']} Mask", color_masks[max_contours['max1']['color']])
    #cv2.imshow(f"{max_contours['max2']['color']} Mask", color_masks[max_contours['max2']['color']])

    #if cv2.waitKey(10) & 0xFF == ord('q'):
        #print(f"Max colors are {max_contours['max1']['color']} and {max_contours['max2']['color']}")
        #webcam.release()
        #cv2.destroyAllWindows()
        #break
    return (max_contours['max1']['color'], max_contours['max2']['color'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--img_path', type=str, help='The path of the image you want to run color detection on.')
    args = parser.parse_args()
    YOUR_FN_NAME_HERE(args.img_path)
