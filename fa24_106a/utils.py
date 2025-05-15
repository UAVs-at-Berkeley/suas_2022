import cv2
import math
import time
import statistics
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
from operator import itemgetter

def drawRectangles(img_gray, r, min_gap, white_thresh, drawing_img):
    h, w = img_gray.shape
    # convert to BGR so we can overwrite with white easily
    out = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # --- find all candidate (white) pixels -------------------------------------
    ys, xs = np.where(img_gray >= white_thresh)
    candidates = list(zip(xs, ys))          # (x, y) coordinates

    # --- greedy placement of rectangles ---------------------------------------
    centres: list[tuple[int, int]] = []     # accepted rectangle centres

    for x, y in candidates:
        # skip if too close to an existing rectangle
        if any(math.hypot(x - cx, y - cy) < min_gap for cx, cy in centres):
            continue

        # clamp rectangle to stay inside image borders
        tl = (max(0, x - r), max(0, y - r))          # top-left
        br = (min(w - 1, x + r), min(h - 1, y + r))  # bottom-right

        # cv2.rectangle(drawing_img, tl, br, color=(0, 0, 0), thickness=-1)
        cv2.circle(drawing_img, tl, 2, color=(0, 0, 0), thickness=-1)
        # cv2.circle(drawing_img, tl, 2, color=(255, 255, 255), thickness=-1)
        centres.append((x, y))

    return drawing_img


def get_vector_metres(lat1, lon1, lat2, lon2):
    dlat = lat2 - lat1
    dlong = lon2 - lon1
    return dlat * 1.113195e5, dlong * 1.113195e5



def get_distance_metres_pts(aLocation1, aLocation2):
    """
    Returns the ground distance in metres between two `LocationGlobal` or `LocationGlobalRelative` objects.

    This method is an approximation, and will not be accurate over large distances and close to the
    earth's poles. It comes from the ArduPilot test code:
    https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
    """
    dlat = aLocation2.lat - aLocation1.lat
    dlong = aLocation2.lon - aLocation1.lon
    return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5

def get_distance_metres(lat1, lon1, lat2, lon2):
    """
    Returns the ground distance in metres between two `LocationGlobal` or `LocationGlobalRelative` objects.

    This method is an approximation, and will not be accurate over large distances and close to the
    earth's poles. It comes from the ArduPilot test code:
    https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
    """
    dlat = lat2 - lat1
    dlong = lon2 - lon1
    return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5

def angle_bound(angle):
    if angle < 0.2 or angle > 6.08:
        return 0 # angle is negligible
    elif angle < 1:
        return 1 # angle is reasonable 
    else:
        return 2 #continue because the angle is an error
    
import re
from datetime import datetime

def parse_metadata(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by subtitle blocks
    blocks = content.strip().split('\n\n')
    # print(blocks) 
    # EX) '1\n00:00:00,000 --> 00:00:00,033\n<font size="28">FrameCnt: 1, 
    # DiffTime: 33ms\n2025-05-07 16:02:57.632\n[iso: 100] [shutter: 1/1750.36] 
    # [fnum: 1.7] [ev: 0] [color_md : default] [focal_len: 24.00] [latitude: 37.871293]
    #  [longitude: -122.317558] [rel_alt: 106.500 abs_alt: 71.130] [ct: 5025] </font>',
    results = []

    for block in blocks:
        lines = block.strip().split('\n')
        # detects
        # print(lines)
        # if len(lines) < 4:
        #     continue

        frame_num = int(lines[0].strip())
        timestamp_range = lines[1].strip()
        metadata_line = '\n'.join(lines[2:]).strip()
        # print(metadata_line)
        
        # Extract timestamp info
        start_time, end_time = timestamp_range.split(' --> ')

        # Parse frame count and diff time
        framecnt_match = re.search(r'FrameCnt: (\d+), DiffTime: (\d+)ms', metadata_line)
        if framecnt_match:
            framecnt = int(framecnt_match.group(1))
            difftime = int(framecnt_match.group(2))
        else:
            continue
        # print(framecnt, difftime)

        # Parse datetime
        datetime_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)', metadata_line)
        timestamp = datetime.strptime(datetime_match.group(1), "%Y-%m-%d %H:%M:%S.%f") if datetime_match else None

        # Extract key-value fields in brackets
        kv_matches = re.findall(r'\[(.*?)\]', metadata_line)
        # print(kv_matches)
        kv_data = {}
        for kv in kv_matches:
            if 'rel_alt' in kv and 'abs_alt' in kv:
                match = re.search(r'rel_alt: ([\d.]+) abs_alt: ([\d.]+)', kv)
                if match:
                    kv_data['rel_alt'] = float(match.group(1))
                    kv_data['abs_alt'] = float(match.group(2))
            elif ':' in kv:
                key, val = kv.split(':', 1)
                try:
                    kv_data[key.strip()] = float(val.strip())
                except ValueError:
                    kv_data[key.strip()] = val.strip()


        # Convert types where relevant
        float_fields = ['iso', 'shutter', 'fnum', 'ev', 'focal_len', 'latitude', 'longitude', 'ct']
        for k in float_fields:
            if k in kv_data:
                try:
                    kv_data[k] = float(kv_data[k])
                except:
                    pass
        print()

        results.append({
            'frame': framecnt,
            'diff_ms': difftime,
            'start_time': start_time,
            'end_time': end_time,
            'datetime': timestamp,
            **kv_data
        })

    return results

# Example usage
parsed = parse_metadata("DJI_20250507160257_0026_D.SRT")

# # Optional: print or save as CSV
# import pandas as pd
# df = pd.DataFrame(parsed)
# df.to_csv("parsed_output.csv", index=False)
# print(df.head())


def truncated_adaptive_gamma(image, tau=0.3, alpha=0.2):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Histogram: P_i
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    n = gray.size
    P_i = hist / n

    # Smooth min/max (ignore zero bins)
    nonzero = P_i[P_i > 0]
    P_min = np.min(nonzero)
    P_max = np.max(nonzero)

    # Weighted PDF: P_wi
    normalized = (P_i - P_min) / (P_max - P_min)
    normalized = np.clip(normalized, 0, 1)  # ensures all values in [0, 1]
    P_wi = P_max * (normalized ** alpha)
    P_wi[P_i <= P_min] = 0  # clamp negatives

    # Cumulative weighted distribution: C_wi
    C_wi = np.cumsum(P_wi) / np.sum(P_wi)

    # Adaptive gamma for each intensity
    gamma_i = 1.0 - C_wi
    gamma = np.maximum(tau, gamma_i)

    # Build gamma LUT
    lut = np.array([255 * ((i / 255.0) ** gamma[i]) for i in range(256)]).astype(np.uint8)

    # Apply LUT
    result = cv2.LUT(gray, lut)

    return result

# def isolateCurve(grey_image):
#     # Edge detection
#     blurred = cv2.GaussianBlur(grey_image, (11, 11), 0)
#     edges = cv2.Canny(blurred, 100, 200)
    
#     # Contour finding
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Initialize variables to store the best contour and its area
#     len_threshold = 2
#     best_contour = None
#     best_contours = []
#     max_area = -1

#     # Curved line isolation
#     for contour in contours:
#         # ===== isolate single best contours =====
#         # # Calculate the area of the contour
#         # area = cv2.contourArea(contour)
        
#         # # Approximate the contour to simplify its shape
#         # perimeter = cv2.arcLength(contour, True)
#         # approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        
#         # # Check if the contour is curved based on the number of vertices
#         # if len(approx) > 5 and area > max_area:
#         #    best_contour = contour
#         #    max_area = area
#         # ========================================
#         # Calculate the area of the contour
#         area = cv2.contourArea(contour)
        
#         # Approximate the contour to simplify its shape
#         perimeter = cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
#         if len(approx) > len_threshold:
#             best_contours.append(contour)


#     # Masking and extraction
#     mask = np.zeros_like(blurred)
#     # cv2.drawContours(mask, [best_contour], -1, 255, cv2.FILLED)
#     cv2.drawContours(mask, best_contours, -1, 255, cv2.FILLED)
#     result = cv2.bitwise_and(grey_image, grey_image, mask=mask)    
#     return result, mask

# def hsv_filter(image, lower_hsv, upper_hsv):
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
#     result = cv2.bitwise_and(image, image, mask=mask)
#     return result, mask

# def remove_inner_contours(binary_image):
#     contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#     image_copy = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
#     if hierarchy is not None:
#         for i, contour in enumerate(contours):
#             # Check if the contour has a parent
#             if hierarchy[0][i][3] != -1:
#                 # Draw the inner contour in green
#                 cv2.drawContours(image_copy, [contour], -1, (0, 255, 0), cv2.FILLED)
#     green = np.array([0, 255, 0])
#     green_mask = np.all(image_copy == green, axis=2)
#     image_copy[green_mask] = [0, 0, 0]
#     result = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
#     return result

# def remove_smaller_contours(binary_image, max_contour_len):
#     contours, _ = cv2.findContours(eroded_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) <= max_contour_len]

#     image_copy = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
#     cv2.drawContours(image_copy, filtered_contours, -1, (255, 0, 0), cv2.FILLED)
#     blue = np.array([255, 0, 0])
#     blue_mask = np.all(image_copy == blue, axis=2)
#     image_copy[blue_mask] = [0, 0, 0]
#     result = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
#     return result


still_image_dict = {
    0:('media/37.872310N_122.322454W_231.23H_297.8W.png', 37.872310, 122.322454, 231.23, 297.8), 
    # 1:('37.872312N_122.319072W_235.56H_364.08W.png', 37.872312, 122.319072, 235.56, 364.08), 
    1:('media/dji_pic.png', 37.8719660, 122.3186288, 102, 150), 
    2:('media/37.874496H_122.322454W_242.73H_297.8W.png', 37.874496, 122.322454, 242.73, 297.8), 
    3:('media/37.874496N_122.319072W_242.73H_364.08W.png', 37.874496, 122.319072, 242.73, 364.08)
}

still_image_dict = {
    0:('media/37.872310N_122.322454W_231.23H_297.8W.png', 37.872310, 122.322454, 231.23, 297.8), 
    # 1:('media/37.872312N_122.319072W_235.56H_364.08W.png', 37.872312, 122.319072, 235.56, 364.08), 
    # 1:('media/pair2.png', 37.8722765, 122.3193286, 279.09, 318),
    1:('media/pair3.png', 37.8714926, 122.3184300, 81.3, 160.08), 

    # 1:('media/dji_pic.png', 37.8719660, 122.3186288, 102, 150), 
    2:('media/37.874496H_122.322454W_242.73H_297.8W.png', 37.874496, 122.322454, 242.73, 297.8), 
    3:('media/37.874496N_122.319072W_242.73H_364.08W.png', 37.874496, 122.319072, 242.73, 364.08)
}


still_image_dict = {
    0:('ref/earth1.png', 37.8710573, 122.3165491, 78.25, 154.75), 
    0:('ref/earth2.png', 37.8710387, 122.3183065, 191.64, 96.13), 
    0:('ref/earth3.png', 37.8712660, 122.3197400, 78.25, 97.5), 
    0:('ref/earth4.png', 37.8717180, 122.3190810, 63.4, 92.08), 
    0:('ref/earth5.png', 37.8724938, 122.3201798, 71.73, 119.2), 
    0:('ref/earth6.png', 37.8729508, 122.3182209, 84, 112.8), 
    0:('ref/earth7.png', 37.8716630, 122.3179006, 64, 106.3)
}

still_image_dict = ('DJI_20250507160257_0026_D.png', 37.8714865, 122.3183067, 66, 117)
