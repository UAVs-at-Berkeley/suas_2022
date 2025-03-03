import cv2
import numpy as np
from statistics import median
from datetime import datetime

KING_SIZE = (1920, 1080)
def enforceResize(img):
    img = cv2.resize(img, KING_SIZE, interpolation = cv2.INTER_LINEAR)
    return img

#gets the difference in x pixels between two points in a SIFT matching.
def getDeltaX(m, kp1, kp2, leftwidth):
        lx = kp1[m.queryIdx].pt[0]
        rx = kp2[m.trainIdx].pt[0]
        deltax = rx+abs(leftwidth-lx)
        return deltax
def getLeftRightX(m, kp1, kp2, leftwidth):
    lx = abs(leftwidth-kp1[m.queryIdx].pt[0])
    rx = kp2[m.trainIdx].pt[0]
    return (lx, rx)
#gets the difference in y pixels between two points
def getDeltaY(m, kp1, kp2):
    ly = kp1[m.queryIdx].pt[1]
    ry = kp2[m.trainIdx].pt[1]
    deltay = ry-ly
    return deltay

def getSiftAlignment(img1, img2, overlapx, startyfrac, endyfrac, pos1=None, pos2=None):
    h, w1 = img1.shape[:2]
    
    # Initialize SIFT detector with increased contrast threshold
    sift = cv2.SIFT_create(contrastThreshold=0.04)  # Make it more selective

    # Extract regions of interest for matching
    partialx1 = overlapx
    partialx2 = overlapx
    partialy1 = int(h*startyfrac)
    partialy2 = int(h*endyfrac)
    pimg1 = img1[partialy1:partialy2, w1-partialx1:]
    pimg2 = img2[partialy1:partialy2, 0:partialx2]
    
    # Detect lines in both images using Hough transform
    gray1 = cv2.cvtColor(pimg1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(pimg2, cv2.COLOR_BGR2GRAY)
    edges1 = cv2.Canny(gray1, 50, 150)
    edges2 = cv2.Canny(gray2, 50, 150)
    lines1 = cv2.HoughLines(edges1, 1, np.pi/180, threshold=100)
    lines2 = cv2.HoughLines(edges2, 1, np.pi/180, threshold=100)
    
    # Get dominant horizontal line angles
    def get_horizontal_angle(lines):
        if lines is None:
            return 0
        angles = []
        for rho, theta in lines[:, 0]:
            # Convert theta to degrees and normalize around 0
            angle = (theta * 180 / np.pi) - 90
            # Only consider nearly horizontal lines (within ±15 degrees)
            if abs(angle) < 15:
                angles.append(angle)
        return np.median(angles) if angles else 0
    
    line_rotation1 = get_horizontal_angle(lines1)
    line_rotation2 = get_horizontal_angle(lines2)
    runway_rotation = line_rotation2 - line_rotation1
    
    # Regular SIFT matching
    kp1, des1 = sift.detectAndCompute(pimg1, None)
    kp2, des2 = sift.detectAndCompute(pimg2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:30]
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    transform, mask = cv2.estimateAffinePartial2D(
        pts1, pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=3,
        maxIters=2000
    )
    
    if transform is None or len(matches) < 4:
        print("No reliable transformation found, using fallback")
        return (overlapx//2, overlapx//2, 0, 0)
    
    # Filter matches using RANSAC mask
    matches = [m for i, m in enumerate(matches) if mask[i][0]]
    
    # Calculate SIFT-based rotation
    sift_rotation = np.arctan2(transform[1,0], transform[0,0]) * 180 / np.pi
    
    # Combine SIFT and runway line information
    if abs(runway_rotation) < 5:
        # Weight runway lines more heavily when they're detected
        final_rotation = 0.7 * runway_rotation + 0.3 * sift_rotation
    else:
        # Fall back to SIFT when runway lines aren't clear
        final_rotation = sift_rotation
    
    # Debug visualization
    img_matches = cv2.drawMatches(pimg1, kp1, pimg2, kp2, matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Draw detected lines
    if lines1 is not None:
        for rho, theta in lines1[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img_matches, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.putText(img_matches, 
                f"Runway: {runway_rotation:.1f}° SIFT: {sift_rotation:.1f}° Final: {final_rotation:.1f}°",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    if pos1 and pos2:
        filename = f'debug_match_horizontal_r{pos1[0]}c{pos1[1]}_c{pos2[1]}.png'
    else:
        from datetime import datetime
        filename = f'debug_match_{datetime.now().strftime("%H%M%S_%f")}.png'
    cv2.imwrite("debug/"+filename, img_matches)
    
    # Calculate other transformations as before
    pt_LRx = [getLeftRightX(m, kp1, kp2, partialx1) for m in matches]
    pt_Lx = [x[0] for x in pt_LRx]
    pt_Rx = [x[1] for x in pt_LRx]
    pt_deltay = [getDeltaY(m, kp1, kp2) for m in matches]
    
    lx = int(median(pt_Lx))
    rx = int(median(pt_Rx))
    vshift = int(median(pt_deltay))
    
    return (lx, rx, vshift, final_rotation)


def leftRightStitch(img1, img2, overlapx, pos1=None, pos2=None):
    # Pass position information to getSiftAlignment
    sA = getSiftAlignment(img1, img2, overlapx, 0, 1, pos1, pos2)
    hcrop = sA[0] + sA[1]
    vshift = sA[2]
    rotation = sA[3]

    # Only apply rotation if it's significant but not too large
    if 0.5 < abs(rotation) < 5.0:
        print(f"Applying rotation correction of {rotation:.2f}°")
        height, width = img2.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -rotation, 1.0)
        img2 = cv2.warpAffine(img2, rotation_matrix, (width, height))
    elif abs(rotation) >= 5.0:
        print(f"Warning: Large rotation detected {rotation:.2f}°, using previous alignment")
        rotation = 0  # Don't contribute to cumulative rotation

    # Apply stitching
    if vshift > 0:
        img1 = cv2.copyMakeBorder(img1, vshift, 0, 0, 0, cv2.BORDER_CONSTANT)
        img1 = img1[:-vshift]
    elif vshift < 0:
        img2 = cv2.copyMakeBorder(img2, -vshift, 0, 0, 0, cv2.BORDER_CONSTANT)
        img2 = img2[:vshift]
    
    img1 = img1[0:, 0:-sA[0]]
    img2 = img2[0:, sA[1]:]
    
    return cv2.hconcat((img1, img2)), rotation


#img1 = cv2.imread('12-picture-map-test/1-1.png')
#img2 = cv2.imread('12-picture-map-test/1-2.png')
def stitchRow(imgArr, overlapx, row_idx_or_prefix):
    result = imgArr[0]
    cumulative_rotation = 0  # Track total rotation applied
    
    for i in range(len(imgArr) - 1):
        if isinstance(row_idx_or_prefix, int):
            # Horizontal stitching
            pos1 = (row_idx_or_prefix, i)
            pos2 = (row_idx_or_prefix, i+1)
            print(f"Stitching horizontal pair: Row {row_idx_or_prefix}, Cols {i} and {i+1}")
        else:
            # Vertical stitching (using rotated images)
            pos1 = ('v', i)
            pos2 = ('v', i+1)
            print(f"Stitching vertical pair: Rows {i} and {i+1}")
        
        # Apply inverse of cumulative rotation to next image before matching
        if cumulative_rotation != 0:
            height, width = imgArr[i+1].shape[:2]
            center = (width // 2, height // 2)
            correction_matrix = cv2.getRotationMatrix2D(center, -cumulative_rotation, 1.0)
            corrected_img = cv2.warpAffine(imgArr[i+1], correction_matrix, (width, height))
        else:
            corrected_img = imgArr[i+1]
            
        result, new_rotation = leftRightStitch(result, corrected_img, overlapx, pos1, pos2)
        cumulative_rotation += new_rotation
        
        print(f"Cumulative rotation: {cumulative_rotation:.2f}°")
    return result

def stitchMatrix(imgMatrix):
    total_w = sum([x.shape[1] for x in imgMatrix[0]])
    total_h = sum([r[0].shape[0] for r in imgMatrix])
    typical_h = imgMatrix[0][0].shape[0]

    # First stitch rows horizontally, but maintain original orientations
    print("\nFirst pass: Basic row stitching")
    rowImages = []
    row_rotations = []  # Track rotations for each row
    
    for row_idx, row in enumerate(imgMatrix):
        print(f"\nStitching row {row_idx}")
        stitched_row, row_rotation = stitchRowWithRotation(row, OVERLAP_X, row_idx)
        rowImages.append(stitched_row)
        row_rotations.append(row_rotation)
    
    # Calculate median row rotation as reference
    median_rotation = np.median(row_rotations)
    print(f"\nMedian row rotation: {median_rotation:.2f}°")
    
    # Correct each row to match median rotation
    print("\nSecond pass: Normalizing row rotations")
    normalized_rows = []
    for idx, (row_img, row_rot) in enumerate(zip(rowImages, row_rotations)):
        correction = median_rotation - row_rot
        if abs(correction) > 0.5:  # Only correct significant differences
            print(f"Correcting row {idx} by {correction:.2f}°")
            height, width = row_img.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, correction, 1.0)
            row_img = cv2.warpAffine(row_img, rotation_matrix, (width, height))
        normalized_rows.append(row_img)
    
    # Resize rows to same width
    normalized_rows = [cv2.resize(img, (total_w, typical_h), interpolation=cv2.INTER_LINEAR) 
                      for img in normalized_rows]
    
    # Vertical stitching with minimal rotation allowed
    print("\nFinal pass: Vertical stitching")
    rotatedImages = [cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) for img in normalized_rows]
    rotatedResult = stitchRowWithRotation(rotatedImages, OVERLAP_Y, 'v', max_rotation=1.0)[0]
    
    result = cv2.rotate(rotatedResult, cv2.ROTATE_90_CLOCKWISE)
    result = cv2.resize(result, (total_w, total_h), interpolation=cv2.INTER_LINEAR)
    return result

def stitchRowWithRotation(imgArr, overlapx, row_idx_or_prefix, max_rotation=5.0):
    """Modified version that returns both result and cumulative rotation"""
    result = imgArr[0]
    cumulative_rotation = 0
    
    for i in range(len(imgArr) - 1):
        if isinstance(row_idx_or_prefix, int):
            pos1 = (row_idx_or_prefix, i)
            pos2 = (row_idx_or_prefix, i+1)
            print(f"Stitching horizontal pair: Row {row_idx_or_prefix}, Cols {i} and {i+1}")
        else:
            pos1 = ('v', i)
            pos2 = ('v', i+1)
            print(f"Stitching vertical pair: Rows {i} and {i+1}")
        
        # Get alignment with restricted rotation
        sA = getSiftAlignment(result, imgArr[i+1], overlapx, 0, 1, pos1, pos2)
        rotation = min(max(sA[3], -max_rotation), max_rotation)  # Clamp rotation
        
        # Apply transformation
        height, width = imgArr[i+1].shape[:2]
        if abs(rotation) > 0.5:
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -rotation, 1.0)
            img2 = cv2.warpAffine(imgArr[i+1], rotation_matrix, (width, height))
        else:
            img2 = imgArr[i+1]
            rotation = 0
            
        # Stitch
        if sA[2] > 0:  # vshift
            result = cv2.copyMakeBorder(result, sA[2], 0, 0, 0, cv2.BORDER_CONSTANT)
            result = result[:-sA[2]]
        elif sA[2] < 0:
            img2 = cv2.copyMakeBorder(img2, -sA[2], 0, 0, 0, cv2.BORDER_CONSTANT)
            img2 = img2[:sA[2]]
            
        result = result[:, :-sA[0]]
        img2 = img2[:, sA[1]:]
        result = cv2.hconcat((result, img2))
        
        cumulative_rotation += rotation
        
    return result, cumulative_rotation


    
#result = leftRightStitch(img1, img2)
'''
COLS = int(input("Columns: "))
ROWS = int(input("Rows: "))
DIRECTORY = input("Enter the path destination of the cut images: ").replace("\\", "/")
IMG_PATH = input("Enter the path of the output image: ").replace("\\", "/")
'''
#for testing
COLS = 4
ROWS = 3
DIRECTORY = "runway1"
IMG_PATH = "result.png"
OVERLAP_X = int(0.14*KING_SIZE[0])
OVERLAP_Y = int(0.18*KING_SIZE[1])
#OVERLAP_X = int(0.2*KING_SIZE[0])
#OVERLAP_Y = int(0.1*KING_SIZE[1])


#imgArr = [[cv2.imread(f'12-picture-map-test/{i}-{j}.png') for j in range(1,c+1)] for i in range(1, r+1)]
def grabAndResize(col, row):
    pic = cv2.imread(f'{DIRECTORY}/({row}, {col}).png')
    return enforceResize(pic)

imgArr = [[grabAndResize(j, i) for j in range(COLS)] for i in range(ROWS)]
#cv2.imwrite(IMG_PATH, leftRightStitch(imgArr[0][0], imgArr[0][1], OVERLAP_X))
#cv2.imwrite('result.png', stitchRow(imgArr[0], OVERLAP_X))
cv2.imwrite(IMG_PATH, stitchMatrix(imgArr))
