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
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Extract regions of interest for matching
    partialx1 = overlapx
    partialx2 = overlapx
    partialy1 = int(h*startyfrac)
    partialy2 = int(h*endyfrac)
    pimg1 = img1[partialy1:partialy2, w1-partialx1:]
    pimg2 = img2[partialy1:partialy2, 0:partialx2]
    
    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(pimg1, None)
    kp2, des2 = sift.detectAndCompute(pimg2, None)

    # Use BFMatcher with cross-checking
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    # Enhanced matching filtering
    # First, sort by match distance (SIFT descriptor distance)
    matches = sorted(matches, key=lambda x: x.distance)
    # Take top 30 matches for RANSAC
    matches = matches[:30]

    # Extract matched point coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Use RANSAC to find the best rigid transformation
    # Restrict to similarity transform (rotation + translation + uniform scale)
    transform, mask = cv2.estimateAffinePartial2D(
        pts1, pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=3,  # 3 pixel max deviation
        maxIters=2000,
        confidence=0.99,
        refineIters=10
    )  # estimateAffinePartial2D already restricts to similarity transform

    if transform is None or len(matches) < 4:
        print("No reliable transformation found, using fallback")
        return (overlapx//2, overlapx//2, 0, 0)

    # Filter matches using the RANSAC mask
    matches = [m for i, m in enumerate(matches) if mask[i][0]]

    # Enhanced debug visualization
    if transform is not None:
        rotation_angle = np.arctan2(transform[1,0], transform[0,0]) * 180 / np.pi
        scale = np.sqrt(transform[0,0]**2 + transform[1,0]**2)
        
        # Create descriptive debug message based on position
        if pos1 and pos2:
            if isinstance(pos1[0], str) and pos1[0] == 'v':
                debug_msg = f"Vertical stitch: Rows {pos1[1]}-{pos2[1]}"
            else:
                debug_msg = f"Row {pos1[0]}, Cols {pos1[1]}-{pos2[1]}"
        else:
            debug_msg = "Unknown position"
            
        print(f"\nDebug: Found transformation for {debug_msg}:")
        print(f"  - Rotation: {rotation_angle:.2f}°")
        print(f"  - Scale: {scale:.3f}")
        print(f"  - Translation: ({transform[0,2]:.1f}, {transform[1,2]:.1f})")
        
        # Save debug image with transformation info
        img_matches = cv2.drawMatches(
            pimg1, kp1, pimg2, kp2, matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # Add text with transformation info and position info
        cv2.putText(img_matches, f"{debug_msg}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(img_matches, f"Rot: {rotation_angle:.1f}deg, Scale: {scale:.2f}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        
        # Create unique filename
        if pos1 and pos2:
            if isinstance(pos1[0], str) and pos1[0] == 'v':
                filename = f'debug_match_vertical_r{pos1[1]}_r{pos2[1]}.png'
            else:
                filename = f'debug_match_horizontal_r{pos1[0]}c{pos1[1]}_c{pos2[1]}.png'
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime('%H%M%S_%f')
            filename = f'debug_match_{timestamp}.png'
            
        cv2.imwrite("debug/"+filename, img_matches)

    if len(matches) < 4:
        print("Too few good matches, using fallback")
        return (overlapx//2, overlapx//2, 0, 0)

    # Calculate median offsets from RANSAC-filtered matches
    pt_LRx = [getLeftRightX(m, kp1, kp2, partialx1) for m in matches]
    pt_Lx = [x[0] for x in pt_LRx]
    pt_Rx = [x[1] for x in pt_LRx]
    pt_deltay = [getDeltaY(m, kp1, kp2) for m in matches]
    
    lx = int(median(pt_Lx))
    rx = int(median(pt_Rx))
    vshift = int(median(pt_deltay))

    # Extract rotation angle from transformation matrix
    rotation_angle = np.arctan2(transform[1,0], transform[0,0]) * 180 / np.pi
    if abs(rotation_angle) > 5:
        print(f"Warning: Large rotation detected ({rotation_angle:.1f}°), may need attention")

    return (lx, rx, vshift, rotation_angle)


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

    # First stitch rows horizontally
    rowImages = []
    for row_idx, row in enumerate(imgMatrix):
        print(f"\nStitching row {row_idx}")
        stitched_row = stitchRow(row, OVERLAP_X, row_idx)  # Pass row_idx
        rowImages.append(stitched_row)
    
    # Resize rows to same width
    rowImages = [cv2.resize(img, (total_w, typical_h), interpolation = cv2.INTER_LINEAR) for img in rowImages]
    
    # Rotate for vertical stitching
    rotatedImages = [cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) for img in rowImages]
    
    print("\nStitching columns (rotated rows)")
    # Use modified stitchRow for vertical stitching, with 'v' prefix for debug files
    rotatedResult = stitchRow(rotatedImages, OVERLAP_Y, 'v')
    
    result = cv2.rotate(rotatedResult, cv2.ROTATE_90_CLOCKWISE)
    result = cv2.resize(result, (total_w, total_h), interpolation = cv2.INTER_LINEAR)
    return result


    
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
DIRECTORY = "5 (test)"
IMG_PATH = "result.png"
OVERLAP_X = int(0.14*KING_SIZE[0])
OVERLAP_Y = int(0.18*KING_SIZE[1])
#OVERLAP_X = int(0.2*KING_SIZE[0])
#OVERLAP_Y = int(0.1*KING_SIZE[1])


#imgArr = [[cv2.imread(f'12-picture-map-test/{i}-{j}.png') for j in range(1,c+1)] for i in range(1, r+1)]
def grabAndResize(col, row):
    pic = cv2.imread(f'{DIRECTORY}/({col}, {row}).png')
    return enforceResize(pic)

imgArr = [[grabAndResize(j, i) for j in range(COLS)] for i in range(ROWS)]
#cv2.imwrite(IMG_PATH, leftRightStitch(imgArr[0][0], imgArr[0][1], OVERLAP_X))
#cv2.imwrite('result.png', stitchRow(imgArr[0], OVERLAP_X))
cv2.imwrite(IMG_PATH, stitchMatrix(imgArr))
