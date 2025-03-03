import cv2
import numpy as np
from statistics import median

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

def getSiftAlignment(img1, img2, overlapx, vertical_threshold):
    #get height and width
    h, w1 = img1.shape[:2]

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    partialx1 = overlapx
    partialx2 = overlapx
    partialy1 = 0
    partialy2 = h
    pimg1 = img1[partialy1:partialy2, w1-partialx1:]
    pimg2 = img2[partialy1:partialy2, 0:partialx2]
    
    kp1, des1 = sift.detectAndCompute(pimg1, None)
    kp2, des2 = sift.detectAndCompute(pimg2, None)

    # Use BFMatcher to find matches
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance
    # Filter matches based on expected vertical alignment (images should be roughly aligned)
    VERTICAL_THRESHOLD = vertical_threshold  # pixels of allowed vertical deviation
    matches = filter(lambda m: abs(getDeltaY(m, kp1, kp2)) < VERTICAL_THRESHOLD, matches)
    matches = sorted(matches, key=lambda m: getDeltaX(m, kp1, kp2, partialx1)**2+getDeltaY(m, kp1, kp2)**2)
    matches = matches[:10]
    
    img_matches = cv2.drawMatches(
        pimg1, kp1, pimg2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imwrite('test.png', img_matches)

    if len(matches) == 0:
        print("no matches found, just concatenating")
        return (overlapx//2, overlapx//2, 0, 0)

    pt_LRx = [getLeftRightX(m, kp1, kp2, partialx1) for m in matches]
    pt_Lx = [x[0] for x in pt_LRx]
    pt_Rx = [x[1] for x in pt_LRx]
    pt_deltay = [getDeltaY(m, kp1, kp2) for m in matches]
    
    lx = int(median(pt_Lx))
    rx = int(median(pt_Rx))
    vshift = int(median(pt_deltay))
    
    if abs(vshift) > VERTICAL_THRESHOLD:
        print("hmm, massive vshift. throwing it out")
        return (overlapx//2, overlapx//2, 0, 0)
        
    return (lx, rx, vshift, 0)

def leftRightStitch(img1, img2, overlapx):
    sA = getSiftAlignment(img1, img2, overlapx, 30)
    hcrop = sA[0]+sA[1]
    vshift = sA[2]
    rotation = sA[3]

    # Apply rotation to img2 if needed
    if abs(rotation) > 0:
        center = (img2.shape[1]//2, img2.shape[0]//2)
        M = cv2.getRotationMatrix2D(center, rotation, 1.0)
        img2 = cv2.warpAffine(img2, M, (img2.shape[1], img2.shape[0]))

    if vshift > 0:
        img1 = cv2.copyMakeBorder(img1, vshift, 0, 0, 0, cv2.BORDER_CONSTANT)
        img1 = img1[:-vshift]
    elif vshift < 0:
        img2 = cv2.copyMakeBorder(img2, -vshift, 0, 0, 0, cv2.BORDER_CONSTANT)
        img2 = img2[:vshift]
    
    img1 = img1[0:, 0:-sA[0]]
    img2 = img2[0:, sA[1]:]
    
    return cv2.hconcat((img1, img2))

#img1 = cv2.imread('12-picture-map-test/1-1.png')
#img2 = cv2.imread('12-picture-map-test/1-2.png')
def stitchRow(imgArr, overlapx):
    result = imgArr[0]
    i = 0
    for img in imgArr[1:]:
        result = leftRightStitch(result, img, overlapx)
        print(f"stitched together images {i} and {i+1} successfully")
        i+=1
    return result

def stitchMatrix(imgMatrix):
    total_w = sum([x.shape[1] for x in imgMatrix[0]])
    total_h = sum([r[0].shape[0] for r in imgMatrix])
    typical_h = imgMatrix[0][0].shape[0]
    

    rowImages = [stitchRow(row, OVERLAP_X) for row in imgMatrix]
    rowImages = [cv2.resize(img, (total_w, typical_h), interpolation = cv2.INTER_LINEAR) for img in rowImages]
    rotatedImages = [cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) for img in rowImages]
    rotatedResult = stitchRow(rotatedImages, OVERLAP_Y)
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

