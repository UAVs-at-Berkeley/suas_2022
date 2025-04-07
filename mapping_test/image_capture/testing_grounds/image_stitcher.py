import cv2
import numpy as np
from statistics import median
import os

PATH_OF_SCRIPT = os.path.dirname(os.path.abspath(__file__)) #local directory, NOT the working directory
KING_SIZE = (1920, 1080)
OVERLAP_X = int(0.2*KING_SIZE[0])
OVERLAP_Y = int(0.3*KING_SIZE[1])
DEBUG = True

def enforceResize(img) -> np.ndarray:
    img = cv2.resize(img, KING_SIZE, interpolation = cv2.INTER_LINEAR)
    return img

class ImageNode:

    def __init__(self, img: np.ndarray, row: int, col: int):
        self.img = img
        self.row = row
        self.col = col

        self.rightNeighbor = None
        self.rightNeighborMatches = None
        self.rightComparsionImage = None
        self.rightNeighborKP1 = None
        self.rightNeighborKP2 = None

        self.downNeighbor = None
        self.downNeighborMatches = None
        self.downComparsionImage = None
        self.downNeighborKP1 = None
        self.downNeighborKP2 = None

    def initRightNeighbor(self, rightNeighbor, siftMethod):
        self.rightNeighbor = rightNeighbor
        self.setRightNeighborSIFT(siftMethod)

    def initDownNeighbor(self, downNeighbor, siftMethod):
        self.downNeighbor = downNeighbor
        self.setDownNeighborSIFT(siftMethod)

    def setRightNeighborSIFT(self, siftMethod):
        #Slice self.img so that it only contains the right part of the image, and is overlapx pixels wide
        compImg = self.img[:, KING_SIZE[0] - OVERLAP_X:]

        #Slice self.rightNeighbor.img so that it only contains the left part of the image, and is overlapx pixels wide
        compRightNeighborImg = self.rightNeighbor.img[:, :OVERLAP_X]

        #Make compImg and compRightNeighborImg grayscale
        compImg = cv2.cvtColor(compImg, cv2.COLOR_BGR2GRAY)
        compRightNeighborImg = cv2.cvtColor(compRightNeighborImg, cv2.COLOR_BGR2GRAY)

        #Detect SIFT features
        kp1, des1 = siftMethod.detectAndCompute(compImg, None)
        kp2, des2 = siftMethod.detectAndCompute(compRightNeighborImg, None)

        #Create a BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        #Match descriptors
        matches = bf.match(des1, des2)

        #Sort them in the order of their distance
        matches = sorted(matches, key = lambda x: x.distance)[:30]

        #Draw the top 30 matches
        img_matches = cv2.drawMatches(compImg, kp1, compRightNeighborImg, kp2, matches, None, flags=None)
        self.rightNeighborKP1 = kp1
        self.rightNeighborKP2 = kp2
        self.rightNeighborMatches = matches
        self.rightComparsionImage = img_matches


    def setDownNeighborSIFT(self, siftMethod):
        #Slice self.img so that it only contains the bottom part of the image, and is overlapy pixels high
        compImg = self.img[KING_SIZE[1] - OVERLAP_Y:, :]

        #Slice self.downNeighbor.img so that it only contains the top part of the image, and is overlapy pixels high
        compDownNeighborImg = self.downNeighbor.img[:OVERLAP_Y, :]

        #Make compImg and compRightNeighborImg grayscale
        compImg = cv2.cvtColor(compImg, cv2.COLOR_BGR2GRAY)
        compDownNeighborImg = cv2.cvtColor(compDownNeighborImg, cv2.COLOR_BGR2GRAY)

        #Detect SIFT features
        kp1, des1 = siftMethod.detectAndCompute(compImg, None)
        kp2, des2 = siftMethod.detectAndCompute(compDownNeighborImg, None)

        #Create a BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        #Match descriptors
        matches = bf.match(des1, des2)

        #Sort them in the order of their distance
        matches = sorted(matches, key = lambda x: x.distance)[:30]

        #Draw the top 30 matches
        img_matches = cv2.drawMatches(compImg, kp1, compDownNeighborImg, kp2, matches, None, flags=None)
        self.downNeighborKP1 = kp1
        self.downNeighborKP2 = kp2
        self.downNeighborMatches = matches
        self.downComparsionImage = img_matches

    def imWriteComparsionRight(self, path):
        cv2.imwrite(path+f"/rightComparsion{self.row}{self.col}.png", self.rightComparsionImage)

    def imWriteComparsionDown(self, path):
        cv2.imwrite(path+f"/downComparsion{self.row}{self.col}.png", self.downComparsionImage)

# Takes in a 2D array of images and stitches them together
def stitchMatrix(imgMatrix, subfolder) -> np.ndarray:

    # Get the dimensions of the first image in the matrix
    if len(imgMatrix) == 0 or len(imgMatrix[0]) == 0:
        raise ValueError("Image matrix is empty")
        
    height, width = imgMatrix[0][0].shape[:2]
    rows = len(imgMatrix)
    cols = len(imgMatrix[0])

    # Create a 2D array of ImageNodes
    imgNodes = [[ImageNode(imgMatrix[i][j], i, j) for j in range(cols)] for i in range(rows)]

    #Initialize SIFT detector (or ORB detector)
    sift = cv2.ORB_create()

    #Initialize neighbors and write comparisons if debug enabled
    for i in range(rows):
        for j in range(cols):
            if j < cols - 1:
                imgNodes[i][j].initRightNeighbor(imgNodes[i][j+1], sift)
                if DEBUG:
                    imgNodes[i][j].imWriteComparsionRight(subfolder)
            if i < rows - 1:
                imgNodes[i][j].initDownNeighbor(imgNodes[i+1][j], sift)
                if DEBUG:
                    imgNodes[i][j].imWriteComparsionDown(subfolder)

    #Create a new image that is the size of the final stitched image
    finalImg = np.zeros((rows * (KING_SIZE[1]), cols * (KING_SIZE[0]), 3), dtype=np.uint8)

    #Get variables to calculate the position of the image in the final image
    imgHeight = KING_SIZE[1]
    imgWidth = KING_SIZE[0]
    overlapHeight = OVERLAP_Y
    overlapWidth = OVERLAP_X

    #Put the first image in the final image, starting at (OverlapX, OverlapY)
    srcTri = np.array([[0, 0], [1, 0], [0, 1]]).astype(np.float32)
    dstTri = np.array([[0, 0], [2, 0], [0.5, 1.5]]).astype(np.float32)
    affineMatrix = cv2.getAffineTransform(srcTri, dstTri)
    print(affineMatrix)
    imgNodes[0][0].img = cv2.warpAffine(imgNodes[0][0].img, affineMatrix, (3*imgWidth, 2*imgHeight))
    h, w = imgNodes[0][0].img.shape[:2]
    finalImg[overlapHeight:overlapHeight+h, overlapWidth:overlapWidth+w] = imgNodes[0][0].img

    # For each image, calculate the best way to rotate and translate it so that it fits with its neighbors
    for i in range(rows):
        for j in range(cols):
            if j < cols - 1:
                # Get the best matching keypoints between current image and right neighbor
                m = imgNodes[i][j].rightNeighborMatches
                for match in m:
                    i1 = match.queryIdx  # Index of keypoint in current image
                    i2 = match.trainIdx  # Index of matching keypoint in right neighbor
                    p1 = imgNodes[i][j].rightNeighborKP1[i1].pt  # Point coordinates in current image
                    p2 = imgNodes[i][j].rightNeighborKP2[i2].pt  # Point coordinates in right neighbor
                    #print(p1, p2)
            if i < rows - 1:
                # Get the best matching keypoints between current image and down neighbor
                m = imgNodes[i][j].downNeighborMatches
                for match in m:
                    i1 = match.queryIdx  # Index of keypoint in current image
                    i2 = match.trainIdx  # Index of matching keypoint in down neighbor
                    p1 = imgNodes[i][j].downNeighborKP1[i1].pt  # Point coordinates in current image
                    p2 = imgNodes[i][j].downNeighborKP2[i2].pt  # Point coordinates in down neighbor
                    #print(p1, p2)

            #Calculate the transform

            
    

    # Return the stitched image
    return finalImg

def grabAndResize(col, row) -> np.ndarray:
    # Load image from file using row and column coordinates, then resize to standard dimensions
    pic = cv2.imread(f'{PATH_OF_SCRIPT}/({col}, {row}).png')
    return enforceResize(pic)

def stitch_all(cols, rows):
    # Create a 2D array of images by loading and resizing each image based on row and column position
    imgArr = [[grabAndResize(j, i) for j in range(cols)] for i in range(rows)]
    
    # Find the next available result file number by checking existing files
    i = 0
    while os.path.exists(f'{PATH_OF_SCRIPT}/result{str(i)}'):
        i += 1
    
    # Create a new directory under path_of_script called result{str(i)}

    subfolder = f'{PATH_OF_SCRIPT}/result{str(i)}'
    
    os.makedirs(subfolder, exist_ok=True)
    
    # Stitch all images together using stitchMatrix and save the result
    cv2.imwrite(f'{subfolder}/finalized.png', stitchMatrix(imgArr, subfolder))
