import cv2
import numpy as np
from statistics import median
import os

PATH_OF_SCRIPT = os.path.dirname(os.path.abspath(__file__)) #local directory, NOT the working directory
KING_SIZE = (1920, 1080)
OVERLAP_X = int(0.2*KING_SIZE[0])
OVERLAP_Y = int(0.3*KING_SIZE[1])
DEBUG = True
#round the chained affine matrix to the nearest 1/ROUND_TO_IDENTITY
ROUND_TO_IDENTITY = 100

'''
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

def getSiftAlignment(img1, img2, overlapx, startyfrac, endyfrac):
    #get height and width
    h, w1 = img1.shape[:2]
    #h, w = img1.shape[:2]
    # Read the images to be stitched

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    partialx1 = overlapx
    partialx2 = overlapx
    partialy1 = int(h*startyfrac)
    partialy2 = int(h*endyfrac)
    pimg1 = img1[partialy1:partialy2, w1-partialx1:]
    pimg2 = img2[partialy1:partialy2, 0:partialx2]
    
    kp1, des1 = sift.detectAndCompute(pimg1, None)
    kp2, des2 = sift.detectAndCompute(pimg2, None)

    # Use BFMatcher to find matches
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = filter(lambda m: abs(getDeltaY(m, kp1, kp2)) < OVERLAP_Y, matches)
    matches = sorted(matches, key=lambda m: getDeltaX(m, kp1, kp2, partialx1)**2+getDeltaY(m, kp1, kp2)**2)
    matches = matches[:10]
    #matches = list(filter(lambda m: getDeltaX(m) < 250, matches))

    
    img_matches = cv2.drawMatches(
        pimg1, kp1, pimg2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imwrite('test.png', img_matches)

    if len(matches) == 0:
        print("no matches found, just concatenating")
        return (overlapx//2, overlapx//2, 0)

    pt_LRx = [getLeftRightX(m, kp1, kp2, partialx1) for m in matches]
    pt_Lx = [x[0] for x in pt_LRx]
    pt_Rx = [x[1] for x in pt_LRx]
    pt_deltay = [getDeltaY(m, kp1, kp2) for m in matches]
    #hcrop = int(sum(pt_deltax)/len(pt_deltax))
    #lx = int(sum(pt_Lx)/len(pt_Lx))
    #rx = int(sum(pt_Rx)/len(pt_Rx))
    lx = int(median(pt_Lx))
    rx = int(median(pt_Rx))
    #vshift = int(sum(pt_deltay)/len(pt_deltay))
    vshift = int(median(pt_deltay))
    if abs(vshift) > OVERLAP_Y:
        print("hmm, massive vshift. throwing it out")
        #print([(kp1[m.queryIdx].pt[1], kp2[m.trainIdx].pt[1]) for m in matches])
        #print([getDeltaY(m, kp1, kp2) for m in matches])
        return (overlapx//2, overlapx//2, 0)
    return (lx, rx, vshift)
'''

def enforceResize(img) -> np.ndarray:
    img = cv2.resize(img, KING_SIZE, interpolation = cv2.INTER_LINEAR)
    return img

def matrix_similarity_to_identity(matrix):
    # Extract the 2x2 submatrix
    submatrix = matrix[:2, :2]
    # Create the 2x2 identity matrix
    identity = np.eye(2)
    # Calculate the Frobenius norm of the difference
    return np.linalg.norm(submatrix - identity, 'fro')

def chainAffine(matrix1, matrix2) -> np.ndarray:
    #matrix1 and matrix2 are 2x3 matrices
    #return the composition of the two matrices
    #add a bottom row of [0, 0, 1] to each matrix
    matrix1 = np.vstack([matrix1, [0, 0, 1]])
    matrix2 = np.vstack([matrix2, [0, 0, 1]])
    #multiply the two matrices
    result = np.dot(matrix1, matrix2)
    #remove the bottom row
    result = result[:2, :]
    return result

class ImageEdge:
    #Keeps track of all the information about how two images are positioned relative to each other
    def __init__(self, parentNode, childNode):
        self.parentNode = parentNode
        self.childNode = childNode
        self.kp1 = None
        self.kp2 = None
        self.matches = None
        self.img_matches = None
        self.affineMatrix = None

class ImageNode:

    def __init__(self, img: np.ndarray, row: int, col: int):
        self.img = img
        self.row = row
        self.col = col
        self.rightNeighbor = None
        self.downNeighbor = None
        self.leftAffineMatrix = None
        self.upAffineMatrix = None
        self.chainedAffineMatrix = None
        self.warpedImg = None

    def initRightNeighbor(self, rightNeighbor, siftMethod):
        self.rightNeighbor = rightNeighbor
        self.rightNeighborEdge = ImageEdge(self, rightNeighbor)
        #Slice self.img so that it only contains the right part of the image, and is overlapx pixels wide
        compImg = self.img[:, KING_SIZE[0] - OVERLAP_X:]

        #Slice self.rightNeighbor.img so that it only contains the left part of the image, and is overlapx pixels wide
        compRightNeighborImg = self.rightNeighbor.img[:, :OVERLAP_X]
        self.setNeighborSIFT(siftMethod, self.rightNeighborEdge, compImg, compRightNeighborImg)
        rightNeighbor.leftAffineMatrix = self.rightNeighborEdge.affineMatrix

    def initDownNeighbor(self, downNeighbor, siftMethod):
        self.downNeighbor = downNeighbor
        self.downNeighborEdge = ImageEdge(self, downNeighbor)
        #Slice self.img so that it only contains the bottom part of the image, and is overlapy pixels high
        compImg = self.img[KING_SIZE[1] - OVERLAP_Y:, :]

        #Slice self.downNeighbor.img so that it only contains the top part of the image, and is overlapy pixels high
        compDownNeighborImg = self.downNeighbor.img[:OVERLAP_Y, :]
        self.setNeighborSIFT(siftMethod, self.downNeighborEdge, compImg, compDownNeighborImg)
        downNeighbor.upAffineMatrix = self.downNeighborEdge.affineMatrix

    def setNeighborSIFT(self, siftMethod, neighborEdge, compImg, compNeighborImg):

        #Make compImg and compNeighborImg grayscale
        compImg = cv2.cvtColor(compImg, cv2.COLOR_BGR2GRAY)
        compNeighborImg = cv2.cvtColor(compNeighborImg, cv2.COLOR_BGR2GRAY)

        #Detect SIFT features
        kp1, des1 = siftMethod.detectAndCompute(compImg, None)
        kp2, des2 = siftMethod.detectAndCompute(compNeighborImg, None)

        #Create a BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        #Match descriptors
        matches = bf.match(des1, des2)

        #Sort them in the order of their distance
        matches = sorted(matches, key = lambda x: x.distance)[:30]

        #Draw the top 3 matches
        img_matches = cv2.drawMatches(compImg, kp1, compNeighborImg, kp2, matches, None, flags=None)

        #calculate the affine matrix.
        matches = matches[:3]
        #The source points, SPECFICALLY FOR THE PURPOSES OF AFFINE TRANSFORMATION, 
        # are the points in the neigbhor image, because the affine matrix will transform the 
        # points in the neighbor image to the points in the current image
        #Therefore we use kp2 for the source points
        srcPts = [kp2[m.trainIdx].pt for m in matches]
        #The destination points are the points in the current image
        dstPts = [kp1[m.queryIdx].pt for m in matches]

        #Shift the points in srcPts and dstPts by row*KING_SIZE[1] and col*KING_SIZE[0]
        #recall that the points are in the form (x, y), not (y, x), and that srcPts should be shifted by the right neighbor's coordinates, and dstPts should be shifted by the current image's coordinates
        neighbor = neighborEdge.childNode
        adjustedSrcPts = []
        adjustedDstPts = []
        for pt in srcPts:
            adjustedSrcPts.append((pt[0]+KING_SIZE[0]*neighbor.col, pt[1]+KING_SIZE[1]*neighbor.row))
        for pt in dstPts:
            if neighbor.row == self.row:
                #right neighbor
                adjustedDstPts.append((pt[0]+KING_SIZE[0]*self.col+KING_SIZE[0]-OVERLAP_X, pt[1]+KING_SIZE[1]*self.row))
            else:
                #down neighbor
                adjustedDstPts.append((pt[0]+KING_SIZE[0]*self.col, pt[1]+KING_SIZE[1]*self.row+KING_SIZE[1]-OVERLAP_Y))
        srcTri = np.array(adjustedSrcPts).astype(np.float32)
        dstTri = np.array(adjustedDstPts).astype(np.float32)

        #create the affine matrix
        affineMatrix = cv2.getAffineTransform(srcTri, dstTri)

        #affineMatrix = np.round(affineMatrix*ROUND_TO_IDENTITY)/ROUND_TO_IDENTITY
        '''if any(abs(affineMatrix[0][:2]) > 2) or any(abs(affineMatrix[1][:2]) > 2):
            print(f"affineMatrix is too big: {affineMatrix}. Resetting the rotation to identity.")
            affineMatrix[:2, :2] = np.eye(2, 2)'''

        if DEBUG:
            #put text over top of img_matches in the corner, with:
            # the row and col of the current image
            # the row and col of the neighbor image
            # the affine matrix
            cv2.putText(img_matches, f"Current: row {self.row}, col {self.col}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(img_matches, f"Neighbor: row {neighbor.row}, col {neighbor.col}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(img_matches, f"Affine Matrix Row 1: {[round(x, 3) for x in affineMatrix[0]]}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(img_matches, f"Affine Matrix Row 2: {[round(x, 3) for x in affineMatrix[1]]}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        #update neighborEdge with all the new information in case we need it later
        neighborEdge.affineMatrix = affineMatrix
        neighborEdge.kp1 = kp1
        neighborEdge.kp2 = kp2
        neighborEdge.matches = matches
        neighborEdge.img_matches = img_matches

    def imWriteComparsionRight(self, path):
        cv2.imwrite(path+f"/rightComparsion{self.row}{self.col}.png", self.rightNeighborEdge.img_matches)

    def imWriteComparsionDown(self, path):
        cv2.imwrite(path+f"/downComparsion{self.row}{self.col}.png", self.downNeighborEdge.img_matches)

    def createWarpedImg(self, rows, cols, imgNodes):

        #chain the affine matrices of all the images in the path to the current image and just go down and right.
        #images with multiple neighbors can compare affine matrices.
        # For example, the image at row 1 column 1 can get two different chained affine
        # matrices depending on whether it's going down or right first. We want to compare the
        # two to see which one is better.

        if self.row == 0 and self.col == 0:
            #since we are at the top left corner, we have no neighbors
            self.chainedAffineMatrix = np.eye(2, 3)
        elif self.row == 0:
            #since we are at the top row, we only have one neighbor to the left
            self.chainedAffineMatrix = chainAffine(self.leftAffineMatrix, imgNodes[self.row][self.col-1].chainedAffineMatrix)
        elif self.col == 0:
            #since we are at the leftmost column, we only have one neighbor to the top
            self.chainedAffineMatrix = chainAffine(self.upAffineMatrix, imgNodes[self.row-1][self.col].chainedAffineMatrix)
        else:
            #generate two possibile chained affine matrices, and choose the one with the closest determinant to 1
            upChain = chainAffine(self.upAffineMatrix, imgNodes[self.row-1][self.col].chainedAffineMatrix)
            leftChain = chainAffine(self.leftAffineMatrix, imgNodes[self.row][self.col-1].chainedAffineMatrix)
            self.chainedAffineMatrix = min(upChain, leftChain, key=matrix_similarity_to_identity)
        #Create a new image that is the size of the final stitched image but a little larger
        bigImg = np.zeros((rows * (KING_SIZE[1]), cols * (KING_SIZE[0]), 3), dtype=np.uint8)
        #Put the first image in the final image, starting at (OverlapX, OverlapY)
        bIh, bIw = bigImg.shape[:2]
        h, w = self.img.shape[:2]
        x = KING_SIZE[0]*self.col
        y = KING_SIZE[1]*self.row
        bigImg[y:y+h, x:x+w] = self.img
        #apply the affine matrix to the bigImg
        self.warpedImg = cv2.warpAffine(bigImg, self.chainedAffineMatrix, (bIw, bIh))
    
    def imWriteWarped(self, path):
        cv2.imwrite(path+f"/warped{self.row}{self.col}.png", self.warpedImg)

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

    #Get variables to calculate the position of the image in the final image
    imgHeight = KING_SIZE[1]
    imgWidth = KING_SIZE[0]
    overlapHeight = OVERLAP_Y
    overlapWidth = OVERLAP_X

    #imgNodes[0][0].img = cv2.warpAffine(imgNodes[0][0].img, affineMatrix, (3*imgWidth, 2*imgHeight))

    # Create the warped images by going in a diagonal path through the matrix
    for L in range(rows+cols-1):
        for i in range(L+1):
            j = L-i
            if i < rows and j < cols:
                imgNodes[i][j].createWarpedImg(rows, cols, imgNodes)
                if DEBUG:
                    imgNodes[i][j].imWriteWarped(subfolder)
    
    #Create the final image
    finalImg = np.zeros((rows * (KING_SIZE[1]), cols * (KING_SIZE[0]), 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            finalImg = cv2.max(finalImg, imgNodes[i][j].warpedImg)

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
