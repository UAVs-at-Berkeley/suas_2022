import cv2
import numpy as np

def leftRightStitch(img1, img2):
    def getDeltaX(m):
        lx = kp1[m.queryIdx].pt[0]
        rx = kp2[m.trainIdx].pt[0]
        deltax = rx+abs(w1-lx)
        return deltax
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    # Read the images to be stitched

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Use BFMatcher to find matches
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda m: getDeltaX(m))
    matches = matches[:10]
    #matches = list(filter(lambda m: getDeltaX(m) < 250, matches))


    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imwrite('test.png', img_matches)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    # Extract location of good matches
    src_pts = [kp1[m.queryIdx].pt for m in matches]
    dst_pts = [kp2[m.trainIdx].pt for m in matches]

    #leftcrop = int(sum([w1-p[0] for p in src_pts])/len(src_pts))
    #rightcrop = int(sum([p[0] for p in dst_pts])/len(dst_pts))
    pt_deltax = [getDeltaX(m) for m in matches]
    print(pt_deltax)
    pt_deltay = [-src_pts[i][1]+dst_pts[i][1] for i in range(min(len(src_pts), len(dst_pts)))]
    hcrop = int(sum(pt_deltax)/len(pt_deltax))
    vshift = int(sum(pt_deltay)/len(pt_deltay))
    #if second image is "lower" than first image, vshift will be positive

    print(hcrop, vshift)
    if vshift > 0:
        img1 = np.concatenate((np.zeros((vshift, w1, 3), dtype=np.uint8), img1),axis=0)
        newH = min(vshift+h1, h2)
    elif vshift < 0:
        img2 = np.concatenate((np.zeros((-vshift, w2, 3), dtype=np.uint8), img2),axis=0)
        newH = min(-vshift+h2, h1)
    else:
        newH = min(h2, h1)

    img1 = img1[:newH, 0:-hcrop]
    img2 = img2[:newH]

    return np.concatenate((img1, img2), axis=1)


#img1 = cv2.imread('12-picture-map-test/1-1.png')
#img2 = cv2.imread('12-picture-map-test/1-2.png')
def stitchRow(imgArr):
    result = imgArr[0]
    for img in imgArr[1:]:
        result = leftRightStitch(np.copy(result), img)
    return result

#result = leftRightStitch(img1, img2)
r = 3
c = 4
imgArr = [[cv2.imread(f'12-picture-map-test/{i}-{j}.png') for j in range(1,c+1)] for i in range(1, r+1)]
#cv2.imwrite('result.png', leftRightStitch(imgArr[0][0], imgArr[0][1]))
cv2.imwrite('result.png', stitchRow(imgArr[0]))
