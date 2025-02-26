import cv2
import numpy as np

#for testing
COLS = 4
ROWS = 3
DIRECTORY = "runway1"
IMG_PATH = "runway1rotated/finalized.png"

#imgArr = [[cv2.imread(f'12-picture-map-test/{i}-{j}.png') for j in range(1,c+1)] for i in range(1, r+1)]
imgArr = [[cv2.imread(f'{DIRECTORY}/({i}, {j}).png') for j in range(COLS)] for i in range(ROWS)]

EXPECTED_H_OVERLAP = 0.01
EXPECTED_V_OVERLAP = 0.03

eho = EXPECTED_H_OVERLAP
evo = EXPECTED_V_OVERLAP

total_w = sum([x.shape[1] for x in imgArr[0]])
total_h = sum([r[0].shape[0] for r in imgArr])

