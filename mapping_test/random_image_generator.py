import cv2
import numpy as np
from random import randint
from random import sample
from collections import deque
from copy import deepcopy

PATH = "C:/Users/jonat/.vscode/uav/suas_2022/mapping_test/numbergrid.png"
HEIGHT = 1000
WIDTH = 1500
SPLOTCH_SIZE = 50
SPLOTCH_COUNT = 50

def create_blank_canvas(width, height):
    arr = []
    for i in range(HEIGHT):
        row = []
        for j in range(WIDTH):
            row.append(np.array([255, 255, 255]))
        row = np.array(row)
        arr.append(row)
    arr = np.array(arr, dtype= np.uint8)
    return arr

def generate_points(num_points, size, canvas_width, canvas_height, canvas):
    '''
    Returns a new numpy array with randomly generated splotches with unqiue colors\n 
    Pure function
    '''
    channel_1 = sample(range(255), num_points)
    channel_2 = sample(range(255), num_points)
    channel_3 = sample(range(255), num_points)
    colors = deque([np.array([channel_1[i], channel_2[i], channel_3[i]]) for i in range(num_points)])

    new_canvas = deepcopy(canvas)

    for i in range(num_points):
        x_pos = randint(0, canvas_width - 1 - size)
        y_pos = randint(0, canvas_height - 1 - size)
        curr_color = colors.popleft()
        for y in range(y_pos, y_pos + SPLOTCH_SIZE):
            for x in range(x_pos, x_pos + SPLOTCH_SIZE):
                new_canvas[y][x] = curr_color

    return new_canvas

if __name__ == '__main__':
    image = create_blank_canvas(WIDTH, HEIGHT)
    image =  generate_points(SPLOTCH_COUNT, SPLOTCH_SIZE, WIDTH, HEIGHT, image)
    cv2.imshow('image', image)
    cv2.imwrite(r'c:\Users\jonat\.vscode\uav\suas_2022\mapping_test\random_overlap_images\1.png', image)
    cv2.waitKey()
