import cv2
import imutils
from math import floor, ceil
from random import uniform

RANDOM_VARIATION = input("Include random variation in overlapping images? (y/n): ").lower() == 'y'
SAVE_FULL_IMAGES = True
MAX_ROTATION = int(input("Enter the maximum degrees of rotation (too much may result in black space): "))

IMG_PATH = input("Enter the path of the original image: ").replace("\\", "/")
OUTPUT_PATH = input("Enter the path destination of the cut images: ").replace("\\", "/")

COLS = int(input("Columns: "))
ROWS = int(input("Rows: "))

H_OVERLAP_RATIO = (int(input("Horizontal Overlap Percent(%): "))/100)
V_OVERLAP_RATIO = (int(input("Vertical Overlap Percent(%): "))/100)

'''
#For Testing

RANDOM_VARIATION = True
SAVE_FULL_IMAGES = True

MAX_ROTATION = 5

IMG_PATH = path
OUTPUT_PATH = path

COLS = 10
ROWS = 5

H_OVERLAP_RATIO = .1
V_OVERLAP_RATIO = .1
'''

def split_image():
    cut_images = []

    image = cv2.imread(IMG_PATH)
    real_height, real_width, channels = image.shape

    #throws out 20% of the image to allow for rotational slicing
    height = floor(real_height * 0.8)
    width = floor(real_width * 0.8)

    initial_height_offset, initial_width_offset = ceil(height * 0.1), ceil(width * 0.1)

    cut_height = floor(height/(ROWS))
    cut_width = floor(width/(COLS))

    pixel_overlap_width = floor(H_OVERLAP_RATIO * cut_width)
    pixel_overlap_height = floor(V_OVERLAP_RATIO * cut_height)

    width_correction = ceil(pixel_overlap_width/2)
    height_correction = ceil(pixel_overlap_height/2)

    print("(modified) width, height:", width, height)
    print("width, height of overlaps:", pixel_overlap_width, pixel_overlap_height)
    print("width, height of cuts:", cut_width, cut_height)

    for i in range(ROWS):
        if RANDOM_VARIATION:
            random_height_factor = ceil(uniform(- pixel_overlap_height * 0.5, pixel_overlap_height * 0.5))
        else:
            random_height_factor = 0

        y_start = i * cut_height - height_correction + initial_height_offset + random_height_factor
        y_stop = (i + 1) * cut_height + height_correction + initial_height_offset + random_height_factor

        if y_start < 0 or y_stop > 0:
            y_start = i * cut_height - height_correction + initial_height_offset - random_height_factor
            y_stop = (i + 1) * cut_height + height_correction + initial_height_offset - random_height_factor

        for j in range(COLS):
            if RANDOM_VARIATION:
                random_width_factor = ceil(uniform(- pixel_overlap_width * 0.5, pixel_overlap_width * 0.5))
            else:
                random_width_factor = 0
                
            x_start = j * cut_width - width_correction + initial_width_offset + random_width_factor
            x_stop = (j + 1) * cut_width + width_correction + initial_width_offset + random_width_factor

            if x_start < 0 or x_stop > 0:
                x_start = j * cut_width - width_correction + initial_width_offset - random_width_factor
                x_stop = (j + 1) * cut_width + width_correction + initial_width_offset - random_width_factor

            if (i, j) == (0, 0):
                rotated = image
            else:
                rotation_matrix = cv2.getRotationMatrix2D(((x_start + x_stop)/2, (y_start + y_stop)/2), uniform(-MAX_ROTATION, MAX_ROTATION), 1.0)
                rotated = cv2.warpAffine(image, rotation_matrix, (real_width, real_height))

            cut_images.append([rotated[y_start:y_stop, x_start:x_stop], (j, i)])

    for im in cut_images:
        cv2.imwrite(OUTPUT_PATH + f"\\{str(im[1])}.png", im[0])

    if SAVE_FULL_IMAGES:
        merged_rows = []
        for row in range(ROWS):
            merged_rows.append(cv2.hconcat([im[0] for im in cut_images[row * COLS: (row + 1) * COLS]]))
            
        full_image = cv2.vconcat(merged_rows)
        cv2.imwrite(OUTPUT_PATH + f"\\full.png", full_image)
        cv2.imwrite(OUTPUT_PATH + f"\\original.png", image)
    
    print(f"Images successfully created in {OUTPUT_PATH}")

if __name__ == '__main__':
    split_image()
