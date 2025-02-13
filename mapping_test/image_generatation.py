import cv2
from math import floor
from math import ceil

IMG_PATH = input("Enter the path of the original image: ").replace("\\", "/")
OUTPUT_PATH = input("Enter the path destination of the cut images: ").replace("\\", "/")

COLS = int(input("Columns: "))
ROWS = int(input("Rows: "))
H_OVERLAP_RATIO = (int(input("Horizontal Overlap Percent(%): "))/100)
V_OVERLAP_RATIO = (int(input("Vertical Overlap Percent(%): "))/100)

def split_image():
    cut_images = []

    image = cv2.imread(IMG_PATH)
    height, width, channels = image.shape

    pixel_overlap_width = floor(H_OVERLAP_RATIO * width)
    pixel_overlap_height = floor(V_OVERLAP_RATIO * height)

    width_correction = ceil(pixel_overlap_width/2)
    height_correction = ceil(pixel_overlap_height/2)

    cut_height = floor(height/(ROWS))
    cut_width = floor(width/(COLS))

    print("width, height:", width, height)
    print("width, height: of overlaps", pixel_overlap_width, pixel_overlap_height)
    print("width, height of cuts:", cut_width, cut_height)

    for i in range(ROWS):
        y_start = i * cut_height - height_correction
        y_start = max(y_start, 0)
        y_stop = (i + 1) * cut_height + height_correction
        y_stop = min(y_stop, height)
        for j in range(COLS):
            x_start = j * cut_width - width_correction
            x_start = max(x_start, 0)
            x_stop = (j + 1) * cut_width + width_correction
            x_stop = min(x_stop, width)
            print()
            print(f"x:", x_start, x_stop)
            print(f"y:", y_start, y_stop)
            cut_images.append([image[y_start:y_stop, x_start:x_stop], (i, j)])

    for im in cut_images:
        cv2.imwrite(OUTPUT_PATH + f"\\{str(im[1])}.png", im[0])

if __name__ == '__main__':
    split_image()
