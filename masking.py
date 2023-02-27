import numpy as np
import argparse
import os
from color_detection import get_image, get_colors
import matplotlib.pyplot as plt
import cv2 as cv2

def get_masks(num_clusters, shape, labels):
    mask_lst = []
    for i in range(num_clusters):
        cluster_centers = np.zeros([num_clusters, 3])
        cluster_centers[i] = [255, 255, 255]
        mask = cluster_centers[labels].reshape(shape[0], shape[1], 3)
        mask_lst.append(mask)
    return mask_lst

if __name__ == '__main__':
    main_path = "C:/Users/Surya/SuryaMain/PythonProjects/suas_2022"
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to the image")
    ap.add_argument("--n_colors", type=int, help="number of different colors on image")
    args = vars(ap.parse_args())

    # load the image
    test_image = get_image(args['image'])
    n_colors = args['n_colors']
    shape = (500, 500)
    labels, ordered_colors = get_colors(test_image, shape, n_colors=n_colors)

    masks = get_masks(n_colors, shape, labels)
    for i, mask in enumerate(masks):
        cv2.imwrite(os.path.join(main_path, f"masks/{'mask'+str(i)}" + os.path.basename(args['image'])), mask)




