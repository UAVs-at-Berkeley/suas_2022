import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter


def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# number_of_colors parameter tells cluster algorithm how many colors to search for 
# ideal img would be cropped to have two primary colors (color of shape and alphanumeric)
def get_colors(img, number_of_colors):
    # resizes image to speed up processing
    modified_image = cv2.resize(img, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    counts = Counter(labels)
    center_colors = [list(map(int, lst)) for lst in clf.cluster_centers_]
    # ordered_colors contains the rgb values in order of most frequent to least
    ordered_colors = [center_colors[i] for i in counts.keys()]
    return ordered_colors

img = cv2.imread('./img_simulator/winter_small_10.jpg')
crop = img[455:500, 625:675]
# crop = img[461:495, 632:665]

# Crop settings for other images
# img = cv2.imread('./img_simulator/shrubbery_small_34.jpg')
# crop = img[440:480, 275:315]
# img = cv2.imread('./img_simulator/meadow_20.jpg')
# crop = img[90:140, 310:360]

img = crop
# img = cv2.GaussianBlur(img, (7, 7), 0)
cv2.imshow('test', img)
cv2.waitKey(0)
# exit()

# Set K means parameters and run the algorithm
img = np.float32(img)
img_stretched = img.reshape((-1, 3))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
ret, label, center = cv2.kmeans(img_stretched, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# print(center)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv2.imshow('res2', res2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Retrieve the color corresponding to the most common label in the image.
# We assume that this color cluster corresponds to the shape.
unique_labels, count = np.unique(label, return_counts=True)
shape_color = center[unique_labels[np.argmax(count)]]
# Convert shape color to HSV for easier classification
print(cv2.cvtColor(np.array([[shape_color]]), cv2.COLOR_BGR2HSV))

# Jason: add code to
# 1.) floodfill the outside of res2 with the shape color (I recommend you use the pixel at (0, 0)
# as a starting point and if that's already colored, loop over the image array until you find one that isn't,
# 2.) Conditionally overlay the floodfilled res2 over crop, and
# 3.) Run K-means with K=2 to get text color.
# Once that's done, you're welcome to try sorting the shape_color list into an actual color: you can find the
# list of valid colors in /img_simulator/generate_targets.py. Otherwise, I can take care of it.


