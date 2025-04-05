#Detect colors in an image and identify them
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
import argparse

def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def convert(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_colours (image,color_num):
    img = cv2.resize(image, (600, 400))
    img = img.reshape(img.shape[0] * img.shape[1], 3)
    cluster = KMeans(n_clusters = color_num)
    labels = cluster.fit_predict(img)
    ct = Counter(labels)
    center = cluster.cluster_centers_
    order = [center[i] for i in ct.keys()]
    hex_color = [convert(order[i]) for i in ct.keys()]
    for color in hex_color:
        print(f"[+] {color}")
    chart = input("\n[?] Do you want to plot a PIE Chart for it? [y/n] ")
    if chart.lower() == "y":
        plt.figure(figsize = (8, 6))
        plt.pie(ct.values(), labels = hex_color, colors = hex_color)
        plt.show()

def color_det(path, color_num):
    try:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = get_image(path)
        get_colours(img, color_num)

    except:
            print(f"Error! Please enter a valid path or try entering an image which has {color_num} colors.")
            exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--img_path', type=str, help='The path of the image you want to run color detection on.')
    parser.add_argument('-num', '--num_color', type=int, default=2, help='Number of colors in the image.')
    args = parser.parse_args()
    color_det(args.img_path, args.num_color)