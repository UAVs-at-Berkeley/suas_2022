from PIL import Image
import argparse

def zoom(path, name):
    open_file = f"{path}/{name}.jpg"
    try:
        print(open_file)
        img = Image.open(open_file)
        width, height = img.size
        print(width, height)

        img = img.resize((width * 50, height * 50))

        img.save(f"{path}/{name}_zoom.jpg")
    except IOError:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--img_path', type=str, help='The path of the image you want to zoom.')
    parser.add_argument('-n', '--img_name', type=str, help='The name of the new zoomed image.')
    args = parser.parse_args()
    zoom(args.img_path, args.img_name)
    img2 = Image.open("./maryland_test")
    width2, height2 = img2.size
    print(width2, height2)
#Algorithm to crop or zoom into image from camera to magnify objects of interest for classification