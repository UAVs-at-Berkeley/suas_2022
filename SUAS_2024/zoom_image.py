from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
import argparse
from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
import imageio
import cv2
from cv2 import dnn_superres

def zoom(open_file, name):
    try:
        print(open_file)
        #sharpen(open_file)
        sr = dnn_superres.DnnSuperResImpl_create()
        """
        img = Image.open(open_file)
        width, height = img.size
        print(width, height)

        img = img.resize((width * 50, height * 50), resample=Image.LANCZOS)
        filter = ImageEnhance.Contrast(img)
        new_image = filter.enhance(1.5)
        filter = ImageEnhance.Color(new_image)
        new_image2 = filter.enhance(1.25)
        img.show()
        new_image.show()
        new_image2.show()
        img2 = new_image2.filter(filter=ImageFilter.SHARPEN)
        img2.save(f"{path}/{name}_zoom.jpg")"""
        img2 = cv2.imread(open_file)
        #dim = img2.shape
        path = "./EDSR_x4.pb"
        sr.readModel(path)
        sr.setModel("edsr", 4)
        result = sr.upsample(img2)
        #img2 = cv2.resize(img2, dsize = (dim[1] * 10, dim[0] * 10), interpolation = cv2.INTER_LANCZOS4) # or CUBIC
        #img2 = cv2.pyrUp(img2)
        cv2.imshow('Zoomed', result)
        cv2.imwrite(f"{name}_zoom.jpg", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return f"{name}_zoom.jpg"
    except IOError:
        pass

def upscale(file):
    pic = imageio.imread(file)
    #zoom_pic=ndimage.zoom(pic,0.05)

    fig=plt.figure()

    ax1=fig.add_subplot(1,3,1)
    ax1.imshow(zoom_pic,cmap='gray')
    ax1.title.set_text("Zoomed image")

    ax2=fig.add_subplot(1,3,2)
    ax2.imshow(zoom_pic,cmap=plt.cm.gray,interpolation='bilinear')
    ax2.title.set_text("Bilinear")

    ax2=fig.add_subplot(1,3,3)
    ax2.imshow(zoom_pic,cmap=plt.cm.gray,interpolation='nearest')
    ax2.title.set_text("Nearest")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--img_path', type=str, help='The path of the image you want to zoom.')
    parser.add_argument('-n', '--img_name', type=str, help='The name of the new zoomed image.')
    args = parser.parse_args()
    zoom(f"{args.img_path}/{args.img_name}.jpg", args.img_name)

#Algorithm to crop or zoom into image from camera to magnify objects of interest for classification