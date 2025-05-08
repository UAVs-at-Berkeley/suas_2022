import cv2
import argparse
import matplotlib.pyplot as plt

def find_largest_contour(path):
    img = cv2.imread(path)
    original_image = img

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #_, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    # show it
    #plt.imshow(binary, cmap="gray")
    plt.show()

    edges= cv2.Canny(gray, 50,200)

    contours, hierarchy= cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    cv2.destroyAllWindows()


    def get_contour_areas(contours):

        all_areas= []

        for cnt in contours:
            area= cv2.contourArea(cnt)
            all_areas.append(area)

        return all_areas


    sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)


    largest_item= sorted_contours[0]

    cv2.drawContours(original_image, largest_item, -1, (255,0,0),10)
    while(1):
        cv2.imshow('Largest Object', original_image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--img_path', type=str, help='The path of the image you want to run color detection on.')
    args = parser.parse_args()
    find_largest_contour(args.img_path)