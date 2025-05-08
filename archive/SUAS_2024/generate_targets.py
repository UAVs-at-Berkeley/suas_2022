import argparse
import cv2
import numpy as np
import math
import os

# The characteristics we'll be mixing and matching to generate our targets

shapes = {'circle': np.array([(round(50 + 50 * math.cos(t * math.pi/180)), round(50 - 50 * math.sin(t * math.pi/180))) for t in range(0, 360, 4)]),
          'semicircle': np.array([(round(50 + 50 * math.cos(t * math.pi/180)), round(75 - 50 * math.sin(t * math.pi/180))) for t in range(0, 181, 4)]),
          'quartercircle': np.array([(25, 75)] + [(round(25 + 75 * math.cos(t * math.pi/180)), round(75 - 75 * math.sin(t * math.pi/180))) for t in range(0, 91, 3)]),
          'triangle': np.array([(50, 0), (100, 75), (0, 75)]),
          'rectangle': np.array([(25, 0), (75, 0), (75, 100), (25, 100)]),
          'pentagon': np.array([(50, 0), (98, 35), (79, 90), (21, 90), (2, 35)]),
          'star': np.array([(50, 0), (61, 35), (98, 35), (68, 56), (79, 90), (50, 69), (21, 90), (32, 56), (2, 35), (39, 35)]),
          'cross': np.array([(33, 0), (66, 0), (66, 33), (100, 33), (100, 66), (66, 66), (66, 100), (33, 100), (33, 66), (0, 66), (0, 33), (33, 33)])}

colors = {'white': (255, 255, 255, 255),
          'black': (0, 0, 0, 255),
          'red': (0, 0, 255, 255),
          'blue': (255, 0, 0, 255),
          'green': (0, 255, 0, 255),
          'yellow': (0, 255, 255, 255),
          'purple': (255, 0, 128, 255),
          'brown': (0, 100, 150, 255)}

alphanumerics = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'


# Generator for a blank RGBA image, used as the background our shapes will be drawn on
def blank(n):
    img = np.zeros((n, n, 4), np.uint8)
    return img


# Generate a target image with the specified charateristics in string form
def target(shape, shape_color, alphanum, alphanum_color):
    img = blank(100)
    cv2.fillPoly(img, [shapes[shape]], colors[shape_color])
    retval, _ = cv2.getTextSize(alphanum, cv2.FONT_HERSHEY_DUPLEX, 1.5, 4)
    cv2.putText(img, alphanum, (int(50 - retval[0] / 2), int(50 + retval[1] / 2)), cv2.FONT_HERSHEY_DUPLEX, 1.5,
                colors[alphanum_color], 4)
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate png files of all possible targets falling within a specified range of characteristics.')
    parser.add_argument('-s', '--shape', nargs='+', help='The range of shapes to use.')
    parser.add_argument('-sc', '--shape_color', nargs='+', help='The range of shape colors to use.')
    parser.add_argument('-a', '--alphanum', required=False, help='The range of alphanumeric symbols to use.')
    parser.add_argument('-ac', '--alphanum_color', nargs='+', help='The range of alphanumeric colors to use.')
    parser.add_argument('-w', '--write_targets', action='store_true', help='Save the generated targets to pngs.')
    parser.add_argument('-all', '--all_targets', action='store_true' ,help='Generated all possible targets to pngs.')
    args = parser.parse_args()
    print(args)

    if args.all_targets:
        os.system('python generate_targets.py -s circle semicircle quartercircle triangle square rectangle trapezoid pentagon hexagon heptagon octagon star cross -sc black gray red blue green yellow purple brown orange -a 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ -ac white black gray red blue green yellow purple brown orange -w')

    else:
        for shape in args.shape:
            for shape_color in args.shape_color:
                for alphanum in args.alphanum:
                    for alphanum_color in args.alphanum_color:
                        # print(shape, shape_color, alphanum, alphanum_color)
                        # print(target)
                        if shape_color != alphanum_color:
                            img = target(shape, shape_color, alphanum, alphanum_color)
                            if args.write_targets:
                                cv2.imwrite('./targets/{0}_{1}_{2}_{3}.png'.format(shape, shape_color, alphanum, alphanum_color), img)
                            else:
                                cv2.imshow('result', img)
                                cv2.waitKey(0) 
    
# Loop through all possible combinations of characteristics
#     for shape in shapes.keys():
#         for shape_color in colors.keys():
#             for alphanum in alphanumerics:
#                 for alphanum_color in colors.keys():
#                     if shape_color != alphanum_color:
#                         target(shape, shape_color, alphanum, alphanum_color)
