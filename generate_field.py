import argparse
import cv2
import numpy as np
import math
import random

import generate_targets

# Conversion between orientation and heading angle, used by place_target
orientations = {'N': 0, 'NE': -45, 'E': -90, 'SE': -135, 'S': 180, 'SW': 135, 'W': 90, 'NW': 45}

# Given the target we want as well as some randomized information about its position, place the target onto the mock
# background.
def place_target(target_img, orientation, position, scale):
    # Target preprocessing
    alpha_channel = target_img[:, :, 3]  # Since add_noise requites an HSV conversion that doesn't preserve alpha channel, we have to save it first
    target_img = add_noise(target_img)
    transformed_img = affine_transform(target_img, orientations[orientation], scale)
    transformed_alpha = affine_transform(alpha_channel, orientations[orientation], scale)

    # Target placing
    field = alpha_blend(transformed_img, transformed_alpha, position)


# Compute the affine transformed image from a given rotation and scale
def affine_transform(img, rotation, scale):
    rotation_matrix = cv2.getRotationMatrix2D((50, 50), rotation, 1)
    scale_matrix = cv2.getRotationMatrix2D((0, 0), 0, scale)

    new_dsize = (round(img.shape[0] * scale), round(img.shape[1] * scale))
    transformed_img = cv2.warpAffine(img, rotation_matrix, img.shape[:2])
    transformed_img = cv2.warpAffine(transformed_img, scale_matrix, new_dsize)
    return transformed_img


# Reduce image quality and add lighting effects and noise to give targets the feeling of having been photographed
def add_noise(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            # Desaturate
            img[row, col, 1] *= args.lighting_constant
            # Add a noise value to each of a pixel's saturation, and value
            for i in [1, 2]:
                value = img[row, col, i]
                img[row, col, i] += min(255 - value, max(-int(value), random.randint(-args.noise_intensity, args.noise_intensity)))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img


# Superimpose the modified target image onto the background at the specified offset.
def alpha_blend(img, alpha_channel, offset):
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if alpha_channel[row, col] > 0:
                field[row + offset[0], col + offset[1], :3] = img[row, col]
    return field


# Blur the edges of targets by combining a gaussian blur of the field with a edge detection mask
def blur_edges(field):
    field_blur = cv2.GaussianBlur(field, (5, 5), 0)
    field_mask = cv2.cvtColor(cv2.Canny(field, 200, 600), cv2.COLOR_GRAY2BGR)
    blurred_edges = np.where(field_mask==0, field_blur, field)
    return blurred_edges


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--num_targets', type=int, default=5, help='The number of targets to place in this mock field.')
    parser.add_argument('-s', '--scale_target', type=float, default=0.25, help='The average scale factor for each target.')
    parser.add_argument('-sv', '--scale_variance', type=float, default=0.1, help='The multiplication factor by which the scale of a single target can vary. Set to 0 for a constant scale.')
    parser.add_argument('-l', '--lighting_constant', type=float, default=0.5, help='The amount to scale each pixel saturation by, simulating natural lighting.')
    parser.add_argument('-n', '--noise_intensity', type=int, default=10, help='The maximum increase or decrease applied to HSV values in random noise generation.')
    parser.add_argument('-c', '--clip_maximum', type=float, default=0, help='The greatest proportion of a target\'s width/height that may be out of bounds. Zero by default, but set higher to allow clipping.')
    args = parser.parse_args()

    field_name = 'maryland_test'
    field = cv2.imread('./{0}.png'.format(field_name))

    seed = 50
    random.seed(seed) # Setting the seed insures replicability of results

    for i in range(args.num_targets):
        # Randomize one target
        shape = random.choice(list(generate_targets.shapes.keys()))
        alphanum = random.choice(generate_targets.alphanumerics)
        shape_color, alphanum_color = random.sample(list(generate_targets.colors.keys()), 2)

        orientation = random.choice(list(orientations.keys()))
        scale = random.uniform((1-args.scale_variance)*args.scale_target, (1+args.scale_variance)*args.scale_target)
        pos = (round(random.uniform(-args.clip_maximum*100*scale, field.shape[0]-(1-args.clip_maximum)*100*scale)),
		round(random.uniform(-args.clip_maximum*100*scale, field.shape[1]-(1-args.clip_maximum)*100*scale)))
        place_target(generate_targets.target(shape, shape_color, alphanum, alphanum_color), orientation, pos, scale)
    field = blur_edges(field)
    # place_target(generate_targets.target('circle', 'red', 'V', 'brown'), 'SE', (190, 300), 0.25)
    
    cv2.imwrite('./tests/{0}_{1}.png'.format(field_name, seed), field)
