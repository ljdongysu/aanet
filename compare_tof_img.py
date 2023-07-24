import cv2
import numpy as np
import argparse
def Print_distance(depth, tof):

    image_depth = cv2.imread(depth)
    image_tof = cv2.imread(tof)
    image_tof = image_tof[:, :, 0] + (image_tof[:, :, 1] > 0) * 255 + image_tof[:, :, 1] + (image_tof[:, :, 2] > 0) * 511 + image_tof[:, :, 2]
    image_tof_with_value = image_tof.copy()
    image_tof_with_value[image_tof > 0] = 1
    image_depth = image_depth.sum(axis=2)
    image_depth_tof = np.abs(image_depth * image_tof_with_value - image_tof)
    print("The number tof points: ", np.sum(image_tof_with_value))
    print("median distance of tof: ", np.median(image_tof[image_tof > 0]))
    print("mean distance between tof and depth: ", np.sum(image_depth_tof) / np.sum(image_tof_with_value))
    print("max distance between tof and depth: ", np.max(image_depth_tof), "min distance: ", np.min(image_depth_tof))
    print("The number of point's distance between tof and depth > 5: ", np.sum(image_depth_tof > 5),
          ', ratio: {:.2%}'.format(np.sum(image_depth_tof > 5)/np.sum(image_tof_with_value)))
    print("The number of point's distancebetween tof and depth > 10 : ", np.sum(image_depth_tof > 10),
          ', ratio: {:.2%}'.format(np.sum(image_depth_tof > 10)/np.sum(image_tof_with_value)))
    print("The number of point's distance between tof and depth > 20: ", np.sum(image_depth_tof > 20),
          ', ratio: {:.2%}'.format(np.sum(image_depth_tof > 20)/np.sum(image_tof_with_value)))
    cv2.imwrite("image_depth_tof.png", image_depth_tof + 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training data
    parser.add_argument('--depth', default=None, required=True, type=str, help='Depth image from stereo depth estimation')

    parser.add_argument('--tof', default=None, type=str, help='tof point distance in image')
    args = parser.parse_args()

    Print_distance(args.depth, args.tof)

