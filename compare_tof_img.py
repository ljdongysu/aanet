import cv2
import numpy as np
import argparse

def GetDepthImg(img):
    depth_img_rest = img.copy()
    depth_img_R = depth_img_rest.copy()
    depth_img_R[depth_img_rest > 255] = 255
    depth_img_rest[depth_img_rest < 255] = 255
    depth_img_rest -= 255
    depth_img_G = depth_img_rest.copy()
    depth_img_G[depth_img_rest > 255] = 255
    depth_img_rest[depth_img_rest < 255] = 255
    depth_img_rest -= 255
    depth_img_B = depth_img_rest.copy()
    depth_img_B[depth_img_rest > 255] = 255
    depth_img_rgb = np.stack([depth_img_R, depth_img_G, depth_img_B], axis=2)
    return depth_img_rgb.astype(np.uint8)
def Print_distance(depth, tof, image=None):

    if image is None:
        pass
    else:
        image_with_tof_box = cv2.imread(image)
    image_depth = cv2.imread(depth)
    image_tof = cv2.imread(tof)
    image_tof = image_tof[:, :, 0] + (image_tof[:, :, 1] > 0) * 255 + image_tof[:, :, 1] + (image_tof[:, :, 2] > 0) * 511 + image_tof[:, :, 2]
    image_tof_with_value = image_tof.copy()
    image_tof_with_value[image_tof > 0] = 1
    image_depth = image_depth.sum(axis=2)
    if image is None:
        image_depth_tof = np.abs(image_depth * image_tof_with_value - image_tof)
    else:
        image_box = (image_with_tof_box[:, :, 0] == 0) * (image_with_tof_box[:, :, 1] == 0) * (image_with_tof_box[:, :, 2] == 255) * image_tof_with_value
        image_depth_tof = np.abs(image_depth * image_box - image_tof * image_box)

    print("选中的tof点数量: {}，全部tof点数量： {}".format(np.sum(image_box), np.sum(image_tof_with_value)))
    print("选中的tof点中值: ", np.median(image_tof[image_box > 0]))
    print("深度估计与选中tof点的平均误差: ", np.sum(image_depth_tof) / np.sum(image_box))
    print("深度点与选中tof点最大误差: ", np.max(image_depth_tof))
    print("误差 > 5 点的数量： {}, 比例： {:.2%} ".format(np.sum(image_depth_tof > 5),np.sum(image_depth_tof > 5)/np.sum(image_tof_with_value)))
    print("误差 > 10 点的数量： {}, 比例： {:.2%} ".format(np.sum(image_depth_tof > 10),np.sum(image_depth_tof > 10)/np.sum(image_tof_with_value)))
    print("误差 > 20 点的数量： {}, 比例： {:.2%} ".format(np.sum(image_depth_tof > 20),np.sum(image_depth_tof > 20)/np.sum(image_tof_with_value)))
    cv2.imwrite("image_depth_tof.png", GetDepthImg(image_depth_tof))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training data
    parser.add_argument('--depth', default=None, required=True, type=str, help='Depth image from stereo depth estimation')

    parser.add_argument('--tof', default=None, type=str, help='tof point distance in image')

    parser.add_argument("--image", default=None, type=str, help="image with tof point labeled with red box")

    args = parser.parse_args()

    Print_distance(args.depth, args.tof,args.image)

