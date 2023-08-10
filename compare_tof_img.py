import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

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

    # 深度估计图像获取非0的index
    image_depth = image_depth[:, :, 0] + (image_depth[:, :, 1] > 0) * 255 + image_depth[:, :, 1] + (image_depth[:, :, 2] > 0) * 511 + image_depth[:, :, 2]
    image_depth_with_value = image_depth.copy()
    image_depth_with_value[image_depth_with_value > 0] = 1

    image_tof = image_tof[:, :, 0] + (image_tof[:, :, 1] > 0) * 255 + image_tof[:, :, 1] + (image_tof[:, :, 2] > 0) * 511 + image_tof[:, :, 2]
    # 仅计算深度估计非0位置
    image_tof = image_tof * image_depth_with_value

    image_tof_with_value = image_tof.copy()

    image_tof_with_value[image_tof > 0] = 1

    if image is None:
        image_depth_tof = np.abs(image_depth * image_tof_with_value - image_tof)
    else:
        image_box = (image_with_tof_box[:, :, 0] == 0) * (image_with_tof_box[:, :, 1] == 0) * (image_with_tof_box[:, :, 2] == 255) * image_tof_with_value
        image_depth_tof = np.abs(image_depth * image_box - image_tof * image_box)
    print("选中的tof点数量: {}".format(np.sum(image_box)))
    print("选中的tof点中值: ", np.median(image_tof[image_box > 0]))
    print("深度估计与选中tof点的平均误差: {:.2}".format(np.sum(image_depth_tof) / np.sum(image_box)))
    print("深度点与选中tof点最大误差: ", np.max(image_depth_tof))
    print("误差 > 2cm 点的数量： {}, 比例： {:.2%} ".format(np.sum(image_depth_tof > 2),np.sum(image_depth_tof > 2)/np.sum(image_box)))
    print("误差 > 5cm 点的数量： {}, 比例： {:.2%} ".format(np.sum(image_depth_tof > 5),np.sum(image_depth_tof > 5)/np.sum(image_box)))
    perception = image_depth_tof[image_depth_tof > 0] / (image_tof * image_box)[image_depth_tof > 0]

    print("误差比例平均值: {:.2%}".format(np.mean(perception)))

    error_bigger_0 = image_depth_tof[image_depth_tof > 0]

    plt.hist(error_bigger_0, bins=int(np.max(error_bigger_0)))
    font = FontProperties(fname=r"/usr/share/fonts/Fonts/simsun.ttf", size=14)
    plt.title("误差值--数量直方图", fontproperties=font)
    plt.xlabel("误差值：(双目-tof点）", fontproperties=font)
    plt.ylabel("数量", fontproperties=font)
    plt.xticks(np.linspace(0,np.max(error_bigger_0), num=10))
    plt.show()
    plt.hist(perception, bins=int(np.max(perception) * 100))
    font = FontProperties(fname=r"/usr/share/fonts/Fonts/simsun.ttf", size=14)
    plt.title("误差率--数量直方图", fontproperties=font)
    plt.xlabel("误差率：(双目-tof点)/tof点", fontproperties=font)
    plt.xticks(np.linspace(0,np.max(perception), num=10))
    plt.ylabel("数量", fontproperties=font)

    plt.show()

    print("误差/真实距离 < 1%的点数数量: {}, 比例： {:.2%}".format(np.sum(perception < 0.01), np.sum(perception < 0.01) / np.sum(image_box)))
    print("误差/真实距离 > 2%的点数数量: {}, 比例： {:.2%}".format(np.sum(perception > 0.02), np.sum(perception > 0.02) / np.sum(image_box)))
    print("误差/真实距离 > 5%的点数数量: {}, 比例： {:.2%}".format(np.sum(perception > 0.05) , np.sum(perception > 0.05) / np.sum(image_box)))
    image_depth_tof[image_depth_tof>0] = image_depth_tof[image_depth_tof>0] + 100
    cv2.imwrite("image_depth_tof.png", GetDepthImg(image_depth_tof))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training data
    parser.add_argument('--depth', default=None, required=True, type=str, help='Depth image from stereo depth estimation')

    parser.add_argument('--tof', default=None, type=str, help='tof point distance in image')

    parser.add_argument("--image", default=None, type=str, help="image with tof point labeled with red box")

    args = parser.parse_args()

    Print_distance(args.depth, args.tof,args.image)

