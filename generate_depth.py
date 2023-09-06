import time

import cv2
import torch
import torch.nn.functional as F

import skimage.io
import argparse
import numpy as np
import os
import math

import nets
from dataloader import transforms
from utils import utils
from utils.file_io import write_pfm
from glob import glob
from utils.file_io import read_img
from numpy import savez_compressed
import re

__all__ = ["GetDepthImgPSL", "Walk", "MkdirSimple", "GetDepthImg"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def MkdirSimple(path):
    path_current = path
    suffix = os.path.splitext(os.path.split(path)[1])[1]

    if suffix != "":
        path_current = os.path.dirname(path)
        if path_current in ["", "./", ".\\"]:
            return
    if not os.path.exists(path_current):
        os.makedirs(path_current)

def Walk(path, suffix:list):
    file_list = []
    suffix = [s.lower() for s in suffix]
    if not os.path.exists(path):
        print("not exist path {}".format(path))
        return []

    if os.path.isfile(path):
        return [path,]

    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1].lower()[1:] in suffix:
                file_list.append(os.path.join(root, file))

    try:
        file_list.sort(key=lambda x:int(re.findall('\d+', os.path.splitext(os.path.basename(x))[0])[0]))
    except:
        pass

    return file_list
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

def GetDepthImgPSL(img):
    depth_img_rest = img.copy()
    depth_img_R = depth_img_rest.copy()
    depth_img_R[depth_img_rest > 255] = 0
    depth_img_rest[depth_img_rest < 255] = 255
    depth_img_rest -= 255
    depth_img_G = depth_img_rest.copy()
    depth_img_G[depth_img_rest > 255] = 0
    depth_img_rest[depth_img_rest < 255] = 255
    depth_img_rest -= 255
    depth_img_B = depth_img_rest.copy()
    depth_img_B[depth_img_rest > 255] = 255
    depth_img_rgb = np.stack([depth_img_R, depth_img_G, depth_img_B], axis=2)
    return depth_img_rgb.astype(np.uint8)


def WriteDepthScale(depth, limg, path, name, scale=1):
    # output_concat_color = os.path.join(path, "concat_color", name)
    # output_concat_gray = os.path.join(path, "concat_gray", name)
    # output_gray = os.path.join(path, "gray", name)
    # output_gray_scale = os.path.join(path, "gray_scale", name)
    output_depth = os.path.join(path, name)
    # output_depth_psl = os.path.join(path, "depth_psl", name)
    # output_color = os.path.join(path, "color", name)
    # output_concat_depth = os.path.join(path, "concat_depth", name)
    # output_concat = os.path.join(path, "concat", name)
    # output_display = os.path.join(path, "display", name)
    # MkdirSimple(output_concat_color)
    # MkdirSimple(output_concat_gray)
    # MkdirSimple(output_concat_depth)
    # MkdirSimple(output_gray)
    MkdirSimple(output_depth)
    # MkdirSimple(output_depth_psl)
    # MkdirSimple(output_color)
    # MkdirSimple(output_concat)
    # MkdirSimple(output_display)
    # MkdirSimple(output_gray_scale)
    print("write depth image: {}".format(output_depth))
    depth = depth * scale
    cv2.imwrite(output_depth, depth.astype(np.uint16))
    return
    predict_np = depth.squeeze()
    print(predict_np.max(), " ", predict_np.min())
    predict_scale = (predict_np - np.min(predict_np))* 255 / (np.max(predict_np) - np.min(predict_np))

    predict_scale = predict_scale.astype(np.uint8)
    predict_np_int = predict_scale
    color_img = cv2.applyColorMap(predict_np_int, cv2.COLORMAP_HOT)
    limg_cv = limg  # cv2.cvtColor(np.asarray(limg), cv2.COLOR_RGB2BGR)
    concat_img_color = np.vstack([limg_cv, color_img])
    predict_np_rgb = np.stack([predict_np, predict_np, predict_np], axis=2)
    concat_img_gray = np.vstack([limg_cv, predict_np_rgb])

    # get depth
    depth_img_rgb = GetDepthImg(predict_np)
    depth_img_rgb_psl = GetDepthImgPSL(predict_np)
    concat_img_depth = np.vstack([limg_cv, depth_img_rgb])
    concat = np.hstack([np.vstack([limg_cv, color_img]), np.vstack([predict_np_rgb, depth_img_rgb])])

    cv2.imwrite(output_concat_color, concat_img_color)
    cv2.imwrite(output_concat_gray, concat_img_gray)
    cv2.imwrite(output_color, color_img)

    # cv2.imwrite(output_gray_scale, predict_np * 255 / np.max(predict_np))
    predict_np_gray_scale = predict_np * 3
    cv2.imwrite(output_gray_scale, predict_np_gray_scale)
    cv2.imwrite(output_gray, predict_np)
    cv2.imwrite(output_depth, depth_img_rgb)
    cv2.imwrite(output_depth_psl, depth_img_rgb_psl)
    cv2.imwrite(output_concat_depth, concat_img_depth)
    cv2.imwrite(output_concat, concat)


parser = argparse.ArgumentParser()

# Training data
parser.add_argument('--data_dir', default=None, required=True, type=str, help='Data directory for prediction')

parser.add_argument('--num_workers', default=0, type=int, help='Number of workers for data loading')
parser.add_argument('--img_height', default=544, type=int, help='Image height for inference')
parser.add_argument('--img_width', default=960, type=int, help='Image width for inference')

# Model
parser.add_argument('--seed', default=326, type=int, help='Random seed for reproducibility')
parser.add_argument('--output_dir', default=None, type=str,
                    help='Directory to save inference results')
parser.add_argument('--max_disp', default=192, type=int, help='Max disparity')

# AANet
parser.add_argument('--feature_type', default='aanet', type=str, help='Type of feature extractor')
parser.add_argument('--no_feature_mdconv', action='store_true', help='Whether to use mdconv for feature extraction')
parser.add_argument('--feature_pyramid', action='store_true', help='Use pyramid feature')
parser.add_argument('--feature_pyramid_network', action='store_true', help='Use FPN')
parser.add_argument('--feature_similarity', default='correlation', type=str,
                    help='Similarity measure for matching cost')
parser.add_argument('--num_downsample', default=2, type=int, help='Number of downsample layer for feature extraction')
parser.add_argument('--aggregation_type', default='adaptive', type=str, help='Type of cost aggregation')
parser.add_argument('--num_scales', default=3, type=int, help='Number of stages when using parallel aggregation')
parser.add_argument('--num_fusions', default=6, type=int, help='Number of multi-scale fusions when using parallel'
                                                               'aggragetion')
parser.add_argument('--num_stage_blocks', default=1, type=int, help='Number of deform blocks for ISA')
parser.add_argument('--num_deform_blocks', default=3, type=int, help='Number of DeformBlocks for aggregation')
parser.add_argument('--no_intermediate_supervision', action='store_true',
                    help='Whether to add intermediate supervision')
parser.add_argument('--deformable_groups', default=2, type=int, help='Number of deformable groups')
parser.add_argument('--mdconv_dilation', default=2, type=int, help='Dilation rate for deformable conv')
parser.add_argument('--refinement_type', default='stereodrnet', help='Type of refinement module')

parser.add_argument('--pretrained_aanet', default=None, type=str, help='Pretrained network')

parser.add_argument('--save_type', default='png', choices=['pfm', 'png', 'npy', 'npz'], help='Save file type')
parser.add_argument('--visualize', action='store_true', help='Visualize disparity map')

# Log
parser.add_argument('--save_suffix', default='pred', type=str, help='Suffix of save filename')
parser.add_argument('--save_dir', default='pred', type=str, help='Save prediction directory')
parser.add_argument('--scale', type=int, required=True, help='scale depth(cm)')

args = parser.parse_args()

def main():
    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test loader
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    aanet = nets.AANet(args.max_disp,
                       num_downsample=args.num_downsample,
                       feature_type=args.feature_type,
                       no_feature_mdconv=args.no_feature_mdconv,
                       feature_pyramid=args.feature_pyramid,
                       feature_pyramid_network=args.feature_pyramid_network,
                       feature_similarity=args.feature_similarity,
                       aggregation_type=args.aggregation_type,
                       num_scales=args.num_scales,
                       num_fusions=args.num_fusions,
                       num_stage_blocks=args.num_stage_blocks,
                       num_deform_blocks=args.num_deform_blocks,
                       no_intermediate_supervision=args.no_intermediate_supervision,
                       refinement_type=args.refinement_type,
                       mdconv_dilation=args.mdconv_dilation,
                       deformable_groups=args.deformable_groups).to(device)

    if os.path.exists(args.pretrained_aanet):
        print('=> Loading pretrained AANet:', args.pretrained_aanet)
        utils.load_pretrained_net(aanet, args.pretrained_aanet, no_strict=True)
    else:
        print('=> Using random initialization')

    if torch.cuda.device_count() > 1:
        print('=> Use %d GPUs' % torch.cuda.device_count())
        aanet = torch.nn.DataParallel(aanet)

    # Inference
    aanet.eval()
    print("depth (cm) * scale({})".format(args.scale))
    left_images = []
    right_images = []
    root_len = len(args.data_dir)
    if os.path.isdir(args.data_dir):
        paths = Walk(args.data_dir, ['jpg', 'png', 'jpeg'])
        print(paths)
        for image_name in paths:
            if "left" in image_name or "cam0" in image_name:
                left_images.append(image_name)
            elif "right" in image_name or "cam1" in image_name:
                right_images.append(image_name)
    else:
        print("need --images for input images' dir")
        assert 0
    for lp, rp in zip(left_images, right_images):
        if lp[root_len:][0] == '/':
            op = os.path.join(args.save_dir, lp[root_len + 1:])
        else:
            op = os.path.join(args.save_dir, lp[root_len:])
        # MkdirSimple(op)

        left = read_img(lp)
        left_copy = left.copy()
        right = read_img(rp)
        sample = {'left': left,
                  'right': right}
        sample = test_transform(sample)  # to tensor and normalize

        left = sample['left'].to(device)  # [3, H, W]
        left = left.unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device)
        right = right.unsqueeze(0)

        # Pad
        ori_height, ori_width = left.size()[2:]

        # Automatic
        factor = 48 if args.refinement_type != 'hourglass' else 96
        args.img_height = math.ceil(ori_height / factor) * factor
        args.img_width = math.ceil(ori_width / factor) * factor

        if ori_height < args.img_height or ori_width < args.img_width:
            top_pad = args.img_height - ori_height
            right_pad = args.img_width - ori_width

            # Pad size: (left_pad, right_pad, top_pad, bottom_pad)
            left = F.pad(left, (0, right_pad, top_pad, 0))
            right = F.pad(right, (0, right_pad, top_pad, 0))

        with torch.no_grad():
            pred_disp = aanet(left, right)[-1]  # [B, H, W]
        if pred_disp.size(-1) < left.size(-1):
            pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
            pred_disp = F.interpolate(pred_disp, (left.size(-2), left.size(-1)),
                                      mode='bilinear') * (left.size(-1) / pred_disp.size(-1))
            pred_disp = pred_disp.squeeze(1)  # [B, H, W]

        # Crop
        if ori_height < args.img_height or ori_width < args.img_width:
            if right_pad != 0:
                pred_disp = pred_disp[:, top_pad:, :-right_pad]
            else:
                pred_disp = pred_disp[:, top_pad:]

        disp = pred_disp[0].detach().cpu().numpy()  # [H, W]
        op = op.replace(".jpg", ".png")

        WriteDepthScale(3423/disp,left_copy,args.save_dir, op, args.scale)

if __name__ == '__main__':
    main()
