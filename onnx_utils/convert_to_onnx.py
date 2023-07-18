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

args = parser.parse_args()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def main():
    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    aanet.eval()
    height = 384
    width = 1248
    input_names = ['L', 'R']
    output_names = ['disp']
    input_L = torch.randn(1, 3, height, width, device='cuda:0')
    input_R = torch.randn(1, 3, height, width, device='cuda:0')
    torch.onnx.export(
        aanet,
        (input_L,input_R),
        "./aanetKITTI2015.onnx",
        verbose=True,
        opset_version=12,
        input_names=input_names,
        output_names=output_names)
    print("Finish!")

if __name__ == '__main__':
    main()