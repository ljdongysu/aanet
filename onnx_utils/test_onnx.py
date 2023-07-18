import sys, os
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))
import onnxruntime
# import torch
from onnxmodel import ONNXModel
import cv2
import numpy as np
# import torchvision
from utils.file_io import read_img
from dataloader import transforms
from predict import IMAGENET_MEAN,IMAGENET_STD

model = ONNXModel("../aanetKITTI2015.onnx")

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lp = "/home/indemind/Code/CLionprojects/18R_workspace/npu/aml_npu_sdk_6.4.8/acuity-toolkit/demo/data/image_1248_384/left/0000000000.png"
rp = "/home/indemind/Code/CLionprojects/18R_workspace/npu/aml_npu_sdk_6.4.8/acuity-toolkit/demo/data/image_1248_384/right/0000000000.png"
left = read_img(lp)
right = read_img(rp)

left = np.resize(left, (384, 1248, 3))
right = np.resize(right, (384, 1248, 3))

sample = {'left': left,
          'right': right}

test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

sample = test_transform(sample)  # to tensor and normalize

left = sample['left']#.to(device)  # [3, H, W]
left = left.unsqueeze(0).numpy()  # [1, 3, H, W]
right = sample['right']#.to(device)
right = right.unsqueeze(0).numpy()

print(left.shape)
# output = model.forward({"L":left, "R":right})
output = model.forward2((left, right))
print("finsih")

# disp = output[0][:, 0:1]
# disp = torch.from_numpy(disp)
# disp = np.clip(disp / 192 * 255, 0, 255).long()
#
# disp = apply_colormap(disp)
# output1 = [torch.from_numpy(left), disp]
# print(left.shape, disp.shape)
# output1 = torch.cat(output1, dim=0)
# torchvision.utils.save_image(output1, "disp_from_onnx.png", nrow=1)