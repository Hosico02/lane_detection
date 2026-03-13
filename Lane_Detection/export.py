import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor
from PIL import Image

# 将模型导出可以与LibTorch一起使用的TorchScrip格式
torch.backends.cudnn.benchmark = True

# CULane数据, 如果使用TuSimple需要改变
cls_num_per_lane = 18
griding_num = 200
backbone = 18

net = parsingNet(pretrained=False, backbone='18', cls_dim=(griding_num + 1, cls_num_per_lane, 4),
                 use_aux=False)

# 更改存储模型的test_model.
test_model = 'culane_18.pth'

state_dict = torch.load(test_model, map_location='cpu')['model']  # CPU
# state_dict = torch.load(test_model, map_location='cuda')['model'] # CUDA
compatible_state_dict = {}
for k, v in state_dict.items():
    if 'module.' in k:
        compatible_state_dict[k[7:]] = v
    else:
        compatible_state_dict[k] = v

net.load_state_dict(compatible_state_dict, strict=False)
net.eval()

# Test Input Image
img = torch.zeros(1, 3, 288, 800)   # image size(1,3,320,192) iDetection
y = net(img)  # dry run

ts = torch.jit.trace(net, img)

ts.save('车道线检测.torchscript-cpu.pt')  # CPU
# ts.save('车道线检测.torchscript-cuda.pt') # CUDA
