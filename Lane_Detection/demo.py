import cv2
import os

import numpy as np
import scipy.special
import torch
import torchvision.transforms as transforms
import tqdm

from data.constant import culane_row_anchor, tusimple_row_anchor
from data.dataset import LaneTestDataset
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # 让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
    args, cfg = merge_config()  # 获取配置项
    dist_print('start testing...')
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(pretrained=False, backbone=cfg.backbone, cls_dim=(cfg.griding_num + 1, cls_num_per_lane, 4),
                     use_aux=False).cpu()  # 我们在测试中不需要辅助分割

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if cfg.dataset == 'CULane':
        splits = ['test0_normal.txt', 'test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt', 'test4_noline.txt',
                  'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt']
        # splits = ['test0_normal.txt']
        datasets = [LaneTestDataset(cfg.data_root, os.path.join(cfg.data_root, 'list/test_split/' + split),
                                    img_transform=img_transforms) for split in splits]
        img_w, img_h = 1640, 590
        row_anchor = culane_row_anchor
    elif cfg.dataset == 'Tusimple':
        splits = ['test.txt']
        datasets = [LaneTestDataset(cfg.data_root, os.path.join(cfg.data_root, split), img_transform=img_transforms) for
                    split in splits]
        img_w, img_h = 1280, 720
        row_anchor = tusimple_row_anchor
    else:
        raise NotImplementedError
    for split, dataset in zip(splits, datasets):
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        print(split[:-3] + 'avi')
        vout = cv2.VideoWriter(split[:-3] + 'avi', fourcc, 5.0, (img_w, img_h))    # 打开视频输出
        for i, data in enumerate(tqdm.tqdm(loader)):
            imgs, names = data
            imgs = imgs.cpu()
            with torch.no_grad():
                out = net(imgs)

            col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
            col_sample_w = col_sample[1] - col_sample[0]

            out_j = out[0].data.cpu().numpy()
            out_j = out_j[:, ::-1, :]
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
            idx = np.arange(cfg.griding_num) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == cfg.griding_num] = 0
            out_j = loc

            # import pdb; pdb.set_trace()
            vis = cv2.imread(os.path.join(cfg.data_root, names[0]))
            for i in range(out_j.shape[1]):     # 4条车道线
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):     # 18条行锚框
                        if out_j[k, i] > 0:     # 有车道线
                            # 获取真正的车道线位置
                            ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                                   int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1)
                            cv2.circle(vis, ppp, 5, (0, 255, 0), -1)    # 标点
            vout.write(vis)     # 保存视频

        vout.release()  # 关闭视频输出
