import torch
from PIL import Image
import os
import pdb
import numpy as np
import cv2
from data.mytransforms import find_start_pos


def loader_func(path):
    return Image.open(path)  # 图片读取


# 设置测试集
class LaneTestDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform=None):
        super(LaneTestDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        with open(list_path, 'r') as f:
            self.list = f.readlines()  # 读取每一行的内容
        self.list = [l[1:] if l[0] == '/' else l for l in self.list]  # 排除不正确的路径前缀 '/' of CULane

    def __getitem__(self, index):
        name = self.list[index].split()[0]  # 图片的名字
        img_path = os.path.join(self.path, name)  # 图片的路径
        img = loader_func(img_path)  # 读取图像

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, name

    def __len__(self):
        return len(self.list)  # 总共的图片数量


class LaneClsDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform=None, target_transform=None, simu_transform=None, griding_num=50,
                 load_name=False,
                 row_anchor=None, use_aux=False, segment_transform=None, num_lanes=4):
        super(LaneClsDataset, self).__init__()
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.segment_transform = segment_transform
        self.simu_transform = simu_transform
        self.path = path
        self.griding_num = griding_num
        self.load_name = load_name
        self.use_aux = use_aux
        self.num_lanes = num_lanes

        with open(list_path, 'r') as f:
            self.list = f.readlines()       # 读取文件中的每一行

        self.row_anchor = row_anchor
        self.row_anchor.sort()

    def __getitem__(self, index):
        # 取出数据路径和标签路径
        l = self.list[index]  # 对每一行的路径进行处理
        l_info = l.split()      # 将图像和标签路径分开
        img_name, label_name = l_info[0], l_info[1]  # 获取图像，标签路径
        # 处理路径，去掉首部的'/'
        if img_name[0] == '/':
            img_name = img_name[1:]
            label_name = label_name[1:]

        # 读进标签
        label_path = os.path.join(self.path, label_name)
        label = loader_func(label_path)

        # 读进图像
        img_path = os.path.join(self.path, img_name)
        img = loader_func(img_path)

        # 数据增强，轻微的旋转和平移，主要以平移为主
        if self.simu_transform is not None:
            img, label = self.simu_transform(img, label)

        lane_pts = self._get_index(label)  # 处理标签，获取行锚点处车道的坐标

        # 构建网络，将标签映射到网络中
        w, h = img.size  # 图像的宽和长
        cls_label = self._grid_pts(lane_pts, self.griding_num, w)
        # 制作分类标签的坐标
        if self.use_aux:
            assert self.segment_transform is not None
            seg_label = self.segment_transform(label)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.use_aux:
            return img, cls_label, seg_label

        if self.load_name:
            return img, cls_label, img_name

        return img, cls_label

    def __len__(self):
        return len(self.list)   # 数据集的长度

    def _grid_pts(self, pts, num_cols, w):
        # 将每条车道线的坐标映射到200*200的网络中
        num_lane, n, n2 = pts.shape     # 4 18 2
        col_sample = np.linspace(0, w - 1, num_cols)    # 在0到w-1之间等间距取200个数

        assert n2 == 2
        to_pts = np.zeros((n, num_lane))    # 18*4
        for i in range(num_lane):
            pti = pts[i, :, 1]
            to_pts[:, i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
        return to_pts.astype(int)

    def _get_index(self, label):
        w, h = label.size  # 标签的宽和长
        # 将先验值缩放至图片长度等比大小的位置上
        if h != 288:
            scale_f = lambda x: int((x * 1.0 / 288) * h)
            sample_tmp = list(map(scale_f, self.row_anchor))  # 18个先验值，anchor为18

        # 4*18*2的矩阵，4代表4条车道线，18代表18行，2代表车道线的位置的行和列
        all_idx = np.zeros((self.num_lanes, len(sample_tmp), 2))  # 将标签存为4*18*2
        for i, r in enumerate(sample_tmp):  # 遍历每一个先验值
            label_r = np.asarray(label)[int(round(r))]
            for lane_idx in range(1, self.num_lanes + 1):  # 遍历每一条车道线
                pos = np.where(label_r == lane_idx)[0]      # 找到车道线位置
                if len(pos) == 0:       # 不存在车道线
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = -1
                    continue
                pos = np.mean(pos)  # 将位置平均，因为获取到的位置是一个范围，有宽度
                all_idx[lane_idx - 1, i, 0] = r
                all_idx[lane_idx - 1, i, 1] = pos

        # 对每条车道线进行线性拟合，以进行延伸
        all_idx_cp = all_idx.copy()  # 标签已经做好
        for i in range(self.num_lanes):
            if np.all(all_idx_cp[i, :, 1] == -1):  # 如果没有车道线
                continue

            valid = all_idx_cp[i, :, 1] != -1  # 获取所有有效车道点的索引
            valid_idx = all_idx_cp[i, valid, :]  # 获取所有有效车道点
            if valid_idx[-1, 0] == all_idx_cp[0, -1, 0]:
                # 如果最后一个有效车道点的y坐标已经是所有行的最后一个y坐标
                # 这意味着该车道已到达图像的底部边界，所以跳过
                continue
            if len(valid_idx) < 6:  # 如果车道太短而无法延伸
                continue

            valid_idx_half = valid_idx[len(valid_idx) // 2:, :]     # 找到车道线中间的点
            p = np.polyfit(valid_idx_half[:, 0], valid_idx_half[:, 1], deg=1)  # 线性拟合，延长这条线到边界
            start_line = valid_idx_half[-1, 0]
            pos = find_start_pos(all_idx_cp[i, :, 0], start_line) + 1   # 找到起始的位置

            fitted = np.polyval(p, all_idx_cp[i, pos:, 0])  # 得到拟合值
            fitted = np.array([-1 if y < 0 or y > w - 1 else y for y in fitted])  # 对拟合值进行判断

            assert np.all(all_idx_cp[i, pos:, 1] == -1)     # 判断是否都为-1
            all_idx_cp[i, pos:, 1] = fitted  # 下一个位置等于拟合值
        if -1 in all_idx[:, :, 0]:
            pdb.set_trace()
        return all_idx_cp
