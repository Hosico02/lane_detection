import torch, os
import numpy as np

import torchvision.transforms as transforms
import data.mytransforms as mytransforms
from data.constant import tusimple_row_anchor, culane_row_anchor
from data.dataset import LaneClsDataset, LaneTestDataset


def get_train_loader(batch_size, data_root, griding_num, dataset, use_aux, distributed, num_lanes):
    # 创建一个数据转换序列
    target_transform = transforms.Compose([
        mytransforms.FreeScaleMask((288, 800)),     # 调整图像大小
        mytransforms.MaskToTensor(),    # 转换为张量
    ])
    segment_transform = transforms.Compose([
        mytransforms.FreeScaleMask((36, 100)),
        mytransforms.MaskToTensor(),
    ])
    img_transform = transforms.Compose([
        transforms.Resize((288, 800)),  # 调整图像大小
        transforms.ToTensor(),      # 转换为张量
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),     # 标准化
    ])
    # 为了数据增强
    simu_transform = mytransforms.Compose2([
        mytransforms.RandomRotate(6),           # 随机旋转
        mytransforms.RandomUDoffsetLABEL(100),  # 上下随机平移
        mytransforms.RandomLROffsetLABEL(200)   # 左右随机平移
    ])
    if dataset == 'CULane':
        train_dataset = LaneClsDataset(data_root,
                                       os.path.join(data_root, 'list/train_gt.txt'),
                                       img_transform=img_transform, target_transform=target_transform,
                                       simu_transform=simu_transform,
                                       segment_transform=segment_transform,
                                       row_anchor=culane_row_anchor,
                                       griding_num=griding_num, use_aux=use_aux, num_lanes=num_lanes)
        cls_num_per_lane = 18

    elif dataset == 'Tusimple':
        train_dataset = LaneClsDataset(data_root,
                                       os.path.join(data_root, 'train_gt.txt'),
                                       img_transform=img_transform, target_transform=target_transform,
                                       simu_transform=simu_transform,
                                       griding_num=griding_num,
                                       row_anchor=tusimple_row_anchor,
                                       segment_transform=segment_transform, use_aux=use_aux, num_lanes=num_lanes)
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)

    return train_loader, cls_num_per_lane


def get_test_loader(batch_size, data_root, dataset, distributed):
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if dataset == 'CULane':
        test_dataset = LaneTestDataset(data_root, os.path.join(data_root, 'list/test.txt'),
                                       img_transform=img_transforms)
        cls_num_per_lane = 18
    elif dataset == 'Tusimple':
        test_dataset = LaneTestDataset(data_root, os.path.join(data_root, 'test.txt'), img_transform=img_transforms)
        cls_num_per_lane = 56

    if distributed:
        sampler = SeqDistributedSampler(test_dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(test_dataset)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    return loader


class SeqDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    """
    将DistributedSampler的行为更改为顺序分布式采样
    顺序采样有助于多线程测试的稳定性，这需要多线程IO文件
    如果不按顺序采样，线程上的IO文件可能会干扰其他线程
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas, rank, shuffle)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)   # 随机数种子为epoch
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # 添加额外的样本使其可被整除
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        num_per_rank = int(self.total_size // self.num_replicas)

        # 顺序抽样
        indices = indices[num_per_rank * self.rank: num_per_rank * (self.rank + 1)]

        assert len(indices) == self.num_samples

        return iter(indices)
