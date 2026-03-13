import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 交叉熵损失函数
class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cpu()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        # 交叉熵损失函数
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


# 分类损失
class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)   # 分类器

    def forward(self, logits, labels):  # softmax(x)+log(x)+nn.NLLLoss=====>nn.CrossEntropyLoss
        scores = F.softmax(logits, dim=1)  # 将预测值做归一化
        factor = torch.pow(1. - scores, self.gamma)  # 1-预测值的gamma次幂，当前样本的权重项
        log_score = F.log_softmax(logits, dim=1)  # 先取对然后再归一化
        log_score = factor * log_score      # 权重
        loss = self.nll(log_score, labels)
        return loss


# 相似损失
class ParsingRelationLoss(nn.Module):
    def __init__(self):
        super(ParsingRelationLoss, self).__init__()

    def forward(self, logits):
        n, c, h, w = logits.shape
        loss_all = []
        for i in range(0, h - 1):
            loss_all.append(logits[:, :, i, :] - logits[:, :, i + 1, :])
        loss = torch.cat(loss_all)  # 将多个张量拼接在一起
        # 使用smooth_l1_loss来计算损失值
        return torch.nn.functional.smooth_l1_loss(loss, torch.zeros_like(loss))


# 结构损失
class ParsingRelationDis(nn.Module):
    def __init__(self):
        super(ParsingRelationDis, self).__init__()
        self.l1 = torch.nn.L1Loss()     # 累加求平均

    def forward(self, x):
        # 求实际预测的期望
        n, dim, num_rows, num_cols = x.shape
        x = torch.nn.functional.softmax(x[:, :dim - 1, :, :], dim=1)    # 每个位置预测的概率
        embedding = torch.Tensor(np.arange(dim - 1)).float().to(x.device).view(1, -1, 1, 1)
        pos = torch.sum(x * embedding, dim=1)   # 期望
        diff_list1 = []
        # 从位置角度，同一条车道线位置接近
        for i in range(0, num_rows // 2):
            diff_list1.append(pos[:, i, :] - pos[:, i + 1, :])
        loss = 0
        for i in range(len(diff_list1) - 1):
            loss += self.l1(diff_list1[i], diff_list1[i + 1])
        loss /= len(diff_list1) - 1
        return loss
