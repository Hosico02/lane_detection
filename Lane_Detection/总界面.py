import datetime
import os
import sys
import time
import cv2
import numpy as np
import scipy
import torch
import torchvision.transforms as transforms
from PIL import Image
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from data.constant import culane_row_anchor
from data.dataloader import get_train_loader
from model.model import parsingNet
from utils.dist_utils import dist_tqdm, DistSummaryWriter
from utils.factory import MultiStepLR, CosineAnnealingLR
from utils.loss import SoftmaxFocalLoss, ParsingRelationLoss, ParsingRelationDis
from utils.metrics import reset_metrics, update_metrics, MultiLabelAcc, AccTopk, Metric_mIoU
import webbrowser
backbone = '18'


def loader_func(path):
    return Image.open(path)


def calc_loss(loss_dict, results, logger, global_step):
    loss = 0
    for i in range(len(loss_dict['name'])):

        data_src = loss_dict['data_src'][i]

        datas = [results[src] for src in data_src]

        loss_cur = loss_dict['op'][i](*datas)

        if global_step % 20 == 0:
            logger.add_scalar('loss/' + loss_dict['name'][i], loss_cur, global_step)

        loss += loss_cur * loss_dict['weight'][i]
    return loss


def resolve_val_data(results, use_aux):
    results['cls_out'] = torch.argmax(results['cls_out'], dim=1)
    if use_aux:
        results['seg_out'] = torch.argmax(results['seg_out'], dim=1)
    return results


def inference(net, data_label, use_aux):
    if use_aux:
        img, cls_label, seg_label = data_label
        img, cls_label, seg_label = img.cpu(), cls_label.long().cpu(), seg_label.long().cpu()
        cls_out, seg_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label, 'seg_out': seg_out, 'seg_label': seg_label}
    else:
        img, cls_label = data_label
        img, cls_label = img.cpu(), cls_label.long().cpu()
        cls_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label}


def use_model(frame, label_1, label_2, net, transforms, griding_num):
    frame_1 = cv2.resize(frame, (1640, 590))
    frame = cv2.resize(frame, (520, 400))
    # 视频色彩转换回RGB，OpenCV images as BGR
    frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 变成QImage形式
    qImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
    # 往显示视频的Label里 显示QImage
    label_1.setPixmap(QtGui.QPixmap.fromImage(qImage))
    # 后处理
    img = frame_1
    frame_1 = Image.fromarray(np.uint8(frame_1))  # 转换为图像张量
    frame_1 = transforms(frame_1)  # 图像预处理
    frame_1 = frame_1.unsqueeze(0)  #
    frame_1 = frame_1.cpu()  # 使用CPU
    with torch.no_grad():  # 测试代码不计算梯度
        out = net(frame_1)  # 模型预测 输出张量：[1,201,18,4]

    col_sample = np.linspace(0, 800 - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]
    out_j = out[0].data.cpu().numpy()  # 数据类型转换成numpy [201,18,4]
    out_j = out_j[:, ::-1, :]  # 将第二维度倒着取[201,18,4]
    # [200,18,4]softmax计算（概率映射到0-1之间且沿着维度0概率总和=1）
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = np.arange(griding_num) + 1  # 产生 1-200
    idx = idx.reshape(-1, 1, 1)  # [200,1,1]
    loc = np.sum(prob * idx, axis=0)  # [18,4]
    out_j = np.argmax(out_j, axis=0)  # 返回最大值的索引
    loc[out_j == griding_num] = 0  # 若最大值的索引=griding_num，归零
    out_j = loc  # [18,4]
    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    ppp = (int(out_j[k, i] * col_sample_w * img.shape[1] / 800) - 1, int(img.shape[0] - k * 20) - 1)
                    cv2.circle(img, ppp, 5, (0, 255, 0), -1)

    img = cv2.resize(img, (520, 400))
    qImage = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
    label_2.setPixmap(QPixmap.fromImage(qImage))
    return img


class MainWindow(QTabWidget):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.selected_imgsPath = None
        self.current_image_index = None
        self.setupUI()
        # 按钮功能添加
        self.camBtn.clicked.connect(self.startCamera)
        self.stopBtn.clicked.connect(self.stop)
        self.videoBtn.clicked.connect(self.Video)
        self.Btn_1.clicked.connect(self.Tensorboard)
        self.Btn_2.clicked.connect(self.Train)
        self.BBox_1.clicked.connect(self.Photo)
        self.BBox_3.clicked.connect(self.showPrevImage)
        self.BBox_4.clicked.connect(self.showNextImage)
        self.BBox_2.clicked.connect(self.save_phonto)
        # 定义定时器，用于控制显示视频的帧率
        self.timer_camera = QTimer(self)
        self.time_1 = QTimer(self)
        self.time_2 = QTimer(self)
        # 定时到了，回调 self.show_camera
        self.timer_camera.timeout.connect(self.show_camera)
        self.time_1.timeout.connect(self.start_tensorboard)
        self.time_2.timeout.connect(self.stop_tensorboard)
        # 图片和视频
        self.SAVE = None
        self.img = None
        self.cap = None
        # 图片视频路径
        self.road = ''
        # 模型训练初始变量
        self.num_lanes = 4
        self.log_path = './log'
        self.learning_rate = 4e-4
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.steps = [25, 38]
        self.gamma = 0.1
        self.warmup = 'linear'
        self.warmup_iters = 695  # 预热轮次
        self.use_aux = True  # 使用数据增强
        self.sim_loss_w = 0.0
        self.shp_loss_w = 0.0
        self.dataset = 'CULane'
        self.data_root = './CULane'
        self.optimizer = 'SGD'
        self.scheduler = 'multi'
        self.epoch = 10
        self.batch_size = 32
        # 指标参数
        self.train_epoch = 0
        self.Loss = None
        self.t1 = None
        self.Miou = None
        # 初始化
        torch.backends.cudnn.benchmark = True
        self.test_model = r'/Users/mack/Desktop/work/车道线检测/模型/culane_18.pth'
        self.griding_num = 200
        self.cls_num_per_lane = 18
        self.net = parsingNet(pretrained=False, backbone=backbone,
                              cls_dim=(self.griding_num + 1, self.cls_num_per_lane, 4),
                              use_aux=False).cpu()
        state_dict = torch.load(self.test_model, map_location='cpu')['model']
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v
        self.net.load_state_dict(compatible_state_dict, strict=False)
        self.net.eval()
        self.img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.img_w, self.img_h = 1280, 720
        self.row_anchor = culane_row_anchor

    def setupUI(self):
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.addTab(self.tab1, "Tab 1")
        self.addTab(self.tab2, "Tab 2")
        self.addTab(self.tab3, "Tab 3")
        self.tab1UI()
        self.tab2UI()
        self.tab3UI()
        self.setWindowTitle("车道线检测")  # 标题
        self.resize(1200, 700)  # 界面大小

    def tab1UI(self):
        layout = QFormLayout()
        mainLayout = QtWidgets.QVBoxLayout()
        # 上半部分
        topLayout = QtWidgets.QHBoxLayout()
        self.label_ori_video = QtWidgets.QLabel(self)
        self.label_treated = QtWidgets.QLabel(self)
        # QT框的长宽
        self.label_ori_video.setFixedSize(520, 400)
        self.label_treated.setFixedSize(520, 400)
        # 设置边框的格式
        self.label_ori_video.setStyleSheet('border:1px solid #D7E2F9;')
        self.label_treated.setStyleSheet('border:1px solid #D7E2F9;')
        topLayout.addWidget(self.label_ori_video)
        topLayout.addWidget(self.label_treated)
        mainLayout.addLayout(topLayout)
        # 下半部分
        bottomLayout = QtWidgets.QHBoxLayout()
        leftLayout = QtWidgets.QVBoxLayout()
        self.selectBox_1 = QtWidgets.QComboBox()
        self.videoBtn = QtWidgets.QPushButton('🎞️视频文件')
        self.camBtn = QtWidgets.QPushButton('📹摄像头')
        self.saveBtn = QtWidgets.QPushButton('📃保存结果视频')
        self.stopBtn = QtWidgets.QPushButton('🛑停止或清空文本框')
        self.selectBox_1.addItem('Ours-CULane')
        self.selectBox_1.addItem('Ours-Tusimple')
        self.selectBox_1.addItem('SCNN-CULane')
        self.selectBox_1.addItem('SCNN-Tusimple')
        self.selectBox_1.addItem('SAD-CULane')
        self.selectBox_1.addItem('SAD-Tusimple')
        self.selectBox_1.addItem('ResNet-CULane')
        self.selectBox_1.addItem('ResNet-Tusimple')
        leftLayout.addWidget(QtWidgets.QLabel('模型选择：'))
        leftLayout.addWidget(self.selectBox_1)
        leftLayout.addWidget(QtWidgets.QLabel('按钮：'))
        leftLayout.addWidget(self.videoBtn)
        leftLayout.addWidget(self.camBtn)
        leftLayout.addWidget(self.saveBtn)
        leftLayout.addWidget(self.stopBtn)
        bottomLayout.addLayout(leftLayout)
        # 界面右边部分：显示框和输出框
        rightLayout = QtWidgets.QHBoxLayout()
        self.textLog = QtWidgets.QTextBrowser()
        rightLayout.addWidget(self.textLog)
        bottomLayout.addLayout(rightLayout)
        mainLayout.addLayout(bottomLayout)
        layout.addRow(mainLayout)
        self.setTabText(0, "视频演示")  # 标题
        self.tab1.setLayout(layout)  # 加载

    def tab2UI(self):
        layout = QFormLayout()
        mainLayout = QtWidgets.QHBoxLayout()
        leftLayout = QtWidgets.QVBoxLayout()
        self.SBox_1 = QComboBox()
        self.SBox_1.addItem('Ours-CULane')
        self.SBox_1.addItem('Ours-Tusimple')
        self.SBox_1.addItem('SCNN-CULane')
        self.SBox_1.addItem('SCNN-Tusimple')
        self.SBox_1.addItem('SAD-CULane')
        self.SBox_1.addItem('SAD-Tusimple')
        self.SBox_1.addItem('ResNet-CULane')
        self.SBox_1.addItem('ResNet-Tusimple')
        self.BBox_1 = QtWidgets.QPushButton('选择图片')
        self.BBox_2 = QtWidgets.QPushButton('保存图片')
        self.BBox_3 = QtWidgets.QPushButton('上一张图片')
        self.BBox_4 = QtWidgets.QPushButton('下一张图片')
        q1 = QLabel('数据集选择：')
        q1.setFixedSize(100, 50)
        q2 = QLabel('按钮：')
        q2.setFixedSize(100, 50)
        leftLayout.addWidget(q1)
        leftLayout.addWidget(self.SBox_1)
        leftLayout.addWidget(q2)
        leftLayout.addWidget(self.BBox_1)
        leftLayout.addWidget(self.BBox_2)
        leftLayout.addWidget(self.BBox_3)
        leftLayout.addWidget(self.BBox_4)
        rightLayout = QtWidgets.QVBoxLayout()
        labelLayout = QtWidgets.QHBoxLayout()
        self.ori = QtWidgets.QLabel(self)
        self.tri = QtWidgets.QLabel(self)
        # QT框的长宽
        self.ori.setFixedSize(520, 400)
        self.tri.setFixedSize(520, 400)
        # 设置边框的格式
        self.ori.setStyleSheet('border:1px solid #D7E2F9;')
        self.tri.setStyleSheet('border:1px solid #D7E2F9;')
        labelLayout.addWidget(self.ori)
        labelLayout.addWidget(self.tri)
        rightLayout.addLayout(labelLayout)
        self.textLog_1 = QtWidgets.QTextBrowser()
        rightLayout.addWidget(self.textLog_1)
        mainLayout.addLayout(leftLayout)
        mainLayout.addLayout(rightLayout)
        layout.addRow(mainLayout)
        self.setTabText(1, "图片演示")  # 标题
        self.tab2.setLayout(layout)

    def tab3UI(self):
        layout = QFormLayout()
        mainLayout = QtWidgets.QVBoxLayout()
        mainLayout.setAlignment(Qt.AlignVCenter)
        # 上半部分
        topLayout = QtWidgets.QHBoxLayout()
        leftLayout = QtWidgets.QVBoxLayout()
        # leftLayout.setAlignment(Qt.AlignVCenter)
        rightLayout = QtWidgets.QVBoxLayout()
        # rightLayout.setAlignment(Qt.AlignVCenter)
        self.selectB_1 = QtWidgets.QComboBox()
        self.selectB_1.addItem("CULane")
        self.selectB_1.addItem("Tusimple")
        self.selectB_1.setFixedWidth(200)
        self.selectB_2 = QtWidgets.QComboBox()
        self.selectB_2.addItem('多步长衰减策略')
        self.selectB_2.addItem('余弦退火衰减策略')
        self.selectB_2.setFixedWidth(200)
        self.selectB_3 = QtWidgets.QComboBox()
        self.selectB_3.addItem('200')
        self.selectB_3.addItem('50')
        self.selectB_3.addItem('100')
        self.selectB_3.addItem('150')
        self.selectB_3.setFixedWidth(200)
        self.selectB_4 = QtWidgets.QComboBox()
        self.selectB_4.addItem('SGD（随机梯度下降）')
        self.selectB_4.addItem('Adam（应性矩估计）')
        self.Text_1 = QtWidgets.QLineEdit()
        self.Text_2 = QtWidgets.QLineEdit()
        self.Text_1.setFixedWidth(200)
        self.Text_1.setValidator(QIntValidator())
        self.Text_2.setFixedWidth(200)
        self.Text_2.setValidator(QIntValidator())
        self.Text_1.setText('10')
        self.Text_2.setText('32')
        leftLayout.addWidget(QtWidgets.QLabel('\n\n\n\n\n\n\n\n数据集选择：'))
        leftLayout.addWidget(self.selectB_1)
        leftLayout.addWidget(QtWidgets.QLabel('epoch：'))
        leftLayout.addWidget(self.Text_1)
        leftLayout.addWidget(QtWidgets.QLabel('batch_size(为2的倍数)：'))
        leftLayout.addWidget(self.Text_2)
        leftLayout.addWidget(QtWidgets.QLabel('学习率衰减策略：'))
        leftLayout.addWidget(self.selectB_2)
        leftLayout.addWidget(QtWidgets.QLabel('网格单元数量：'))
        leftLayout.addWidget(self.selectB_3)
        leftLayout.addWidget(QtWidgets.QLabel('优化器：'))
        leftLayout.addWidget(self.selectB_4)
        self.Btn_1 = QtWidgets.QPushButton('可视化')
        self.Btn_2 = QtWidgets.QPushButton("训练")
        self.Btn_1.setFixedSize(200, 100)
        self.Btn_2.setFixedSize(200, 100)
        rightLayout.addWidget(self.Btn_1)
        rightLayout.addWidget(self.Btn_2)
        topLayout.addLayout(leftLayout, 1)
        topLayout.addLayout(rightLayout, 1)
        mainLayout.addLayout(topLayout)
        # 下半部分
        bottomLayout = QtWidgets.QVBoxLayout()
        self.bar = QtWidgets.QProgressBar(self)
        self.bar.setRange(0, 100)  # 设置进度条的范围（从0到100）
        self.bar.move(50, 50)
        self.bar.setFixedHeight(50)
        self.bar.setStyleSheet(
            "QProgressBar { border: 2px solid grey; border-radius: 5px; color: rgb(20,20,20);  background-color: #FFFFFF; text-align: center;}QProgressBar::chunk {background-color: rgb(100,200,200); border-radius: 10px; margin: 0.1px;  width: 1px;}")
        font = QFont()
        label = QLabel('\n\n进度条')
        label.setAlignment(Qt.AlignCenter)  # 设置标签文本居中
        label_1 = QLabel('\n\n\n\n\n\n\n')
        self.train_epoch = 0
        self.Miou = 0.0
        self.t1 = 0.0
        self.Loss = 0.0
        self.train_label = QLabel(
            'epoch：' + str(self.train_epoch) + '\t\t\t\tmiou：' + str(self.Miou) + '\t\t\t\ttop1：' + str(
                self.t1) + '\t\t\t\tloss：' + str(self.Loss))
        self.train_label.setAlignment(Qt.AlignCenter)  # 设置标签文本居中
        self.bar.setValue(39)
        bottomLayout.addWidget(label_1)
        bottomLayout.addWidget(self.train_label)
        bottomLayout.addWidget(label)
        bottomLayout.addWidget(self.bar)
        mainLayout.addLayout(bottomLayout)
        layout.addRow(mainLayout)
        self.setTabText(2, "模型训练")  # 标题
        self.tab3.setLayout(layout)

    def startCamera(self):
        # 在 windows上指定使用 cv2.CAP_DSHOW 会让打开摄像头快很多，
        # 在 Linux/Mac上 指定 V4L, FFMPEG 或者 GSTREAMER
        self.textLog.clear()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.textLog.setText('摄像头不能打开')
            return

        if not self.timer_camera.isActive():  # 若定时器未启动
            self.timer_camera.start(50)

    def stop(self):
        self.timer_camera.stop()  # 关闭定时器
        self.cap.release()  # 释放视频流
        self.label_ori_video.clear()  # 清空视频显示区域
        self.label_treated.clear()  # 清空视频显示区域
        self.textLog.clear()  # 清除文本框内容

    def Video(self):
        self.textLog.clear()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择视频文件', '', 'Video Files (*.mp4 *.avi)')
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.road = file_path
            self.timer_camera.start(30)

    def show_camera(self):
        ret, frame = self.cap.read()  # 从视频流中读取
        if not ret:
            return
        if self.selectBox_1.currentText() == 'Ours-CULane':
            self.test_model = r'/Users/mack/Desktop/work/车道线检测/模型/culane_18.pth'
        if self.selectBox_1.currentText() == 'Ours-Tusimple':
            self.test_model = r'/Users/mack/Desktop/work/车道线检测/模型/tusimple_18.pth'
        if self.selectBox_1.currentText() == 'SCNN-CULane':
            self.test_model = r'/Users/mack/Desktop/work/车道线检测/模型/culane_SCNN.pth'
        if self.selectBox_1.currentText() == 'SCNN-Tusimple':
            self.test_model = r'/Users/mack/Desktop/work/车道线检测/模型/tusimple_SCNN.pth'
        if self.selectBox_1.currentText() == 'SAD-CULane':
            self.test_model = r'/Users/mack/Desktop/work/车道线检测/模型/culane_SAD.pth'
        if self.selectBox_1.currentText() == 'SAD-Tusimple':
            self.test_model = r'/Users/mack/Desktop/work/车道线检测/模型/tusimple_SAD.pth'
        if self.selectBox_1.currentText() == 'ResNet-CULane':
            self.test_model = r'/Users/mack/Desktop/work/车道线检测/模型/culane_ResNet.pth'
        if self.selectBox_1.currentText() == 'ResNet-Tusimple':
            self.test_model = r'/Users/mack/Desktop/work/车道线检测/模型/tusimple_ResNet.pth'
        nn = self.selectBox_1.currentText()
        SR = '视频路径：' + self.road + '\n模型来源：' + nn + '\t单元网格数量：' + str(self.griding_num)
        SR += '\n模型路径：' + self.test_model
        self.textLog.setText(SR)
        use_model(frame, self.label_ori_video, self.label_treated, self.net, self.img_transforms, self.griding_num)

    def closeEvent(self, event):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        event.accept()

    def frameAnalyzeThreadFunc(self):
        while True:
            if not self.frameToAnalyze:
                time.sleep(0.01)
                continue

            frame = self.frameToAnalyze.pop(0)
            results = self.model(frame)[0]
            img = results.plot(line_width=1)
            # 变成QImage形式
            qImage = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
            # 往显示Label里 显示QImage
            self.label_treated.setPixmap(QtGui.QPixmap.fromImage(qImage))
            time.sleep(0.5)

    def Train(self):
        if self.selectB_1.currentText() == 'CULane':
            self.data_root = './CULane'
            self.dataset = 'CULane'
        else:
            self.data_root = './Tusimple'
            self.dataset = 'Tusimple'

        self.batch_size = int(self.Text_2.text())
        self.griding_num = int(self.selectB_3.currentText())
        self.epoch = int(self.Text_1.text())
        # 处理数据
        train_loader, cls_num_per_lane = get_train_loader(self.batch_size, self.data_root, self.griding_num,
                                                          self.dataset,
                                                          self.use_aux, False, self.num_lanes)
        # 设置网络
        net = parsingNet(pretrained=True, backbone=backbone,
                         cls_dim=(self.griding_num + 1, cls_num_per_lane, self.num_lanes), use_aux=self.use_aux).cpu()
        # 获取优化方式
        training_params = filter(lambda p: p.requires_grad, net.parameters())
        if self.selectB_4.currentText() == 'SGD（随机梯度下降）':
            self.optimizer = 'SGD'
        else:
            self.optimizer = 'Adam'

        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(training_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(training_params, lr=self.learning_rate, momentum=self.momentum,
                                        weight_decay=self.weight_decay)
        else:
            raise NotImplementedError
        # 获取学习率衰减策略
        iters_per_epoch = len(train_loader)
        if self.selectB_2.currentText() == '多步长衰减策略':
            self.scheduler = 'multi'
        else:
            self.scheduler = 'cos'

        if self.scheduler == 'multi':  # 多步长衰减
            scheduler = MultiStepLR(optimizer, self.steps, self.gamma, self, self.warmup,
                                    iters_per_epoch if self.warmup_iters is None else self.warmup_iters)
        elif self.scheduler == 'cos':  # 余弦退火衰减
            scheduler = CosineAnnealingLR(optimizer, self.epoch * iters_per_epoch, eta_min=0, warmup=self.warmup,
                                          warmup_iters=self.warmup_iters)
        else:
            raise NotImplementedError
        # 获取指标策略
        if self.use_aux:
            metric_dict = {
                'name': ['top1', 'top2', 'top3', 'iou'],
                'op': [MultiLabelAcc(), AccTopk(self.griding_num, 2), AccTopk(self.griding_num, 3),
                       Metric_mIoU(self.num_lanes + 1)],
                'data_src': [('cls_out', 'cls_label'), ('cls_out', 'cls_label'), ('cls_out', 'cls_label'),
                             ('seg_out', 'seg_label')]
            }
        else:
            metric_dict = {
                'name': ['top1', 'top2', 'top3'],
                'op': [MultiLabelAcc(), AccTopk(self.griding_num, 2), AccTopk(self.griding_num, 3)],
                'data_src': [('cls_out', 'cls_label'), ('cls_out', 'cls_label'), ('cls_out', 'cls_label')]
            }
        # 获取损失函数策略
        if self.use_aux:
            loss_dict = {
                'name': ['cls_loss', 'relation_loss', 'aux_loss', 'relation_dis'],
                'op': [SoftmaxFocalLoss(2), ParsingRelationLoss(), torch.nn.CrossEntropyLoss(), ParsingRelationDis()],
                'weight': [1.0, self.sim_loss_w, 1.0, self.shp_loss_w],
                'data_src': [('cls_out', 'cls_label'), ('cls_out',), ('seg_out', 'seg_label'), ('cls_out',)]
            }
        else:
            loss_dict = {
                'name': ['cls_loss', 'relation_loss', 'relation_dis'],
                'op': [SoftmaxFocalLoss(2), ParsingRelationLoss(), ParsingRelationDis()],
                'weight': [1.0, self.sim_loss_w, self.shp_loss_w],
                'data_src': [('cls_out', 'cls_label'), ('cls_out',), ('cls_out',)]
            }
        # 打开文件，以配置参数写入日志文件中的cfg.txt文件中
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # 获取现在的时间
        hyper_param_str = '_lr_%1.0e_b_%d' % (self.learning_rate, self.batch_size)  # 获取学习率和batch size
        work_dir = os.path.join(self.log_path, now + hyper_param_str)  # 将上面两个组合
        logger = DistSummaryWriter(work_dir)  # 获取项目所在路径
        # 训练
        for epoch in range(0, self.epoch):
            net.train()  # 模型训练
            progress_bar = dist_tqdm(train_loader)
            t_data_0 = time.time()  # 获取当前时间
            for b_idx, data_label in enumerate(progress_bar):
                t_data_1 = time.time()  # 获取当前时间
                reset_metrics(metric_dict)  # 重置指标参数
                global_step = epoch * len(train_loader) + b_idx  # 轮次
                Hhh = int(global_step / (self.epoch * len(train_loader)))
                t_net_0 = time.time()  # 获取当前时间
                results = inference(net, data_label, self.use_aux)  # 将损失值导入训练中
                loss = calc_loss(loss_dict, results, logger, global_step)  # 损失值
                optimizer.zero_grad()  # 梯度清零
                loss.backward()  # 反向传播
                optimizer.step()  # 参数更新
                scheduler.step(global_step)  # 对学习率进行更新
                t_net_1 = time.time()  # 获取当前时间
                results = resolve_val_data(results, self.use_aux)  # 将结果归一化
                update_metrics(metric_dict, results)  # 更新度量参数
                # 将相关参数存入到日志文件中，方便后续使用Tensorboard可视化观察
                if global_step % 20 == 0:
                    for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                        logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)
                logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)
                if hasattr(progress_bar, 'set_postfix'):
                    kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in
                              zip(metric_dict['name'], metric_dict['op'])}
                    progress_bar.set_postfix(loss='%.3f' % float(loss),
                                             data_time='%.3f' % float(t_data_1 - t_data_0),
                                             net_time='%.3f' % float(t_net_1 - t_net_0),
                                             **kwargs)
                t_data_0 = time.time()  # 获取当前时间
                self.bar.setFormat('Loaded  %p%'.format(Hhh))
                if epoch * iters_per_epoch != self.train_epoch:
                    self.train_epoch += 1
                self.Loss = loss
                for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                    if me_name == 'top1':
                        self.t1 = me_op.get()
                    if me_name == 'iou':
                        self.Miou = me_op.get()
                self.train_label = QLabel(
                    'epoch：' + str(self.train_epoch) + '\t\t\t\tmiou：' + str(self.Miou) + '\t\t\t\ttop1：' + str(
                        self.t1) + '\t\t\t\tloss：' + str(self.Loss))

    def Tensorboard(self):
        self.process = QProcess(self)
        self.process.start('tensorboard --logdir /Users/mack/Desktop/work/车道线检测/log')
        if not self.process.waitForStarted():
            QMessageBox.critical(self, '错误', 'Tensorboard 启动失败')
        self.time_1.start(5000)

    def start_tensorboard(self):
        url = 'http://localhost:6006/'
        webbrowser.open(url)
        self.time_1.stop()
        self.time_2.start(5000)

    def stop_tensorboard(self):
        self.process.terminate()
        self.process.waitForFinished()
        self.time_2.stop()

    def Photo(self):
        if self.SBox_1.currentText() == 'Ours-CULane':
            self.test_model = r'/Users/mack/Desktop/work/车道线检测/模型/culane_18.pth'
        if self.SBox_1.currentText() == 'Ours-Tusimple':
            self.test_model = r'/Users/mack/Desktop/work/车道线检测/模型/tusimple_18.pth'
        if self.SBox_1.currentText() == 'SCNN-CULane':
            self.test_model = r'/Users/mack/Desktop/work/车道线检测/模型/culane_SCNN.pth'
        if self.SBox_1.currentText() == 'SCNN-Tusimple':
            self.test_model = r'/Users/mack/Desktop/work/车道线检测/模型/tusimple_SCNN.pth'
        if self.SBox_1.currentText() == 'SAD-CULane':
            self.test_model = r'/Users/mack/Desktop/work/车道线检测/模型/culane_SAD.pth'
        if self.SBox_1.currentText() == 'SAD-Tusimple':
            self.test_model = r'/Users/mack/Desktop/work/车道线检测/模型/tusimple_SAD.pth'
        if self.SBox_1.currentText() == 'ResNet-CULane':
            self.test_model = r'/Users/mack/Desktop/work/车道线检测/模型/culane_ResNet.pth'
        if self.SBox_1.currentText() == 'ResNet-Tusimple':
            self.test_model = r'/Users/mack/Desktop/work/车道线检测/模型/tusimple_ResNet.pth'
        self.textLog_1.clear()
        self.selected_imgsPath, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "打开图片", "./测试数据/图片", "*.jpg;;*.png;;All Files(*)")
        if len(self.selected_imgsPath) == 0:
            self.empty_information()
            return
        img = cv2.imread(self.selected_imgsPath[0])
        self.SAVE = use_model(img, self.ori, self.tri, self.net, self.img_transforms, self.griding_num)
        self.current_image_index = 0
        SStr = '一共有' + str(len(self.selected_imgsPath)) + '张图片\n'
        SStr += '此为第' + str(self.current_image_index + 1) + '张图片\n'
        self.textLog_1.setText(SStr)

    def showNextImage(self):
        if len(self.selected_imgsPath) == 0:
            return
        self.current_image_index += 1
        if self.current_image_index >= len(self.selected_imgsPath):
            self.current_image_index = 0
        img = cv2.imread(self.selected_imgsPath[self.current_image_index])
        self.SAVE = use_model(img, self.ori, self.tri, self.net, self.img_transforms, self.griding_num)
        SStr = '一共有' + str(len(self.selected_imgsPath)) + '张图片\n'
        SStr += '此为第' + str(self.current_image_index + 1) + '张图片\n'
        self.textLog_1.setText(SStr)

    def showPrevImage(self):
        if len(self.selected_imgsPath) == 0:
            return
        self.current_image_index -= 1
        if self.current_image_index < 0:
            self.current_image_index = len(self.selected_imgsPath) - 1
        img_path = self.selected_imgsPath[self.current_image_index]
        img = cv2.imread(img_path)
        self.SAVE = use_model(img, self.ori, self.tri, self.net, self.img_transforms, self.griding_num)
        SStr = '一共有' + str(len(self.selected_imgsPath)) + '张图片\n'
        SStr += '此为第' + str(self.current_image_index + 1) + '张图片\n'
        self.textLog_1.setText(SStr)

    def save_phonto(self):
        if self.SAVE is not None:
            cv2.imwrite('./保存/图片', self.SAVE)

    def save_video(self):
        return


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    