import torch, os, datetime
from model.model import parsingNet
from data.dataloader import get_train_loader
from utils.dist_utils import dist_print, dist_tqdm
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import update_metrics, reset_metrics
from utils.common import merge_config, save_model, cp_projects
from utils.common import get_work_dir, get_logger
import time


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


def resolve_val_data(results, use_aux):
    results['cls_out'] = torch.argmax(results['cls_out'], dim=1)
    if use_aux:
        results['seg_out'] = torch.argmax(results['seg_out'], dim=1)
    return results


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


def train(net, data_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, use_aux):
    net.train()     # 模型训练
    progress_bar = dist_tqdm(train_loader)
    t_data_0 = time.time()      # 获取当前时间
    for b_idx, data_label in enumerate(progress_bar):
        t_data_1 = time.time()  # 获取当前时间
        reset_metrics(metric_dict)  # 重置指标参数
        global_step = epoch * len(data_loader) + b_idx  # 轮次
        t_net_0 = time.time()   # 获取当前时间
        results = inference(net, data_label, use_aux)   # 将损失值导入训练中
        loss = calc_loss(loss_dict, results, logger, global_step)   # 损失值
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新
        scheduler.step(global_step)  # 对学习率进行更新
        t_net_1 = time.time()  # 获取当前时间
        results = resolve_val_data(results, use_aux)    # 将结果归一化
        update_metrics(metric_dict, results)    # 更新度量参数
        # 将相关参数存入到日志文件中，方便后续使用Tensorboard可视化观察
        if global_step % 20 == 0:
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)
        logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        if hasattr(progress_bar, 'set_postfix'):
            kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
            progress_bar.set_postfix(loss='%.3f' % float(loss),
                                     data_time='%.3f' % float(t_data_1 - t_data_0),
                                     net_time='%.3f' % float(t_net_1 - t_net_0),
                                     **kwargs)
        t_data_0 = time.time()  # 获取当前时间


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True   # 让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
    args, cfg = merge_config()  # 获取配置行参数culane.py
    work_dir = get_work_dir(cfg)  # 找到项目所在的路径

    distributed = False     # 判断是否并行
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cpu.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    dist_print(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')  # 打印开始训练
    dist_print(cfg)     # 打印出config配置项
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']   # 断言，看配置参数是否正确
    # 处理数据
    train_loader, cls_num_per_lane = get_train_loader(cfg.batch_size, cfg.data_root, cfg.griding_num, cfg.dataset,
                                                      cfg.use_aux, distributed, cfg.num_lanes)
    # 设置网络
    net = parsingNet(pretrained=True, backbone=cfg.backbone,
                     cls_dim=(cfg.griding_num + 1, cls_num_per_lane, cfg.num_lanes), use_aux=cfg.use_aux).cpu()
    # 单机多GPU并行
    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])

    optimizer = get_optimizer(net, cfg)     # 获取优化方式，此处为SGD随机梯度下降

    if cfg.finetune is not None:    # 微调
        dist_print('finetune from ', cfg.finetune)
        state_all = torch.load(cfg.finetune)['model']
        state_clip = {}  # 仅使用主干参数
        for k, v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)
    # 从保存点检测点恢复训练
    if cfg.resume is not None:
        dist_print('==> Resume model from ' + cfg.resume)
        resume_dict = torch.load(cfg.resume, map_location='cpu')
        net.load_state_dict(resume_dict['model'])
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1
    else:
        resume_epoch = 0

    scheduler = get_scheduler(optimizer, cfg, len(train_loader))    # 获取学习率衰减策略
    dist_print(len(train_loader))       # 打印出一轮需要训练多少次
    metric_dict = get_metric_dict(cfg)      # 获取指标策略
    loss_dict = get_loss_dict(cfg)      # 获取损失函数策略
    logger = get_logger(work_dir, cfg)      # 打开文件，以配置参数写入日志文件中的cfg.txt文件中
    cp_projects(args.auto_backup, work_dir)     # common.py

    for epoch in range(resume_epoch, cfg.epoch):
        train(net, train_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, cfg.use_aux)  # 训练模型
        save_model(net, optimizer, epoch, work_dir)    # 保存模型

    logger.close()  # 关闭文件
    