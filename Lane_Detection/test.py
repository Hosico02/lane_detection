import torch, os
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
from evaluation.eval_wrapper import eval_lane
import torch

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True   # 加速
    args, cfg = merge_config()  # 获取配置项
    distributed = False     # 是否并行
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    if distributed:
        torch.cpu.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    dist_print('start testing...')  # 打印开始测试
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']
    # 选择数据集对应的行锚数量
    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError
    # 网络，测试时不使用分割
    net = parsingNet(pretrained=False, backbone=cfg.backbone,
                     cls_dim=(cfg.griding_num + 1, cls_num_per_lane, cfg.num_lanes),
                     use_aux=False).cpu()
    # 模型参数
    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
    # 导入参数
    net.load_state_dict(compatible_state_dict, strict=False)
    # 如果并行
    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])
    # 打开保存的路径
    if not os.path.exists(cfg.test_work_dir):
        os.mkdir(cfg.test_work_dir)
    # 保存测试结果
    eval_lane(net, cfg.dataset, cfg.data_root, cfg.test_work_dir, cfg.griding_num, False, distributed)
