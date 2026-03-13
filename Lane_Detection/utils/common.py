import argparse
from utils.dist_utils import is_main_process, dist_print, DistSummaryWriter
from utils.config import Config
import torch
import time
import pathspec
import datetime, os


# 定义一种新的类型来表示TRUE OR FALSE
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# 定义命令行参数的初始化语句
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to config file')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--data_root', default=None, type=str)
    parser.add_argument('--epoch', default=None, type=int)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--optimizer', default=None, type=str)
    parser.add_argument('--learning_rate', default=None, type=float)
    parser.add_argument('--weight_decay', default=None, type=float)
    parser.add_argument('--momentum', default=None, type=float)
    parser.add_argument('--scheduler', default=None, type=str)
    parser.add_argument('--steps', default=None, type=int, nargs='+')
    parser.add_argument('--gamma', default=None, type=float)
    parser.add_argument('--warmup', default=None, type=str)
    parser.add_argument('--warmup_iters', default=None, type=int)
    parser.add_argument('--backbone', default=None, type=str)
    parser.add_argument('--griding_num', default=None, type=int)
    parser.add_argument('--use_aux', default=None, type=str2bool)
    parser.add_argument('--sim_loss_w', default=None, type=float)
    parser.add_argument('--shp_loss_w', default=None, type=float)
    parser.add_argument('--note', default=None, type=str)
    parser.add_argument('--log_path', default=None, type=str)
    parser.add_argument('--finetune', default=None, type=str)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--test_model', default=None, type=str)
    parser.add_argument('--test_work_dir', default=None, type=str)
    parser.add_argument('--num_lanes', default=None, type=int)
    parser.add_argument('--auto_backup', action='store_true', help='automatically backup current code in the log path')

    return parser


# 获取由配置项文件或者命令行定义的参数
def merge_config():
    args = get_args().parse_args()
    cfg = Config.fromfile(args.config)

    items = ['dataset', 'data_root', 'epoch', 'batch_size', 'optimizer', 'learning_rate',
             'weight_decay', 'momentum', 'scheduler', 'steps', 'gamma', 'warmup', 'warmup_iters',
             'use_aux', 'griding_num', 'backbone', 'sim_loss_w', 'shp_loss_w', 'note', 'log_path',
             'finetune', 'resume', 'test_model', 'test_work_dir', 'num_lanes']
    for item in items:
        if getattr(args, item) is not None:
            dist_print('merge ', item, ' config')
            setattr(cfg, item, getattr(args, item))
    return args, cfg


# 保存训练好的模型
def save_model(net, optimizer, epoch, save_path):
    if is_main_process():
        model_state_dict = net.state_dict()     # 各个层的权重和偏置
        state = {'model': model_state_dict, 'optimizer': optimizer.state_dict()}
        # state = {'model': model_state_dict}
        assert os.path.exists(save_path)
        model_path = os.path.join(save_path, 'ep%03d.pth' % epoch)
        torch.save(state, model_path)


def cp_projects(auto_backup, to_path):
    if is_main_process() and auto_backup:
        with open('./.gitignore', 'r') as fp:
            ign = fp.read()
        ign += '\n.git'
        spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, ign.splitlines())
        all_files = {os.path.join(root, name) for root, dirs, files in os.walk('./') for name in files}
        matches = spec.match_files(all_files)
        matches = set(matches)
        to_cp_files = all_files - matches
        dist_print('Copying projects to ' + to_path + ' for backup')
        t0 = time.time()        # 获取当前时间
        warning_flag = True
        for f in to_cp_files:
            dirs = os.path.join(to_path, 'code', os.path.split(f[2:])[0])
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            os.system('cp %s %s' % (f, os.path.join(to_path, 'code', f[2:])))
            elapsed_time = time.time() - t0
            if elapsed_time > 5 and warning_flag:
                dist_print(
                    '如果程序被卡住，它可能正在复制此目录中的大文件. 请不要设置--auto_backup '
                    '或者请将工作目录清理干净,不要像数据集那样放置大文件，将结果记录在此目录下')
                warning_flag = False


# 得到所需创建的文件夹的名字（年月日，时分秒，学习率，batch size）
def get_work_dir(cfg):
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')     # 获取现在的时间
    hyper_param_str = '_lr_%1.0e_b_%d' % (cfg.learning_rate, cfg.batch_size)    # 获取学习率和batch size
    work_dir = os.path.join(cfg.log_path, now + hyper_param_str + cfg.note)     # 将上面两个组合
    return work_dir


# 以配置参数写入日志文件中的cfg.txt文件中
def get_logger(work_dir, cfg):
    logger = DistSummaryWriter(work_dir)    # 获取项目所在路径
    config_txt = os.path.join(work_dir, 'cfg.txt')  # 找到所需写入的txt文件中
    if is_main_process():
        with open(config_txt, 'w') as fp:   # 打开txt文件
            fp.write(str(cfg))      # 将配置参数写入

    return logger
