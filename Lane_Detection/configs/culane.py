# DATA
dataset = 'CULane'
data_root = './CULane'
# TRAIN
epoch = 10
batch_size = 32
optimizer = 'SGD'  # ['SGD','Adam']，优化器
learning_rate = 4e-4     # 学习率
weight_decay = 1e-4     # 权重衰减
momentum = 0.9      # 动量
scheduler = 'multi'  # ['multi', 'cos']，学习率衰减策略
steps = [25, 38]    # 学习率变化相关的参数
gamma = 0.1         # 学习率更新的一个参数
warmup = 'linear'   # 预热方式
warmup_iters = 695  # 预热轮次
# NETWORK
use_aux = True      # 使用数据增强
griding_num = 200  # 网格数量Cell
backbone = '18'     # 主干网
# LOSS
sim_loss_w = 0.0
shp_loss_w = 0.0
# EXP
note = ''
log_path = './log'  # 日志
# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None
# TEST
test_model = None
test_work_dir = None
num_lanes = 4
