"""Implement an ArgParser common to both brew_poison.py and dist_brew_poison.py ."""
"""A: 可能因为参数太多了专门写了个传参函数 =.="""

import argparse


def options():
    """Construct the central argument parser, filled with useful defaults.

    The first block is essential to test poisoning in different scenarios.
    The options following afterwards change the algorithm in various ways and are set to reasonable defaults.
    """
    parser = argparse.ArgumentParser(description='Construct poisoned training data for the given network and dataset')

    ###########################################################################
    # Central:
    # A: 网络结构选择，可用列表输入
    parser.add_argument('--net', default='ResNet18', type=lambda s: [str(item) for item in s.split(',')])
    # A: 只支持choices里等各种数据集
    parser.add_argument('--dataset', default='CIFAR10', type=str,
                        choices=['CIFAR10', 'CIFAR100', 'ImageNet', 'ImageNet1k', 'MNIST', 'TinyImageNet'])
    # A: 不同投毒方式，metapoison的方式自己看
    parser.add_argument('--recipe', default='gradient-matching', type=str,
                        choices=['gradient-matching', 'gradient-matching-private',
                                 'watermarking', 'poison-frogs', 'metapoison', 'bullseye'])
    # A: 投毒目标
    parser.add_argument('--threatmodel', default='single-class', type=str,
                        choices=['single-class', 'third-party', 'random-subset'])

    # Reproducibility management:
    # A: 可复现性保证，详见代码根目录等readme
    parser.add_argument('--poisonkey', default=None, type=str,
                        help='Initialize poison setup with this key.')  # Also takes a triplet 0-3-1
    parser.add_argument('--modelkey', default=None, type=int, help='Initialize the model with this key.')
    parser.add_argument('--deterministic', action='store_true', help='Disable CUDNN non-determinism.')

    # Poison properties / controlling the strength of the attack:
    # A: Lp范数扰动: epsilon, delta与目标个数
    parser.add_argument('--eps', default=16, type=float)
    parser.add_argument('--budget', default=0.01, type=float, help='Fraction of training data that is poisoned')
    parser.add_argument('--targets', default=1, type=int, help='Number of targets')

    # Files and folders
    # A: 文件IO目录
    parser.add_argument('--name', default='', type=str,
                        help='Name tag for the result table and possibly for export folders.')
    parser.add_argument('--table_path', default='tables/', type=str)
    parser.add_argument('--poison_path', default='poisons/', type=str)
    parser.add_argument('--data_path', default='~/data', type=str)
    ###########################################################################

    # Poison brewing:
    """
    A: 攻击时采用的优化器
    注意!在未来的工作中此处应改为ZO-AdaMM
    阅读代码时请除功能外尤其注意attackoptim的通信机制与接口
    """
    parser.add_argument('--attackoptim', default='signAdam', type=str)
    parser.add_argument('--attackiter', default=250, type=int)
    parser.add_argument('--init', default='randn', type=str)  # randn / rand
    parser.add_argument('--tau', default=0.1, type=float)
    parser.add_argument('--scheduling', action='store_false', help='Disable step size decay.')
    # A: 使用交叉熵作为loss函数
    parser.add_argument('--target_criterion', default='cross-entropy', type=str, help='Loss criterion for target loss')
    parser.add_argument('--restarts', default=8, type=int, help='How often to restart the attack.')

    parser.add_argument('--pbatch', default=512, type=int, help='Poison batch size during optimization')
    # A: 是否打乱数据顺序
    parser.add_argument('--pshuffle', action='store_true', help='Shuffle poison batch during optimization')
    # A: 数据增强
    parser.add_argument('--paugment', action='store_false', help='Do not augment poison batch during optimization')
    parser.add_argument('--data_aug', type=str, default='default', help='Mode of diff. data augmentation.')

    # Poisoning algorithm changes
    parser.add_argument('--full_data', action='store_true',
                        help='Use full train data (instead of just the poison images)')
    parser.add_argument('--adversarial', default=0, type=float, help='Adversarial PGD for poisoning.')
    parser.add_argument('--ensemble', default=1, type=int, help='Ensemble of networks to brew the poison on')
    parser.add_argument('--stagger', action='store_true', help='Stagger the network ensemble if it exists')
    parser.add_argument('--step', action='store_true', help='Optimize the model for one epoch.')
    parser.add_argument('--max_epoch', default=None, type=int, help='Train only up to this epoch before poisoning.')

    # Use only a subset of the dataset:
    # A: 用多少数据
    parser.add_argument('--ablation', default=1.0, type=float,
                        help='What percent of data (including poisons) to use for validation')

    # Gradient Matching - Specific Options
    # A: 在梯度匹配中使用余弦相似度作为loss
    parser.add_argument('--loss', default='similarity', type=str)  # similarity is stronger in  difficult situations

    # These are additional regularization terms for gradient matching. We do not use them, but it is possible
    # that scenarios exist in which additional regularization of the poisoned data is useful.
    # A: 论文实验里没用，但是可能在某些特殊的场景下会用到的正则化手段（先不管）
    parser.add_argument('--centreg', default=0, type=float)
    parser.add_argument('--normreg', default=0, type=float)
    parser.add_argument('--repel', default=0, type=float)

    # Specific Options for a metalearning recipe
    # A: Metapoison参数，**第二项**可能与ZO-AdaMM会有关系
    parser.add_argument('--nadapt', default=2, type=int, help='Meta unrolling steps')
    parser.add_argument('--clean_grad', action='store_true', help='Compute the first-order poison gradient.')

    # Validation behavior
    parser.add_argument('--vruns', default=1, type=int,
                        help='How often to re-initialize and check target after retraining')
    # A: 验证毒物效果模型选择
    parser.add_argument('--vnet', default=None, type=lambda s: [str(item) for item in s.split(',')],
                        help='Evaluate poison on this victim model. Defaults to --net')
    # A: 是否载入Check_point
    parser.add_argument('--retrain_from_init', action='store_true',
                        help='Additionally evaluate by retraining on the same model initialization.')

    # Optimization setup
    parser.add_argument('--pretrained', action='store_true',
                        help='Load pretrained models from torchvision, if possible [only valid for ImageNet].')
    parser.add_argument('--optimization', default='conservative', type=str, help='Optimization Strategy')
    # Strategy overrides:
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--noaugment', action='store_true', help='Do not use data augmentation during training.')
    # A: 梯度噪声或是修剪(随机干扰)
    parser.add_argument('--gradient_noise', default=None, type=float, help='Add custom gradient noise during training.')
    parser.add_argument('--gradient_clip', default=None, type=float, help='Add custom gradient clip during training.')

    # Optionally, datasets can be stored as LMDB or within RAM:
    # A: LMDB用于提高性能。整个数据库在内存映射中公开，所有数据获取都直接从映射的内存返回数据，因此在数据获取期间不会出现malloc或memcpy。
    parser.add_argument('--lmdb_path', default=None, type=str)
    parser.add_argument('--cache_dataset', action='store_true', help='Cache the entire thing :>')

    # These options allow for testing against the toxicity benchmark found at
    # https://github.com/aks2203/poisoning-benchmark
    # A: 标准性能测试选项
    parser.add_argument('--benchmark', default='', type=str, help='Path to benchmarking setup (pickle file)')
    parser.add_argument('--benchmark_idx', default=0, type=int, help='Index of benchmark test')

    # Debugging:
    parser.add_argument('--dryrun', action='store_true')
    parser.add_argument('--save', default=None,
                        help='Export poisons into a given format. Options are full/limited/automl/numpy.')

    # Distributed Computations
    parser.add_argument("--local_rank", default=None, type=int, help='Distributed rank. This is an INTERNAL ARGUMENT! '
                                                                     'Only the launch utility should set this argument!')

    return parser
