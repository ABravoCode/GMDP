"""General interface script to launch poisoning jobs."""

import torch

import os
import datetime
import time

import forest

torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)

# Parse input arguments
args = forest.options().parse_args()
# 100% reproducibility?
# A: ./forest/options.py中的参数, 是否使用CUDNN中的可复现功能(choose seed)
# 关于此技术，可参考博客-https://vimsky.com/examples/detail/python-method-torch.backends.cudnn.deterministic.html
if args.deterministic:
    forest.utils.set_deterministic()


if __name__ == "__main__":

    setup = forest.utils.system_startup(args)

    # T: 对被投毒对象、数据集、投毒攻击进行初始化
    # A: model初始化时, 主要关注于毒物生成时是否集成多种模型与分布式选项
    model = forest.Victim(args, setup=setup)
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations, setup=setup)
    # A: 选择投毒方式对象
    witch = forest.Witch(args, setup=setup)

    # A: 使用stats_clean变量可进行类似存档读档操作
    start_time = time.time()
    if args.pretrained:
        print('Loading pretrained model...')
        stats_clean = None
        if os.path.exists('./{}_{}_{}_clean_model.pth'.format(args.dataset, args.net, args.optimization)):
            model = torch.load('./{}_{}_{}_clean_model.pth'.format(args.dataset, args.net, args.optimization))
        else:
            raise OSError('Model do not exist')
    else:
        # A: ./forest/victims/victim_base.py -> def train... ->victim_single.py -> def iterate...
        stats_clean = model.train(data, max_epoch=args.max_epoch)
    train_time = time.time()
    torch.save(model, './{}_{}_{}_clean_model.pth'.format(args.dataset, args.net, args.optimization))

    # T:获取投毒攻击
    poison_delta = witch.brew(model, data)
    brew_time = time.time()

    if not args.pretrained and args.retrain_from_init:
        stats_rerun = model.retrain(data, poison_delta)
    else:
        stats_rerun = None  # we dont know the initial seed for a pretrained model so retraining makes no sense

    if args.vnet is not None:  # Validate the transfer model given by args.vnet
        train_net = args.net
        args.net = args.vnet
        if args.vruns > 0:
            model = forest.Victim(args, setup=setup)
            stats_results = model.validate(data, poison_delta)
        else:
            stats_results = None
        args.net = train_net
    else:  # Validate the main model
        if args.vruns > 0:
            stats_results = model.validate(data, poison_delta)
        else:
            stats_results = None
    test_time = time.time()

    torch.save(model, './{}_{}_{}_poisoned_model.pth'.format(args.dataset, args.net, args.optimization))

    # A: 时间戳保存并展示
    timestamps = dict(train_time=str(datetime.timedelta(seconds=train_time - start_time)).replace(',', ''),
                      brew_time=str(datetime.timedelta(seconds=brew_time - train_time)).replace(',', ''),
                      test_time=str(datetime.timedelta(seconds=test_time - brew_time)).replace(',', ''))
    # Save run to table
    results = (stats_clean, stats_rerun, stats_results)
    forest.utils.record_results(data, witch.stat_optimal_loss, results,
                                args, model.defs, model.model_init_seed, extra_stats=timestamps)

    # Export
    if args.save is not None:
        data.export_poison(poison_delta, path=args.poison_path, mode=args.save)

    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with train time: {str(datetime.timedelta(seconds=train_time - start_time))}')
    print(f'--------------------------- brew time: {str(datetime.timedelta(seconds=brew_time - train_time))}')
    print(f'--------------------------- test time: {str(datetime.timedelta(seconds=test_time - brew_time))}')
    print('-------------Job finished.-------------------------')
