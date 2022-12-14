"""Base victim class."""

import torch

from .models import get_model
from .training import get_optimizers, run_step
from .optimization_strategy import training_strategy
from ..utils import average_dicts
from ..consts import BENCHMARK, SHARING_STRATEGY
torch.backends.cudnn.benchmark = BENCHMARK
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


class _VictimBase:
    """Implement model-specific code and behavior.

    Expose:
    Attributes:
     - model
     - optimizer
     - scheduler
     - criterion

     Methods:
     - initialize
     - train
     - retrain
     - validate
     - iterate

     - compute
     - gradient
     - eval

     Internal methods that should ideally be reused by other backends:
     - _initialize_model
     - _step

    """

    def __init__(self, args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize empty victim."""
        super(_VictimBase, self).__init__()
        self.args, self.setup = args, setup
        # A: 容错, 集成模型与参数必须对应
        if self.args.ensemble < len(self.args.net):
            raise ValueError(f'More models requested than ensemble size.'
                             f'Increase ensemble size or reduce models.')
        # A: 初始化在各继承类中实现
        self.initialize()

    # A: 在子类中实现梯度计算
    def gradient(self, images, labels):
        """Compute the gradient of criterion(model) w.r.t to given data."""
        raise NotImplementedError()
        return grad, grad_norm

    # A: 在子类中实现计算
    def compute(self, function):
        """Compute function on all models.

        Function has arguments: model, criterion
        """
        raise NotImplementedError()

    # A: 分布式设置, 子类
    def distributed_control(self, inputs, labels, poison_slices, batch_positions):
        """Control distributed poison brewing, no-op in single network training."""
        randgen = None
        return inputs, labels, poison_slices, batch_positions, randgen

    def sync_gradients(self, input):
        """Sync gradients of given variable. No-op for single network training."""
        return input

    # A: 重设lr
    def reset_learning_rate(self):
        """Reset scheduler object to initial state."""
        raise NotImplementedError()


    """ Methods to initialize a model."""
    # A: 在子类中实现
    def initialize(self, seed=None):
        raise NotImplementedError()

    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED POISONS"""
    # A: 不同模式iterate不同
    def train(self, kettle, max_epoch=None):
        """Clean (pre)-training of the chosen model, no poisoning involved."""
        print('Starting clean training ...')
        return self._iterate(kettle, poison_delta=None, max_epoch=max_epoch)

    def retrain(self, kettle, poison_delta):
        """Check poison on the initialization it was brewed on."""
        self.initialize(seed=self.model_init_seed)
        print('Model re-initialized to initial seed.')
        return self._iterate(kettle, poison_delta=poison_delta)

    # A: 验证集
    def validate(self, kettle, poison_delta):
        """Check poison on a new initialization(s)."""
        run_stats = list()
        for runs in range(self.args.vruns):
            self.initialize()
            print('Model reinitialized to random seed.')
            run_stats.append(self._iterate(kettle, poison_delta=poison_delta))

        return average_dicts(run_stats)

    def eval(self, dropout=True):
        """Switch everything into evaluation mode."""
        raise NotImplementedError()

    # A: [已解决] 这么多函数调用了这个这函数就raise个错??
    # A: 子类继承后单独实现, 同上
    # 技术博客: https://www.cnblogs.com/everfight/p/NotImplementedError.html 
    def _iterate(self, kettle, poison_delta):
        """Validate a given poison by training the model and checking target accuracy."""
        raise NotImplementedError()

    def _adversarial_step(self, kettle, poison_delta, step, poison_targets, true_classes):
        """Step through a model epoch to in turn minimize target loss."""
        raise NotImplementedError()

    # A: 初始化模型(default)
    def _initialize_model(self, model_name):

        model = get_model(model_name, self.args.dataset, pretrained=self.args.pretrained)
        # Define training routine
        defs = training_strategy(model_name, self.args)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer, scheduler = get_optimizers(model, self.args, defs)

        return model, defs, criterion, optimizer, scheduler


    # A: 单步运行
    def _step(self, kettle, poison_delta, loss_fn, epoch, stats, model, defs, criterion, optimizer, scheduler):
        """Single epoch. Can't say I'm a fan of this interface, but ..."""
        run_step(kettle, poison_delta, loss_fn, epoch, stats, model, defs, criterion, optimizer, scheduler)
