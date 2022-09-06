创建于**20220328**，用于记录打卡目标梯度对齐黑盒攻击项目进度。

[TOC]

## 提示

实验跑之前想清楚参数是否合理。

跑代码<font color=red>**提问题**</font> 不知道用什么指令可以看./paper_experiments/*或./Examplars

*代码跑不起来的问题也可以提*

问题往quesion文件夹里塞 记得描述清楚问题 提交就写Update Questions就好


## 待办事项

功能——梯度对齐功能优化: 对齐方向不够准确
消融实验

## 已完成

干净标签投毒攻击论文精读: FC, CP, BP, WB

优化论文阅读: ZO-AdaMM

BP代码理解阅读与部分实验

服务器环境搭建

WB与ZO实验复现

深度学习花书优化章节与基础复习

梯度估计模块搭建与嵌入

单毒物生成运行流程



## 草稿纸/随记

由budget参数影响的毒物数量会直接影响投毒训练的acc，poisonloader中的size被这样定义：

```python
validated_batch_size = max(min(args.pbatch, len(self.poisonset)), 1)
```

而poisonset由函数*_choose_poisons_deterministic*定义，选取前n个类型中的图像作为毒物。

```python
self.poisonset, self.targetset, self.validset = self._choose_poisons_deterministic(target_id)

def _choose_poisons_deterministic(self, target_id):
        # poisons
        class_ids = []
        for index in range(len(self.trainset)):  # we actually iterate this way not to iterate over the images
            target, idx = self.trainset.get_target(index)
            if target == self.poison_setup['poison_class']:
                class_ids.append(idx)

        poison_num = int(np.ceil(self.args.budget * len(self.trainset)))
        if len(class_ids) < poison_num:
            warnings.warn(f'Training set is too small for requested poison budget.')
            poison_num = len(class_ids)
        self.poison_ids = class_ids[:poison_num]
```

遇到规模不匹配问题，使用以下代码解决：

```python
elif self.args.loss == 'cosine1':
                orig_target_grad = []
                for _ in range(500):
                    orig_target_grad.append(target_grad.unsqueeze_(0))
                target_grad = torch.cat(orig_target_grad, dim=0).flatten()
                # print(target_grad.shape)
                passenger_loss -= torch.nn.functional.cosine_similarity((target_grad[i]).flatten(), poison_grad[i].flatten(), dim=0)
```

