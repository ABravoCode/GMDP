# 本地运行参数，pbatch大了可以继续往小了改
conda activate MachineLearning
python3 brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2000000000 --budget 0.01 --restarts 8 --pbatch 64 --ensemble 1 --name ablation_no_aug_no_aug --paugment --noaugment