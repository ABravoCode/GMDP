ssh mist@url -p $port
cd cloud/wichers-brew/
screen -S train

python3 brew_poison.py  --net ResNet18 --vruns 1 --attackiter 120 --poisonkey 2000000000 --budget 0.01 --restarts 1 --pbatch 512 --ensemble 1 --name ablation_no_aug_no_aug --save limited
# Success, Aprox 100min

python3 brew_poison.py  --recipe bullseye --net ResNet18 --vruns 1 --attackiter 100 --poisonkey 2000000000 --budget 0.01 --restarts 1 --pbatch 512 --ensemble 1 --name ablation_no_aug_no_aug --save limited
# Fail, Aprox 90min

python3 brew_poison.py  --net ResNet18 --vruns 1 --attackiter 100 --poisonkey 2000000000 --budget 0.01 --restarts 1 --pbatch 512 --ensemble 1 --name ablation_no_aug_no_aug --save limited
# Success, Aprox 50min

python3 brew_poison.py  --recipe poison-frogs --net ResNet18 --vruns 1 --attackiter 100 --poisonkey 2000000000 --budget 0.01 --restarts 1 --pbatch 512 --ensemble 1 --name ablation_no_aug_no_aug --save limited
# Fail, Aprox 50min 

python3 brew_poison.py  --recipe metapoison --net ResNet18 --vruns 1 --attackiter 100 --poisonkey 2000000000 --budget 0.01 --restarts 1 --pbatch 512 --ensemble 1 --name ablation_no_aug_no_aug --save limited
# Fail, Aprox 50min 
