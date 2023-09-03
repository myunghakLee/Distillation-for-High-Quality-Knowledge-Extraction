# Distillation for High-Quality Knowledge Extraction via Explainable Oracle Approach


## Requirements
- PyTorch (> 1.2.0)
- torchvision
- numpy

## Training
```
python main.py --data_dir ./data --data CIFAR100 --model_t resnet20 --model_s resnet20 --lrp_temperature 0.5 --temperature 1.5 --lrp_gamma 0.2 --ce_weight 1.0 --beta 100 --alpha 1.0 --weight_decay 0.0001 --save_dir_name MyTest
```
