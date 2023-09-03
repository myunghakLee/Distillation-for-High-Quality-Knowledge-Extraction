import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torch

from time import time
import numpy as np
import argparse
import datetime
import random
import shutil
import json
import copy
import os

from dataset import create_loader
from KD import DistillKL
import models
import utils

def main():
    utils.set_seed()
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--data', type=str, default='cifar10')

    # Training
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--schedule', default=[150, 180, 210], type=int, nargs='+')
    parser.add_argument('--epoch', default=240, type=int)

    # KD option
    parser.add_argument('--alpha', type=float, default=0.9, help='weight for KD (Hinton)')
    parser.add_argument('--beta', type=float, default=200)
    parser.add_argument('--ce_weight', type=float, default=1.0)
    parser.add_argument('--lrp_temperature', type=float, default=4)
    parser.add_argument('--model_t', type=str, default='')
    parser.add_argument('--model_s', type=str, default='')
    parser.add_argument('--lrp_gamma', type=float, default=1.0, help='intensity of GI(gradient*input)')
    parser.add_argument('--temperature', default=4, type=float)
    parser.add_argument('--save_dir_name', type=str, default='')
    parser.add_argument('--test_student', action='store_true', help='Test before distillation')
    parser.add_argument('--test_teacher', action='store_true', help='Test before distillation')

    args = parser.parse_args()
    args_dict = copy.deepcopy(vars(args))
    device = torch.device("cuda")


    train_loader, test_loader, args.num_classes, args.image_size = \
        create_loader(args.batch_size, args.data_dir, args.data)

    model_t = models.__dict__[args.model_t](num_classes=args.num_classes)
    model_s = models.__dict__[args.model_s](num_classes=args.num_classes)

    if args.data == "CIFAR100":
        trained_dir = f"teacher_models/CIFAR100/{args.model_t}/model.pth"

    elif args.data == "CIFAR10":
        trained_dir = f"teacher_models/CIFAR10/{args.model_t}/best.pth"

    model_t.load_state_dict(torch.load(trained_dir))

    model_t.eval()
    model_s.eval()

    refiner = utils.Refiner(teacher=model_t, lrp_gamma=args.lrp_gamma)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = DistillKL(args.temperature)

    criterion = nn.ModuleList([])
    criterion.append(criterion_ce)
    criterion.append(criterion_kl)

    optimizer = optim.SGD(
        trainable_list.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    module_list.append(model_t)

    module_list = module_list.to(device)
    criterion.cuda()


    if args.test_teacher:
        print(f"TEST TEACHER MODEL[{args.model_t}]!!")

        test_acc1, test_acc5 = utils.test(model_t, test_loader, device)
        print("acc1 : ", test_acc1.item())
        print("acc5 : ", test_acc5.item())

    if args.test_student:
        print(f"TEST STUDENT MODEL[{args.model_s}]!!")
        test_acc1, test_acc5 = utils.test(model_s, test_loader, device)
        print("acc1 : ", test_acc1.item())
        print("acc5 : ", test_acc5.item())

    save_dir = utils.get_save_dir(args)
    print("save_dir : ", save_dir)
    os.makedirs(save_dir, exist_ok=True)

    train_losses = []
    train_acc1s, train_acc5s = [], []
    test_acc1s, test_acc5s = [], []

    best_acc = 0

    utils.set_seed()
    for epoch in range(1, args.epoch + 1):
        s2 = time()
        utils.adjust_learning_rate(optimizer, epoch, args)
        train_loss, train_acc1, train_acc5, train_acc1_T = utils.train_kd(
            module_list, optimizer, criterion, train_loader, device, refiner, args
        )
        model_s = model_s.eval()
        test_acc1, test_acc5 = utils.test(model_s, test_loader, device)
        model_s = model_s.train()
        print(
            "Epoch: {0:>3d} |Train Loss: {1:>2.4f} |Test Top1: {2:.4f} |Test Top5: {3:.4f}".format(
                epoch, train_loss, test_acc1, test_acc5
            )
        )

        train_losses.append(train_loss)
        train_acc1s.append(train_acc1.item())
        train_acc5s.append(train_acc5.item())
        test_acc1s.append(test_acc1.item())
        test_acc5s.append(test_acc5.item())

        if best_acc < test_acc1:
            utils.save_model(save_dir, module_list, args, train_losses, train_acc1s, train_acc5s, test_acc1s, test_acc5s, train_acc1_T, args_dict)

        print(f"time consume : {round(time() - s2, 3)}\n")

    
    utils.save_model(save_dir, module_list, args, train_losses, train_acc1s, train_acc5s, test_acc1s, test_acc5s, train_acc1_T, args_dict, option="last")

if __name__ == '__main__':
    main()