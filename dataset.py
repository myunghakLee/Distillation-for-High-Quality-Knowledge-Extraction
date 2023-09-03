from torchvision import transforms, datasets
import torch

import numpy as np
import random
import os

def create_loader(batch_size, data_dir, data):
    data_dir = os.path.join(data_dir, data)
    if data == 'CIFAR100':
        img_size = 32
        num_classes = 100

        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        transform_train = transforms.Compose(
            [transforms.RandomCrop(img_size, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                normalize])
        transform_test = transforms.Compose(
            [transforms.ToTensor(), normalize])

        trainset = datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, pin_memory=True)

        return train_loader, test_loader, num_classes, img_size

    elif data == 'CIFAR10':
        img_size = 32
        num_classes = 10
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2470, 0.2435, 0.2616])
        transform_train = transforms.Compose(
            [transforms.RandomCrop(img_size, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                normalize])
        transform_test = transforms.Compose(
            [transforms.ToTensor(), normalize])
        
        trainset = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, pin_memory=True)

        return train_loader, test_loader, num_classes, img_size

    elif data.lower() == "tiny_imagenet":
        img_size = 64
        num_classes = 200

        normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                         std=[0.2764, 0.2689, 0.2816])
        
        transform_train = transforms.Compose(
            [transforms.RandomCrop(img_size, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                normalize])
        transform_test = transforms.Compose(
            [transforms.ToTensor(), normalize])

        trainset = datasets.ImageFolder(root=os.path.join(
            data_dir, 'train'), transform=transform_train)
        testset = datasets.ImageFolder(root=os.path.join(
            data_dir, 'val'), transform=transform_test)

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, pin_memory=True)

        return train_loader, test_loader, num_classes, img_size

