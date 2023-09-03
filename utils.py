import torch.backends.cudnn as cudnn
import torch

from tqdm import tqdm
import numpy as np
import random
import datetime
import json

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)


def get_save_dir(args):
    save_dir = f"student_models/{args.data}/{args.save_dir_name}/{args.model_t}_2_{args.model_s}_{args.lrp_gamma}_{args.ce_weight}______"

    d = datetime.datetime.now()
    save_dir += (
        str(d)
        .replace("-", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace(".", "_")[5:-5]
        + "/"
    )
    return save_dir


def save_model(save_dir, module_list, args, train_losses, train_acc1s, train_acc5s, test_acc1s, test_acc5s, train_acc1_T, args_dict, option="best"):
    model_save_dir = f"{save_dir}/{option}.pth"
    torch.save(module_list[0].state_dict(), model_save_dir)

    print("max train accuracy : ", max(train_acc1s), max(train_acc5s))
    print("max test accuracy : ", max(test_acc1s), max(test_acc5s))
    with open(f"{save_dir}/{args.model_t}_2_{args.model_s}.json", "w") as f:
        json.dump(
            {
                "train_losses": train_losses,
                "train_acc1s": train_acc1s,
                "train_acc5s": train_acc5s,
                "test_acc1s": test_acc1s,
                "test_acc5s": test_acc5s,
                "train_acc1_T": train_acc1_T,
                "max_test_acc": [max(test_acc1s), max(test_acc5s)],
                "args": args_dict,
            },
            f,
            indent=4,
        )


def train_kd(module_list, optimizer, criterion, train_loader, device, refiner, args):
    module_list[0].train()
    module_list[1].eval()

    model_s = module_list[0]
    model_t = module_list[1]

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    criterion_ce, criterion_kl = criterion

    T_correct = 0
    all_data = 0
    model_t.eval()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        input_lrp = refiner.get_refined_image(inputs, targets)

        with torch.no_grad():
            output_t = model_t(input_lrp)

        _, output_s = model_s(inputs, is_feat=True)

        loss_ce = criterion_ce(output_s, targets)  # classification loss
        loss_kl = criterion_kl(output_s, output_t)  # Hinton loss

        loss = args.ce_weight * loss_ce + args.alpha * loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(output_s, targets, topk=(1, 5))

        batch_size = targets.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc1, batch_size)
        top5.update(acc5, batch_size)

        T_correct += sum(targets == torch.argmax(output_t, dim=1))
        all_data += len(targets)

    acc_T = (T_correct / all_data).item()
    return losses.avg, top1.avg, top5.avg, acc_T


def test(model, test_loader, device):
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

            batch_size = targets.size(0)
            top1.update(acc1, batch_size)
            top5.update(acc5, batch_size)
    model.train()
    return top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(1.0 / batch_size))
        return res

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch in args.schedule:
        args.lr = args.lr * args.lr_decay
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr
            print(param_group["lr"])

class Refiner:
    def __init__(self, teacher, lrp_gamma=1.0):
        super(Refiner)
        self.teacher = teacher
        self.lrp_gamma = lrp_gamma
        self.criterion_ce = torch.nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(self.teacher.parameters(), lr=0)

    def get_refined_image(self, img, label):
        img.requires_grad = True
        output = self.teacher(img)
        loss = self.criterion_ce(output, label)
        loss.backward()

        return self.get_adversarial_img_v1_abs(img) # refined image

    def get_adversarial_img_v1_abs(self, img, sine=1):
        perturbation = img.grad * torch.abs(img.detach())
        output_img = img - perturbation * sine * self.lrp_gamma
        return output_img    