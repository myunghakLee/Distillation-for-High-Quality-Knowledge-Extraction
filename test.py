from dataset import create_loader
import argparse
import models
import torch
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--data", type=str, default="cifar10")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--model", type=str, default="")

    args = parser.parse_args()
    device = torch.device("cuda")

    _, test_loader, args.num_classes, args.image_size = create_loader(
        args.batch_size, args.data_dir, args.data
    )

    model = models.__dict__[args.model](num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    model.eval()
    
    test_acc1, test_acc5 = utils.test(model, test_loader, device)
    print(args.model_path)
    print("acc1 : ", test_acc1.item())
    print("acc5 : ", test_acc5.item())


if __name__ == "__main__":
    main()
