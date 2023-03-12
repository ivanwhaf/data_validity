"""
2021/3/8
train mnist Mean-Teacher
"""
import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from models import *

parser = argparse.ArgumentParser()
parser.add_argument('-project_name', type=str, help='project name', default='pretrain_ensemble_models_mnist')
parser.add_argument('-dataset_path', type=str, help='relative path of dataset',
                    default='H:\Data\datasets')  # config your dataset path here!
parser.add_argument('-batch_size', type=int, help='batch size', default=64)
parser.add_argument('-lr', type=float, help='learning rate', default=0.01)
parser.add_argument('-epochs', type=int, help='training epochs', default=100)
parser.add_argument('-log_dir', type=str, help='log dir', default='output')
parser.add_argument('-seed', type=int, help='numpy and pytorch seed', default=0)
args = parser.parse_args()


def create_dataloader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    # load dataset
    train_set = datasets.MNIST(args.dataset_path, train=True, transform=transform, download=True)
    test_set = datasets.MNIST(args.dataset_path, train=False, transform=transform, download=False)
    sub_set1, sub_set2, sub_set3 = torch.utils.data.random_split(train_set, [10000, 20000, 50000])

    # split train set into train-val set
    train_set, val_set = torch.utils.data.random_split(train_set, [50000, 10000])

    # generate DataLoader
    train_loader1 = DataLoader(sub_set1, batch_size=args.batch_size, shuffle=True)
    train_loader2 = DataLoader(sub_set2, batch_size=args.batch_size, shuffle=True)
    train_loader3 = DataLoader(sub_set3, batch_size=args.batch_size, shuffle=True)
    train_loaders = [train_loader1, train_loader2, train_loader3]

    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    return train_loaders, val_loader, test_loader


def train(model, train_loader, optimizer, epoch, device, train_loss_lst, train_acc_lst):
    model.train()  # Set the module in training mode
    correct = 0
    train_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        pred = outputs.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # show batch0 dataset
        if batch_idx == 0 and epoch == 0:
            fig = plt.figure()
            inputs = inputs.detach().cpu()  # convert to cpu
            grid = utils.make_grid(inputs)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.savefig(os.path.join(output_path, 'batch0.png'))
            plt.close(fig)

        # print train loss and accuracy
        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]  Loss: {:.6f}'
                  .format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))

    # record loss and accuracy
    train_loss /= len(train_loader)  # must divide iter num
    train_loss_lst.append(train_loss)
    train_acc_lst.append(correct / len(train_loader.dataset))
    return train_loss_lst, train_acc_lst


def validate(model, val_loader, device, val_loss_lst, val_acc_lst):
    model.eval()  # Set the module in evaluation mode
    val_loss = 0
    correct = 0
    # no need to calculate gradients
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            criterion = nn.CrossEntropyLoss()
            val_loss += criterion(output, target).item()
            # val_loss += F.nll_loss(output, target, reduction='sum').item()

            # find index of max prob
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    # print val loss and accuracy
    val_loss /= len(val_loader)
    print('\nVal set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'
          .format(val_loss, correct, len(val_loader.dataset),
                  100. * correct / len(val_loader.dataset)))

    # record loss and accuracy
    val_loss_lst.append(val_loss)
    val_acc_lst.append(correct / len(val_loader.dataset))
    return val_loss_lst, val_acc_lst


def test(model, test_loader, device):
    model.eval()  # Set the module in evaluation mode
    test_loss = 0
    correct = 0
    # no need to calculate gradients
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            criterion = nn.CrossEntropyLoss()
            test_loss += criterion(output, target).item()

            # find index of max prob
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    # print test loss and accuracy
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    set_seed(args.seed)  # set seed

    # create output folder
    output_path = os.path.join(args.log_dir, args.project_name)
    os.makedirs(output_path)

    train_loaders, val_loader, test_loader = create_dataloader()  # get data loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = [MNISTNet().to(device), MNISTNet().to(device), MNISTNet().to(device)]

    train_loss_lst, val_loss_lst = [], []
    train_acc_lst, val_acc_lst = [], []

    for idx, model in enumerate(models):
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

        train_loader = train_loaders[idx]

        for epoch in range(args.epochs):
            train_loss_lst, train_acc_lst = train(model, train_loader, optimizer,
                                                  epoch, device, train_loss_lst, train_acc_lst)

            val_loss_lst, val_acc_lst = validate(model, val_loader, device, val_loss_lst, val_acc_lst)

            # modify learning rate
            if epoch in [40, 60, 80]:
                args.lr *= 0.1
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

        test(model, test_loader, device)

        # plot loss and accuracy curve
        fig = plt.figure('Loss and acc')
        plt.plot(range(args.epochs), train_loss_lst, 'g', label='train loss')
        plt.plot(range(args.epochs), val_loss_lst, 'k', label='val loss')
        plt.plot(range(args.epochs), train_acc_lst, 'r', label='train acc')
        plt.plot(range(args.epochs), val_acc_lst, 'b', label='val acc')
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(output_path, 'loss_acc.png'))
        # plt.show()
        plt.close(fig)

        # save model
        torch.save(model.state_dict(), os.path.join(output_path, "model" + str(idx) + ".pth"))
