"""
2021/3/16
train forgetting events
"""
import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils

from forgetting_events import ForgettingEvents
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('-project_name', type=str, help='project name', default='forgetting_events')
parser.add_argument('-dataset_path', type=str, help='relative path of dataset', default='../dataset')
parser.add_argument('-batch_size', type=int, help='batch size', default=64)
parser.add_argument('-lr', type=float, help='learning rate', default=0.01)
parser.add_argument('-epochs', type=int, help='training epochs', default=100)
parser.add_argument('-log_dir', type=str, help='log dir', default='output')
args = parser.parse_args()


def create_dataloader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    # load dataset
    train_set = datasets.MNIST(
        args.dataset_path, train=True, transform=transform, download=True)
    test_set = datasets.MNIST(
        args.dataset_path, train=False, transform=transform, download=False)

    # split train set into train-val set
    train_set, val_set = torch.utils.data.random_split(train_set, [
        50000, 10000])

    # generate DataLoader
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False)

    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_set, val_set, test_set


def train(model, train_loader, optimizer, epoch, device, train_loss_lst, train_acc_lst, fe):
    model.train()  # Set the module in training mode
    correct = 0
    train_loss = 0
    for batch_idx, (inputs, labels, indices) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        pred_labels = outputs.max(1, keepdim=True)[1]
        correct += pred_labels.eq(labels.view_as(pred_labels)).sum().item()

        # record forgetting
        fe.record_forgetting(pred_labels, labels, indices)

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

    train_loader, val_loader, test_loader, train_set, val_set, test_set = create_dataloader()  # get data loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MNISTNet().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    train_loss_lst, val_loss_lst = [], []
    train_acc_lst, val_acc_lst = [], []

    fe = ForgettingEvents(len(train_set))

    # main loop(train,val,test)
    for epoch in range(args.epochs):
        train_loss_lst, train_acc_lst = train(model, train_loader, optimizer, epoch, device, train_loss_lst,
                                              train_acc_lst, fe)
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
    plt.show()
    plt.close(fig)

    # calculate each sample's forgetting times
    ranks = [(idx, forgetting) for idx, forgetting in enumerate(fe.forgetting_times.tolist())]
    ranks.sort(key=lambda x: x[1], reverse=True)
    print('top 100 forgetting times:', ranks[:100])

    # sub noisy set and clean set
    noisy_indices = [item[0] for item in ranks[:100]]  # subtract top 100 noisy data from dataset
    noisy_set = Subset(train_set, noisy_indices)  # noisy set

    # save noisy samples
    noisy_loader = DataLoader(noisy_set, batch_size=len(noisy_set), shuffle=False)
    inputs, labels = next(iter(noisy_loader))
    fig = plt.figure()
    inputs = inputs[:100].detach().cpu()  # convert to cpu
    grid = utils.make_grid(inputs)
    print('Noisy labels:', labels)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.savefig(os.path.join(output_path, 'noisy.png'))
    plt.close(fig)

    # save model
    torch.save(model.state_dict(), os.path.join(output_path, args.name + ".pth"))
