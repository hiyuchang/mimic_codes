import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from models.model import *
from other_utils.datasets import CIFAR10_truncated, FashionMNIST_truncated


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass
    

def load_fmnist_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = FashionMNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = FashionMNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=False, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=False, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target
    
    return (X_train, y_train, X_test, y_test)

def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    logger.info('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts

def partition_data(dataset, datadir, logdir, partition, n_clients, n_sample, beta=0.4):

    if dataset == 'fmnist':
        X_train, y_train, X_test, y_test = load_fmnist_data(datadir)
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    else:
        print("WARNING: No such dataset!")
        import os
        os.exit(1)

    n_train = y_train.shape[0]
    
    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_clients)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_clients)}
        assert len(batch_idxs[0]) <= n_sample

    elif partition == "noniid-labeldir":
        min_size = 0
        min_require_size = 10
        K = 10  # classes
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2

        N = y_train.shape[0]
        np.random.seed(2022)
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_clients)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_clients))
                # logger.info("proportions1: ", proportions)
                # logger.info("sum pro1:", np.sum(proportions))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_clients) for p, idx_j in zip(proportions, idx_batch)])
                # logger.info("proportions2: ", proportions)
                proportions = proportions / proportions.sum()
                # logger.info("proportions3: ", proportions)
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # logger.info("proportions4: ", proportions)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_clients <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break

        for j in range(n_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
            if n_sample > 0 and n_sample < len(net_dataidx_map[j]):
                np.random.shuffle(net_dataidx_map[j])
                net_dataidx_map[j] = net_dataidx_map[j][:n_sample]

    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        num = eval(partition[13:])
        n_sample_c = int(n_sample/num)  # maximal numbers of samples per class
        K = 10
        if num == 10:
            net_dataidx_map = {i:np.ndarray(0,dtype=np.int64) for i in range(n_clients)}
            for i in range(10):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,n_clients)
                for j in range(n_clients):
                    np.random.shuffle(split[j])
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[j][:n_sample_c])
        else:
            times=[0 for i in range(10)]
            contain=[]
            for i in range(n_clients):
                current=[i%K]
                times[i%K]+=1
                j=1
                while (j<num):
                    ind=random.randint(0,K-1)
                    if (ind not in current):
                        j=j+1
                        current.append(ind)
                        times[ind]+=1
                contain.append(current)
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_clients)}
            for i in range(K):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,times[i])
                ids=0
                for j in range(n_clients):
                    if i in contain[j]:
                        np.random.shuffle(split[ids])  # TODO: check
                        net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids][:n_sample_c])
                        ids+=1

    else:
        print("WARNING: No such partition!")
        import os
        os.exit(1)

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cuda:0"):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total, loss_sum = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.to(device), target.to(device,dtype=torch.int64)
            out = model(x)
            loss = criterion(out, target)
            _, pred_label = torch.max(out.data, 1)

            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()
            loss_sum += loss.item() * x.data.size()[0]

            pred_labels_list = np.append(pred_labels_list, pred_label.to("cpu").numpy())
            true_labels_list = np.append(true_labels_list, target.data.to("cpu").numpy())

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct/float(total), loss_sum/float(total), conf_matrix

    return correct/float(total), loss_sum/float(total)


def get_dataloader(dataset, datadir, train_bs, test_bs=32, dataidxs=None):
    if dataset == 'fmnist':
        dl_obj = FashionMNIST_truncated
        transform_train = transforms.Compose([
            transforms.ToTensor()])
        transform_test = transforms.Compose([
            transforms.ToTensor()])

    elif dataset == 'cifar10':
        dl_obj = CIFAR10_truncated

        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    else:
        print("WARNING: No such dataset!")
        import os
        os.exit(1)
        
    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

    train_dl = DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl, train_ds, test_ds

