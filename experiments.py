import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import wandb
import datetime

from models.model import *
from models.vggmodel import *
from other_utils.utils import *
from other_utils.dropout import active_clients


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='fmnist', help='dataset used for training')
    parser.add_argument('--partition', type=str, default='noniid-#label2', help='the data partitioning strategy')
    parser.add_argument('--n_sample', type=int, default=2000,  help='number of data samples on each device')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--lr_decay', type=float, default=0.99, help='learning rate decay factor (default: 0.95)')
    parser.add_argument('--lr_decay_interval', type=int, default=1, help='learning rate decay interval (default: 1)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_clients', type=int, default=2,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg', choices=['fedavg', 'mifa', 'mimic'])
    parser.add_argument('--comm_iter', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.05, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')

    # dropout setup
    parser.add_argument('--active_mode', type=str, default="random", help='How do clients participate', choices=["random", "static", "tau"])
    parser.add_argument('--tau_max', type=int, default=10, help='Maximal delay')
    parser.add_argument('--sample', type=float, default=1, help='Participate ratio for each communication iter')

    args = parser.parse_args()
    args.tau_list = None
    return args


def init_nets(n_clients, model_type):

    if model_type == "vgg11":
        # for cifar
        net = vgg11()
    elif model_type == "cnn":
        # for fmnist
        net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
    else:
        print(f"Not supported {model_type} yet")
        exit(1)
        
    net_weihts = {net_i: None for net_i in range(n_clients)}
    for net_i in range(n_clients):
        net_weihts[net_i] = copy.deepcopy(net.state_dict())

    return net, net_weihts


def train_net(net_id, net, train_dataloader, epochs, lr, device="cuda:0"):
    logger.info('Training network %s' % str(net_id))
    net.to(device)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    cum_grad = dict()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

            for k, p in net.named_parameters():
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if k not in cum_grad.keys():
                    cum_grad[k] = torch.clone(d_p).detach()
                    cum_grad[k].mul_(lr)
                else:
                    cum_grad[k].add_(d_p, alpha=lr)

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f (cnt: %d)' % (epoch, epoch_loss, cnt))

    logger.info(' ** Training complete **')
    return cum_grad


def local_train_net(net, selected, args, net_dataidx_map, device="cuda:0"):
    update_list = [None] * args.n_clients
    global_para = copy.deepcopy(net.state_dict())
    net.train()
    net.to(device)

    for idx in selected:
        dataidxs = net_dataidx_map[idx]
        net.load_state_dict(global_para)

        logger.info("Training network %s. n_training: %d" % (str(idx), len(dataidxs)))

        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, dataidxs=dataidxs)
        n_epoch = args.epochs

        update = train_net(idx, net, train_dl_local, n_epoch, args.current_lr, device=device)
        update_list[idx] = update
            
    return update_list


def setup_logger(args):
    dropout_text = args.tau_max if 'tau' in args.active_mode else args.sample
    mkdirs(args.logdir)
    if args.log_file_name is None:
        argument_path='arg_%s_%s_%s_%s_%s_%s_%s.json' % (args.alg, args.dataset, args.partition, args.model, args.active_mode, dropout_text, datetime.datetime.now().strftime("%m%d-%H%M%S"))
        args.log_file_name='log_%s_%s_%s_%s_%s_%s_%s' % (args.alg, args.dataset, args.partition, args.model, args.active_mode, dropout_text, datetime.datetime.now().strftime("%m%d-%H%M%S"))
    else:
        argument_path=args.log_file_name+'.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f: # argument
        json.dump(str(args), f)
    log_path=args.log_file_name+'.log'  # log_path

    # print in the screen and save to log_file
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)-8s %(message)s') # print in the screen
    handler = logging.FileHandler(os.path.join(args.logdir, log_path)) # save to the file
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return dropout_text


if __name__ == '__main__':
    args = get_args()
    device = torch.device(args.device)
    
    # setup logger file
    dropout_text = setup_logger(args)
    
    # wandb init
    wandb.init(
        project="Client-dropout",
        name="{}-{}-{}tot-{}({})-{}E".format(
            args.alg, args.partition, args.n_clients, args.active_mode, dropout_text, args.epochs),
        config=args
    ) 
    logger.info("File name: %s" % args.log_file_name)

    # set seed
    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    logger.info("Partitioning data (%s)" % args.partition)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_clients, args.n_sample, beta=args.beta)
    
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                        args.datadir,
                                                                                        args.batch_size) 
    
    logger.info("Initializing nets")
    net, net_weights = init_nets(args.n_clients, args.model)
    net.to(device)
    global_para = net.state_dict()

    if args.alg == 'fedavg':
        for iteration in range(0, args.comm_iter):
            logger.info("In comm iter:" + str(iteration))

            if iteration == 0:
                selected = [i for i in range(args.n_clients)] 
            else:
                selected = active_clients(args, iteration) 


            args.current_lr = max(args.lr * args.lr_decay ** int(iteration//args.lr_decay_interval), 1e-5)
            update_list = local_train_net(net, selected, args, net_dataidx_map, device=device)
            
            # aggregation weight
            agg_weight = [float(1/len(selected)) for _ in selected]
            
            # Aggregation
            for i in range(len(selected)):
                idx = selected[i]
                cum_grad = update_list[idx]
                for key, grad in cum_grad.items():
                    global_para[key].sub_(grad, alpha=agg_weight[i])
            
            net.load_state_dict(global_para)
            net.to(device)

            train_acc, train_loss = compute_accuracy(net, train_dl_global, device=device)
            test_acc, test_loss, conf_matrix = compute_accuracy(net, test_dl_global, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Test loss: %f' % test_loss)
            wandb.log({"Iteration": iteration, "Train/Acc": train_acc, "Train/Loss": train_loss, \
                            "Test/Acc": test_acc, "Test/Loss": test_loss})
    
    elif args.alg == 'mimic':
        last_grad_table = [None for _ in range(args.n_clients)]
        last_iter_table = [0 for _ in range(args.n_clients)]
        global_grad_table = [{} for _ in range(args.n_clients)]
        client_idxs = [i for i in range(args.n_clients)] 
    
        for iteration in range(0, args.comm_iter):
            old_para = copy.deepcopy(global_para)
            logger.info("In comm iter:" + str(iteration))
            if iteration == 0:
                selected = [i for i in range(args.n_clients)] 
            else:
                selected = active_clients(args, iteration) 
            drop_idxs = list(set(client_idxs) - set(selected))

            args.current_lr = max(args.lr * args.lr_decay ** int(iteration//args.lr_decay_interval), 1e-5)
            update_list = local_train_net(net, selected, args, net_dataidx_map, device=device)
            
            # aggregation weight
            w = float(1/len(selected))
            if iteration > 0:
                agg_weight = [w for _ in range(args.n_clients)]

            # global aggregation
            for key, grad_g in global_para.items():
                update_modified = torch.zeros_like(grad_g)
                for idx in selected:
                    update_modified.add_(update_list[idx][key], alpha=w)  # NOTE: This is gradient * learning rate
                    if iteration > 0:
                        update_modified.sub_(last_grad_table[idx][key], alpha=agg_weight[idx]) 
                        update_modified.add_(global_grad_table[idx][key], alpha=agg_weight[idx])
                global_para[key].sub_(update_modified)
                
            # update list
            for idx in selected:
                last_grad_table[idx] = update_list[idx] # a dict
                last_iter_table[idx] = iteration

            # global accumulation
            for key, grad in global_para.items():
                for idx in selected:
                    global_grad_table[idx].update( {key: torch.sub(old_para[key], global_para[key]) } ) 
            
            # update global model
            net.load_state_dict(global_para)

            train_acc, train_loss = compute_accuracy(net, train_dl_global, device=device)
            test_acc, test_loss, conf_matrix = compute_accuracy(net, test_dl_global, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Test loss: %f' % test_loss)
            wandb.log({"Iteration": iteration, "Train/Acc": train_acc, "Train/Loss": train_loss, \
                            "Test/Acc": test_acc, "Test/Loss": test_loss})

    elif args.alg == 'mifa':
        last_grad_table = [None for _ in range(args.n_clients)]
        for iteration in range(0, args.comm_iter):
            logger.info("In comm iter:" + str(iteration))

            if iteration == 0:
                selected = [i for i in range(args.n_clients)] 
            else:
                selected = active_clients(args, iteration) 
        
            args.current_lr = max(args.lr * args.lr_decay ** int(iteration//args.lr_decay_interval), 1e-5)
            update_list = local_train_net(net, selected, args, net_dataidx_map, device=device)

            # aggregation weight
            agg_weight = [round(1.0/args.n_clients, 4) for _ in range(args.n_clients)]

            # aggregate and update global model
            for idx in range(args.n_clients):
                if idx in selected:
                    net_update = update_list[idx]
                else:
                    net_update = last_grad_table[idx]
                for key in global_para.keys(): 
                    if ('running' in key) or ('num_batches_tracked' in key):
                        continue
                    print(global_para[key].device, net_update[key].device, agg_weight[idx])
                    global_para[key].sub_(net_update[key], alpha=agg_weight[idx])
            
            for idx in selected:
                last_grad_table[idx] = update_list[idx] # a dict
                
            net.load_state_dict(global_para)
            net.to(device)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            train_acc, train_loss = compute_accuracy(net, train_dl_global, device=device)
            test_acc, test_loss, conf_matrix = compute_accuracy(net, test_dl_global, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            wandb.log({"Iteration": iteration, "Train/Acc": train_acc, "Train/Loss": train_loss, \
                            "Test/Acc": test_acc, "Test/Loss": test_loss})
