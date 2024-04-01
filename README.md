# MimiC

This is the code of paper [MimiC: Combating Client Dropouts in Federated Learning by Mimicking Central Updates](https://arxiv.org/pdf/2306.12212.pdf).

This repo contains the codes for our paper **MimiC** published in *Transactions on Mobile Computing*, with an introduction in [link](https://mp.weixin.qq.com/s/7M-OLONznfRvQf-FPIKuIw).


## Usage

### Prepare

- Create the enviroment `conda create -n mimic python=3.7`.

- Install neccessary packages `pip install requirement.txt`.

- Login in your wandb account (for visualization). Or just comment all codes related to wandb.

- Download data to `./data`.

### Run!

Here is one example to run this code:
```
python experiments.py \
	--model=cnn \
	--dataset=fmnist \
	--alg=mimic \
    	--lr=0.01 \
    	--lr_decay=0.95 \
	--n_clients=30 \
	--comm_iter=300 \
	--partition='noniid-#label2' \
	--device='cuda:0' \
	--active_mode='tau' \
	--tau_max=20
```

## Client dropout setup
We simulate the following setups for the 
- **Bounded consecutive dropout iterations**: Each client becomes active every $\tau_{\max}(i)$ iterations. Use the parameter `--active_mode='tau'` and `--tau_max=20`.
- **Static active probabilities**: All clients have the same and fixed active probability. Use `--active_mode='individual'` and `--sample=0.8`.
- **Time-varying active probabilities**: The server waits for $P \times N$ active clients by sampling clients without replacement. Use `--active_mode='random'` and `--sample=0.1`.


## Other parameters

| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model` | The model architecture. Options: `cnn`, `vgg11`. Default = `cnn`. |
| `dataset`      | Dataset to use. Options: `fmnist`, `cifar10`. Default = `fmnist`. |
| `alg` | The training algorithm. Options: `fedavg`, `mifa`, `mimic`. Default = `mimic`. |
| `epochs` | Number of local training epochs, default = `5`. |
| `n_clients` | Number of clients, default = `30`. |
| `comm_iter`    | Number of communication rounds to use, default = `300`. |
| `partition`    | The partition way. Options: `noniid-labeldir`, `noniid-#label1` (or 2, 3, ..., which means the fixed number of labels each party owns). Default = `noniid-#label2` |
| `beta` | The concentration parameter of the Dirichlet distribution for heterogeneous partition, default = `0.5`. |
| `device` | Specify the device to run the program, default = `cuda:0`. |



## Citation

If you find this repository useful, please cite our paper:

```
@article{sun2023mimic,
  title={Mimic: Combating client dropouts in federated learning by mimicking central updates},
  author={Sun, Yuchang and Mao, Yuyi and Zhang, Jun},
  journal={IEEE Transactions on Mobile Computing},
  year={2023},
  publisher={IEEE}
}
```

## Acknowledgement

The codes are revised based on the benchmark repo: [NIID-Bench](https://github.com/Xtra-Computing/NIID-Bench).
