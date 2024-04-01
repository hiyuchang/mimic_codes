import os
import numpy as np
from scipy.stats import halfnorm


############ KEY FUNCTION ###########

def active_clients(args, iteration):
    n_clients = args.n_clients
    active_mode = args.active_mode
    sample = args.sample
    tau_max = args.tau_max
    seed = args.init_seed + iteration

    if active_mode == "random":
        np.random.seed(iteration+seed)
        # max_dropout = 1 - sample
        prob_list = []
        for i in range(args.n_clients):
            tmp = np.random.uniform(0.1, 1)
            prob_list.append(tmp)
        p = [item/sum(prob_list) for item in prob_list]
        selected = np.random.choice(n_clients, int(n_clients * sample), replace=False, p=p)
    
    elif active_mode == "static":
        # tau_max = max {1/p_min}
        # NOTE: We use "sample" represents the minimal participation probability
        np.random.seed(iteration+seed)
        p_min = sample - (1-sample)/4 # Virtual!
        p = [p_min + (i%5+1) * (1-sample)/4 for i in range(n_clients)] # NOTE server 2: Five levels (sample, sample+x, ..., sample+4x=1)
        selected = []
        for i in range(n_clients):
            coin = np.random.rand()  # From mifa
            if coin < p[i]:
                selected.append(i)
        if len(selected) < 2:
            selected = np.random.choice(n_clients, 2, replace=False)
    
    elif active_mode == "tau":
        # half-normal probability distribution
        if args.tau_list == None:
            x = halfnorm.rvs(loc=0, scale=3, size=n_clients)
            args.tau_list = [min(round(item), tau_max) for item in x]
            # print("tau list is ", args.tau_list)
        selected = []
        for i in range(n_clients):
            if args.tau_list[i] == 0:
                selected.append(i)
            elif iteration % args.tau_list[i] == 0:
                selected.append(i)
        if len(selected) < 2:
            np.random.seed(iteration)
            selected = np.random.choice(n_clients, 2, replace=False)
            np.random.seed(seed) # change back
        args.tau_max = tau_max
    
    else:
        print("Warning: Invalid sample mode!")
        os.exit(1)

    return selected

