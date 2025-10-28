import os, re, json, torch
import os.path as osp
import numpy as np
from Bio.Cluster import kcluster


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_json_file(file_path):
    with open(file_path, "r") as f:
        s = f.read()
        s = re.sub('\s',"", s)
    return json.loads(s)


def load_best_model(args):
    if (args.load_first_year and args.year <= args.begin_year +  1) or args.train == 0:  # Determine whether to load the first year's model
    # if args.load_first_year:  # Determine whether to load the first year's model
        load_path = args.first_year_model_path  # Set the loading path to the first year model path
        loss = load_path.split("/")[-1].replace(".pkl", "")  # Get the model file name and remove the extension
    else:
        loss = []
        for filename in os.listdir(osp.join(args.model_path, args.logname+"-"+str(args.seed), str(args.year-1))):  # Traverse the files under the model path of the previous year and get all loss values
            loss.append(filename[0:-4])
        loss = sorted(loss)
        load_path = osp.join(args.model_path, args.logname+"-"+str(args.seed), str(args.year-1), loss[0]+".pkl")  # Set the loading path to the model file corresponding to the minimum loss value
        
    args.logger.info("[*] load from {}".format(load_path))  # Recording Load Path
    state_dict = torch.load(load_path, map_location=args.device)["model_state_dict"]  # Loading the model state dictionary
    
    model = args.methods[args.method](args)  # Initialize the model
    
    if args.method == 'EAC' or args.method == 'DepCL':
        if args.year == args.begin_year:
            model.expand_adaptive_params(args.base_node_size)
        else:
            for idx, _ in enumerate(range(args.year - args.begin_year)):
                model.expand_adaptive_params(args.graph_size_list[idx])
    
    if args.method == 'Universal' and args.use_eac == True:
        if args.year == args.begin_year:
            model.expand_adaptive_params(args.base_node_size)
        else:
            for idx, _ in enumerate(range(args.year - args.begin_year)):
                model.expand_adaptive_params(args.graph_size_list[idx])
    
    model.load_state_dict(state_dict)  # Load the state dictionary into the model
    model = model.to(args.device)  # Move the model to the specified device
    return model, loss[0]  # Returns the model and the minimum loss value


def load_test_best_model(args):
    # if args.load_first_year and args.year < args.begin_year +  1:  # Determine whether to load the first year's model
    #     load_path = args.first_year_model_path  # Set the loading path to the first year model path
    #     loss = load_path.split("/")[-1].replace(".pkl", "")  # Get the model file name and remove the extension
    # else:
    loss = []
    for filename in os.listdir(osp.join(args.model_path, args.logname+"-"+str(args.seed), str(args.year))):  # Traverse the files under the model path of the previous year and get all loss values
        loss.append(filename[0:-4])
    loss = sorted(loss)
    load_path = osp.join(args.model_path, args.logname+"-"+str(args.seed), str(args.year), loss[0]+".pkl")  # Set the loading path to the model file corresponding to the minimum loss value
    
    args.logger.info("[*] load from {}".format(load_path))  # Recording Load Path
    state_dict = torch.load(load_path, map_location=args.device)["model_state_dict"]  # Loading the model state dictionary
    
    model = args.methods[args.method](args)  # Initialize the model
    
    if args.method == 'EAC' or args.method == 'DepCL':
        if args.year == args.begin_year:
            model.expand_adaptive_params(args.base_node_size)
        else:
            for idx, _ in enumerate(range(args.year - args.begin_year)):
                model.expand_adaptive_params(args.graph_size_list[idx + 1])
    
    if args.method == 'Universal' and args.use_eac == True:
        if args.year == args.begin_year:
            model.expand_adaptive_params(args.base_node_size)
        else:
            for idx, _ in enumerate(range(args.year - args.begin_year)):
                model.expand_adaptive_params(args.graph_size_list[idx])
    
    model.load_state_dict(state_dict)  # Load the state dictionary into the model
    model = model.to(args.device)  # Move the model to the specified device
    return model, loss[0]  # Returns the model and the minimum loss value



def long_term_pattern(args, long_pattern):
    attention, _, _ = kcluster(long_pattern, nclusters=args.cluster, dist='u')  # [number of nodes, average number of days per day] -> [number of nodes] ranges from 0 to k-1
    np_attention = np.zeros((len(attention), args.cluster))  # [number of nodes, clusters]
    for i in attention:
        np_attention[i][attention[i]] = 1.0
    return np_attention.astype(np.float32)


def get_max_columns(matrix):
    tensor = torch.tensor(matrix)
    max_columns, _ = torch.max(tensor, dim=1)
    return max_columns


def get_space(data, model, args, adj):
    with torch.no_grad():
        space = model.get_fusion(data, adj)
        return torch.mean(space, dim=0)


def update_memory(M, args, L = []):
    threshold = 0.1
    task = args.year - args.begin_year
    if len(L) == 0:
        U, Z, V = torch.linalg.svd(M, full_matrices=False)
        sval_total = (Z**2).sum()
        sval_ratio = (Z**2) / sval_total
        r = torch.sum(torch.cumsum(sval_ratio, dim=-1) < threshold)
        # r = torch.sum(torch.cumsum(sval_ratio, dim=-1) > threshold)
        L.append(U[:, 0:r+1])
    else:
        U1, Z1, V1 = torch.linalg.svd(M, full_matrices=False)
        sval_total = (Z1 ** 2).sum()
        # Projected Representation
        for i in L:
            N = i.shape[0]
            M[:N, :] = M[:N, :] - torch.matmul(torch.matmul(i, i.permute(1, 0)), M[:N, :])
        M_hat = M
        U, S, Vh = torch.linalg.svd(M_hat, full_matrices=False)
        # criteria
        sval_hat = (S ** 2).sum()
        sval_ratio = (S ** 2) / sval_total
        accumulated_sval = (sval_total - sval_hat) / sval_total
        r = 0
        for ii in range(sval_ratio.shape[0]):
            if accumulated_sval < threshold:
            # if (accumulated_sval > threshold) or (accumulated_sval == threshold):

                accumulated_sval += sval_ratio[ii]
                r += 1
            else:
                break
        r = r + 1

        if r == 1:
            L = L

        # update GPM
        # U = torch.hstack((L, U[:, 0:r]))
        # if U.shape[1] > U.shape[0]:
        #     L = U[:, 0:U.shape[0]]
        # else:
        #     L = U
        L = [U[:, 0:r]]

    return L