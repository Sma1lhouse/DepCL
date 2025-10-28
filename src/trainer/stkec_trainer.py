import torch
import torch.nn as nn
import numpy as np
import os.path as osp
import networkx as nx
import torch.nn.functional as func
from torch import optim
from datetime import datetime
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm

from src.model.ewc import EWC
from torch_geometric.loader import DataLoader
from dataer.SpatioTemporalDataset import SpatioTemporalDataset
from utils.metric import cal_metric, masked_mae_np
from utils.common_tools import mkdirs, load_best_model, get_max_columns

'''
class SDC_Module(nn.Module):
    """SDC Module"""
    def __init__(self, num_nodes, freq_bins, groups=4):
        super().__init__()
        self.groups = groups
        self.group_size = freq_bins // groups
        self.lambda_amp = nn.Parameter(torch.zeros(groups, num_nodes, 1))
        self.lambda_phi = nn.Parameter(torch.zeros(groups, num_nodes, 1))

    def forward(self, y_pred):
        # y_pred: [B,1,N,T]
        B, C, N, T = y_pred.shape
        y = y_pred[:,0]  # [B,N,T]
        # FFT -> [B,N,M]
        Yf = torch.fft.rfft(y, dim=-1)
        A = torch.abs(Yf)
        P = torch.angle(Yf)
        
        Yf_corr = torch.zeros_like(Yf)
        for g in range(self.groups):
            start = g * self.group_size
            end = T//2+1 if g==self.groups-1 else (g+1)*self.group_size
            lam_a = self.lambda_amp[g].unsqueeze(0)  # -> [1,N,1]
            lam_p = self.lambda_phi[g].unsqueeze(0)
            
            A_g = A[:,:,start:end] * (1 + lam_a)
            P_g = P[:,:,start:end] + lam_p
            
            Yf_corr[:,:,start:end] = A_g * torch.exp(1j * P_g)
        
        y_time = torch.fft.irfft(Yf_corr, n=T, dim=-1)
        return y_time.unsqueeze(1)  # [B,1,N,T]
'''


def train(inputs, args):
    path = osp.join(args.path, str(args.year))  # Define the current year model save path
    mkdirs(path)
    
    # Setting the loss function
    if args.loss == "mse":
        lossfunc = func.mse_loss
    elif args.loss == "huber":
        lossfunc = func.smooth_l1_loss
    
    cluster_lossfunc = nn.CrossEntropyLoss()
    
    # Dataset definition
    if args.strategy == 'incremental' and args.year > args.begin_year:
        # Incremental Policy Data Loader
        train_loader = DataLoader(SpatioTemporalDataset("", "", x=inputs["train_x"][:, :, args.subgraph.numpy()], y=inputs["train_y"][:, :, args.subgraph.numpy()], \
            edge_index="", mode="subgraph"), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=32)
        val_loader = DataLoader(SpatioTemporalDataset("", "", x=inputs["val_x"][:, :, args.subgraph.numpy()], y=inputs["val_y"][:, :, args.subgraph.numpy()], \
            edge_index="", mode="subgraph"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
        # Construct the adjacency matrix of the subgraph
        graph = nx.Graph()
        graph.add_nodes_from(range(args.subgraph.size(0)))
        graph.add_edges_from(args.subgraph_edge_index.numpy().T)
        adj = nx.to_numpy_array(graph)  # Convert to adjacency matrix
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)  # Normalized adjacency matrix
        vars(args)["sub_adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
    else:
        # Common data loader
        train_loader = DataLoader(SpatioTemporalDataset(inputs, "train"), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=32)
        val_loader = DataLoader(SpatioTemporalDataset(inputs, "val"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
        vars(args)["sub_adj"] = vars(args)["adj"]  # Use the adjacency matrix of the entire graph
    
    # Testing the Data Loader
    test_loader = DataLoader(SpatioTemporalDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
    
    vars(args)["past_adj"]=args.sub_adj
    
    args.logger.info("[*] Year " + str(args.year) + " Dataset load!")  # Record dataset loading log

    # Model definition
    if args.init == True and args.year > args.begin_year:
        gnn_model, _ = load_best_model(args)  # If it is not the first year, load the optimal model
        if args.ewc:  # If you use the ewc strategy, use the ewc model
            args.logger.info("[*] EWC! lambda {:.6f}".format(args.ewc_lambda))  # Record EWC related parameters
            model = EWC(gnn_model, args.adj, args.ewc_lambda, args.ewc_strategy)  # Initialize the EWC model
            ewc_loader = DataLoader(SpatioTemporalDataset(inputs, "train"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
            model.register_ewc_params_for_stkec(ewc_loader, lossfunc, args.device)  # Register EWC parameters
        else:
            model = gnn_model  # Otherwise, use the best model loaded
    else:
        gnn_model = args.methods[args.method](args).to(args.device)  # If it is the first year, use the base model
        model = gnn_model

    ''' revise'''
    model.count_parameters()

    # Model Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    args.logger.info("[*] Year " + str(args.year) + " Training start")
    lowest_validation_loss = 1e7
    counter = 0
    patience = 5
    model.train()
    use_time = []
    ''' revise'''
    # patience = 1
    # args.epoch = 1

    for epoch in range(args.epoch):
        
        start_time = datetime.now()
        
        # Training the model
        cn = 0
        training_loss = 0.0
        loss_cluster = 0.0
        for batch_idx, data in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                args.logger.info("node number {}".format(data.x.shape))
            data = data.to(args.device, non_blocking=True)
            optimizer.zero_grad()
            
            pred, attention = model(data, args.sub_adj)
            
            batch_att = pred.shape[0] // args.sub_adj.shape[0]
            if args.year == args.begin_year:
                attention_label = torch.from_numpy(args.attention.repeat(batch_att, axis=0)).to(args.device)
                loss_cluster = cluster_lossfunc(attention.data.cpu(), get_max_columns(attention_label).data.cpu().long())
            
            
            if args.strategy == "incremental" and args.year > args.begin_year:
                pred, _ = to_dense_batch(pred, batch=data.batch)  # to_dense_batch is used to convert a batch of sparse adjacency matrices into a batch of dense adjacency matrices
                data.y, _ = to_dense_batch(data.y, batch=data.batch)
                pred = pred[:, args.mapping, :]  # Slice according to the mapping to obtain the prediction and true value of the change node
                data.y = data.y[:, args.mapping, :]
            
            loss = lossfunc(data.y, pred, reduction="mean") + loss_cluster * 0.1  # Calculating Losses
            
            if args.ewc and args.year > args.begin_year:
                loss += model.compute_consolidation_loss()  # Calculate and add ewc loss
            
            training_loss += float(loss)
            cn += 1
            
            loss.backward()
            optimizer.step()
        
        if epoch == 0:
            total_time = (datetime.now() - start_time).total_seconds()
        else:
            total_time += (datetime.now() - start_time).total_seconds()
        use_time.append((datetime.now() - start_time).total_seconds())
        training_loss = training_loss / cn 
        
        # Validate the model
        validation_loss = 0.0
        cn = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = data.to(args.device, non_blocking=True)
                pred, attention = model(data, args.sub_adj)
                if args.strategy == "incremental" and args.year > args.begin_year:
                    pred, _ = to_dense_batch(pred, batch=data.batch)
                    data.y, _ = to_dense_batch(data.y, batch=data.batch)
                    pred = pred[:, args.mapping, :]
                    data.y = data.y[:, args.mapping, :]
                    
                loss = masked_mae_np(data.y.cpu().data.numpy(), pred.cpu().data.numpy(), 0)
                validation_loss += float(loss)
                cn += 1
        validation_loss = float(validation_loss/cn)
        

        args.logger.info(f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}")
        
        # Early Stopping Strategy
        if validation_loss <= lowest_validation_loss:
            counter = 0
            lowest_validation_loss = round(validation_loss, 4)
            torch.save({'model_state_dict': gnn_model.state_dict()}, osp.join(path, str(round(validation_loss,4))+".pkl"))
        else:
            counter += 1
            if counter > patience:
                break
        
    best_model_path = osp.join(path, str(lowest_validation_loss)+".pkl")  # The model with the lowest validation loss is selected as the optimal model
    best_model = args.methods[args.method](args)
    best_model.load_state_dict(torch.load(best_model_path, args.device)["model_state_dict"])
    best_model = best_model.to(args.device)
    
    # Test the Model
    test_model(best_model, args, test_loader, True)
    args.result[args.year] = {"total_time": total_time, "average_time": sum(use_time)/len(use_time), "epoch_num": epoch+1}
    args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path))


def test_model(model, args, testset, pin_memory):
    model.eval()
    pred_ = []
    truth_ = []
    loss = 0.0
    with torch.no_grad():
        cn = 0
        for data in testset:
            data = data.to(args.device, non_blocking=pin_memory)
            pred, attention = model(data, args.adj)
            loss += func.mse_loss(data.y, pred, reduction="mean")
            pred, _ = to_dense_batch(pred, batch=data.batch)
            data.y, _ = to_dense_batch(data.y, batch=data.batch)
            pred_.append(pred.cpu().data.numpy())
            truth_.append(data.y.cpu().data.numpy())
            cn += 1
        loss = loss / cn
        args.logger.info("[*] loss:{:.4f}".format(loss))
        pred_ = np.concatenate(pred_, 0)
        truth_ = np.concatenate(truth_, 0)
        cal_metric(truth_, pred_, args)




def masked_mae(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.0)

    mask = mask.float()
    mask /= torch.mean(mask)  # Normalize mask to avoid bias in the loss due to the number of valid entries
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

    loss = torch.abs(prediction - target)
    loss = loss * mask  # Apply the mask to the loss
    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero

    return torch.mean(loss)


'''
def test_model_with_ttc(model, args, testset, pin_memory):
    
    model.eval()
    
    T = 12
    M = T // 2 + 1
    groups = 4
    FRP = SDC_Module(args.graph_size, M, groups).to(args.device)
    optim = torch.optim.Adam(FRP.parameters(), lr=1e-4)
    crit = masked_mae
    import queue
    q = queue.Queue(maxsize=T)
    
    pred_ = []
    truth_ = []
    
    for data in tqdm(testset):
        data = data.to(args.device, non_blocking=pin_memory)
        
        with torch.no_grad():
            pred, _ = model(data, args.adj)
        
        pred, _ = to_dense_batch(pred, batch=data.batch)
        data.y, _ = to_dense_batch(data.y, batch=data.batch)
        
        pred = pred.unsqueeze(1)
        
        FRP.eval()
        pred = FRP(pred)
        
        pred = pred.squeeze(1)
        
        pred_.append(pred.cpu().data.numpy())
        truth_.append(data.y.cpu().data.numpy())
        
        q.put((data, data.y))
        if q.full():
            x_o, y_o = q.get()
            with torch.no_grad():
                yb_o, _ = model(x_o, args.adj)
            
            yb_o, _ = to_dense_batch(yb_o, batch=data.batch)
            
            yb_o = yb_o.unsqueeze(1)
            
            FRP.train()
            yc_o = FRP(yb_o)
            
            yc_o = yc_o.squeeze(1)
            
            loss = func.mse_loss(yc_o, y_o)
            loss.backward()
            optim.step()
            optim.zero_grad()
            FRP.eval()
        
    
    pred_ = np.concatenate(pred_, 0)
    truth_ = np.concatenate(truth_, 0)
    cal_metric(truth_, pred_, args)
'''