import sys

import torch
import torch.nn as nn
import numpy as np
import os.path as osp
import networkx as nx
import torch.nn.functional as func
from tqdm import tqdm
from torch import optim
from datetime import datetime
from torch_geometric.utils import to_dense_batch

from src.model.ewc import EWC
from torch_geometric.loader import DataLoader
sys.path.append("src/")

from dataer.SpatioTemporalDataset import SpatioTemporalDataset
from utils.metric import cal_metric, masked_mae_np, masked_mape_np_fair
from utils.common_tools import mkdirs, load_best_model, update_memory, get_space

class Default_trainer(nn.Module):
    def __init__(self):
        super(Default_trainer, self).__init__()
        self.buffer = []


    def init_buffer(self, inputs, model, args):
        train_loader = DataLoader(SpatioTemporalDataset(inputs, "train"), batch_size=args.batch_size, shuffle=True,
                                  pin_memory=True, num_workers=2)
        for batch_idx, data in enumerate(train_loader):
            if batch_idx > 0:
                break
            data_temp = data.to(args.device, non_blocking=True)
            M = get_space(data_temp, model, args, args.adj)
            self.buffer = update_memory(M, args, [])


    def train(self, inputs, args):
        path = osp.join(args.path, str(args.year))  # Define the current year model save path
        mkdirs(path)


        # Setting the loss function
        if args.loss == "mse":
            lossfunc = func.mse_loss
        elif args.loss == "huber":
            lossfunc = func.smooth_l1_loss

        # Dataset definition
        if args.strategy == 'incremental' and args.year > args.begin_year:
            # Incremental Policy Data Loader
            train_loader = DataLoader(SpatioTemporalDataset("", "", x=inputs["train_x"][:, :, args.subgraph.numpy()], y=inputs["train_y"][:, :, args.subgraph.numpy()], \
                edge_index="", mode="subgraph"), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
            val_loader = DataLoader(SpatioTemporalDataset("", "", x=inputs["val_x"][:, :, args.subgraph.numpy()], y=inputs["val_y"][:, :, args.subgraph.numpy()], \
                edge_index="", mode="subgraph"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
            # Construct the adjacency matrix of the subgraph
            graph = nx.Graph()
            graph.add_nodes_from(range(args.subgraph.size(0)))
            graph.add_edges_from(args.subgraph_edge_index.numpy().T)
            adj = nx.to_numpy_array(graph)  # Convert to adjacency matrix
            adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)  # Normalized adjacency matrix
            vars(args)["sub_adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
        else:
            # Common Data Loader
            train_loader = DataLoader(SpatioTemporalDataset(inputs, "train"), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
            val_loader = DataLoader(SpatioTemporalDataset(inputs, "val"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
            vars(args)["sub_adj"] = vars(args)["adj"]  # Use the adjacency matrix of the entire graph

        # Test Data Loader
        test_loader = DataLoader(SpatioTemporalDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

        args.logger.info("[*] Year " + str(args.year) + " Dataset load!")

        # Model definition
        if args.init == True and args.year > args.begin_year:
            gnn_model, _ = load_best_model(args)  # If it is not the first year, load the optimal model
            if args.ewc:  # If you use the ewc strategy, use the ewc model
                args.logger.info("[*] EWC! lambda {:.6f}".format(args.ewc_lambda))  # Record EWC related parameters
                model = EWC(gnn_model, args.adj, args.ewc_lambda, args.ewc_strategy)  # Initialize the EWC model
                ewc_loader = DataLoader(SpatioTemporalDataset(inputs, "train"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
                model.register_ewc_params(ewc_loader, lossfunc, args.device)  # Register EWC parameters
            else:
                model = gnn_model  # Otherwise, use the best model loaded

            if args.method == 'DepCL':
                for name, param in model.named_parameters():

                    if "gcn1" in name or "tcn1" in name or "gcn2" in name or "fc" in name or "linear" in name or "norm" in name:
                        param.requires_grad = False



            if args.method == 'Universal' and args.use_eac == True:
                for name, param in model.named_parameters():
                    if "gcn1" in name or "tcn1" in name or "gcn2" in name or "fc" in name:
                        param.requires_grad = False

            if args.method == 'Universal' and args.use_eac == True:
                model.expand_adaptive_params(args.graph_size)

        else:
            gnn_model = args.methods[args.method](args).to(args.device)  # If it is the first year, use the base model
            model = gnn_model


            if args.method == 'Universal' and args.use_eac == True:
                model.expand_adaptive_params(args.graph_size)

        if args.logname != 'trafficstream':
            model.count_parameters()
            a = 1



        # Model Optimizer
        # optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)



        args.logger.info("[*] Year " + str(args.year) + " Training start")
        lowest_validation_loss = 1e7
        counter = 0
        patience = 5
        ''' revise'''
        # patience = 100

        model.train()
        use_time = []

        ''' revise'''
        # args.epoch = 1
        data_temp = None

        for epoch in range(args.epoch):

            start_time = datetime.now()

            # Training the model
            cn = 0
            training_loss = 0.0


            for batch_idx, data in enumerate(train_loader):


                if epoch == 0 and batch_idx == 0:
                    args.logger.info("node number {}".format(data.x.shape))
                    data_temp = data.to(args.device, non_blocking=True)
                    ''' revise init'''
                    model.init_hash(data.x.to(args.device), args.year > args.begin_year)

                data = data.to(args.device, non_blocking=True)
                optimizer.zero_grad()
                # pred = model(data, args.sub_adj)
                pred, U, Z = model(data, args.sub_adj, training_mode=1)



                if args.strategy == "incremental" and args.year > args.begin_year:
                    pred, _ = to_dense_batch(pred, batch=data.batch)  # to_dense_batch is used to convert a batch of sparse adjacency matrices into a batch of dense adjacency matrices
                    data.y, _ = to_dense_batch(data.y, batch=data.batch)
                    pred = pred[:, args.mapping, :]  # Slice according to the mapping to obtain the prediction and true value of the change node
                    data.y = data.y[:, args.mapping, :]

                loss = lossfunc(data.y, pred, reduction="mean")
                T = 1
                ''' revise'''
                loss_cl = model.calculate_self_contrastive_loss(T)
                loss = loss + loss_cl*0.03


                if args.ewc and args.year > args.begin_year:
                    loss += model.compute_consolidation_loss()  # Calculate and add ewc loss if necessary

                training_loss += float(loss)
                cn += 1

                loss.backward()

                ''' revise svd'''
                # if args.U_1 is not None:
                if len(self.buffer) > 0:
                    # a=args.U_1
                    # U_1 = args.U_1
                    for i in self.buffer:
                        N_old, k = i.shape
                        u_grad = model.prompt_codebook.grad.data[model.hash_index[:N_old].long()]
                        U_1 = i
                        U_1_sum = torch.matmul(U_1, U_1.T)
                        model.prompt_codebook.grad.data[model.hash_index[:N_old].long()] = u_grad - torch.matmul(U_1_sum, u_grad)
                ''' revise svd'''
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
                    pred = model(data, args.sub_adj, training_mode=2)
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
                if args.ewc:
                    torch.save({'model_state_dict': gnn_model.state_dict()}, osp.join(path, str(round(validation_loss,4))+".pkl"))
                else:
                    torch.save({'model_state_dict': model.state_dict()}, osp.join(path, str(round(validation_loss,4))+".pkl"))
            else:
                counter += 1
                if counter > patience:
                    break

        best_model_path = osp.join(path, str(lowest_validation_loss)+".pkl")

        if args.logname == 'trafficstream':
            best_model = args.methods[args.method](args)
        else:
            best_model = model

        best_model.load_state_dict(torch.load(best_model_path, args.device)["model_state_dict"])
        best_model = best_model.to(args.device)

        ''' revise svd'''
        M = get_space(data_temp, best_model, args, args.sub_adj)
        self.buffer = update_memory(M, args, self.buffer)

        # Test the Model
        self.test_model(best_model, args, test_loader, True)
        args.result[args.year] = {"total_time": total_time, "average_time": sum(use_time)/len(use_time), "epoch_num": epoch+1}
        args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path))


    def test_model(self, model, args, testset, pin_memory):

        model.eval()
        pred_ = []
        truth_ = []
        loss = 0.0
        with torch.no_grad():
            cn = 0
            for data in testset:
                data = data.to(args.device, non_blocking=pin_memory)
                pred = model(data, args.adj, training_mode=2)

                loss += func.mse_loss(data.y, pred, reduction="mean")
                pred, _ = to_dense_batch(pred, batch=data.batch)
                data.y, _ = to_dense_batch(data.y, batch=data.batch)


                pred_.append(pred.cpu().data.numpy())
                truth_.append(data.y.cpu().data.numpy())

                ''' revise fairness'''
                # masked_mape_np_fair(pred.cpu().data.numpy(), data.y.cpu().data.numpy(), 0)

                cn += 1
            loss = loss / cn
            args.logger.info("[*] loss:{:.4f}".format(loss))
            pred_ = np.concatenate(pred_, 0)
            truth_ = np.concatenate(truth_, 0)

            ''' revise fairness'''
            # masked_mape_np_fair(truth_, pred_, args, 0)

            cal_metric(truth_, pred_, args)



    def masked_mae(self, prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
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

