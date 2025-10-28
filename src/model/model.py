import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gcn_conv import BatchGCNConv, ChebGraphConv

class DepCL_Model(nn.Module):
    """Some Information about EAC_Model"""

    def __init__(self, args):
        super(DepCL_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.rank = args.rank  # Set a low rank value
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"],
                              kernel_size=args.tcn["kernel_size"],
                              dilation=args.tcn["dilation"],
                              padding=int((args.tcn["kernel_size"] - 1) * args.tcn["dilation"] / 2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()

        self.year = args.year
        self.num_nodes = args.base_node_size

        self.codebook_size = 128
        self.hash_levels = 7
        t = 12
        self.prompt_codebook = nn.Parameter(
            torch.empty(self.codebook_size, t).uniform_(-0.1, 0.1)
        )
        self.linear = nn.Linear(args.gcn["in_channel"], t)
        self.norm = nn.LayerNorm(t)


        self.register_buffer("hash_index", torch.zeros(self.num_nodes))



    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")

    def forward(self, data, adj, training_mode):
        N = adj.shape[0]

        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))  # [bs, N, feature]

        B, N, T = x.shape


        query = self.norm(F.relu(self.linear(x)))

        if N < self.hash_index.shape[0]:
            hash_indices = self.hash_index[:N].long()
        else:
            hash_indices = self.hash_index.long()


        adaptive_params = self.prompt_codebook[hash_indices]  # Shape: [N, feature_dim]

        adj = torch.softmax(torch.matmul(query, adaptive_params.permute(1, 0)), dim=-1)

        x = x + adaptive_params.unsqueeze(0).expand(B, *adaptive_params.shape)  # [bs, N, feature]

        x = F.relu(self.gcn1(x, adj))  # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))  # [bs * N, 1, feature]

        x = self.tcn1(x)  # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))  # [bs, N, feature]
        x = self.gcn2(x, adj)  # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))  # [bs * N, feature]

        x = x + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)


        if training_mode == 1:
            return x, None, None

        elif training_mode == 2:
            return x


    def calculate_self_contrastive_loss(self, T):

        hash_indices = self.hash_index  # [N]
        labels_i = hash_indices.unsqueeze(1)  # [N, 1]
        labels_j = hash_indices.unsqueeze(0)  # [1, N]
        label_matrix = (labels_i == labels_j).float()  # [N, N]
        query_single = self.prompt_codebook[hash_indices.long()]  # [N, t]

        query_norm = F.normalize(query_single, p=2, dim=-1)  # [N, t]
        sim_matrix = torch.matmul(query_norm, query_norm.T)
        sim_matrix = sim_matrix / T
        T_m = torch.ones((sim_matrix.shape)).to(sim_matrix.device) / T
        if self.N_old is not None:
            propotion = (len(hash_indices) - self.N_old) / len(hash_indices)

            T_m[self.N_old:, self.N_old:] = T_m[self.N_old:, self.N_old:] / (1 - propotion)

        sim_matrix = sim_matrix * T_m

        sim_scaled = F.softmax(sim_matrix, dim=-1)

        loss_bce = F.cross_entropy(sim_scaled, label_matrix)

        return loss_bce

    def init_hash(self, first_batch_X, flag):
        with torch.no_grad():
            ''' mod.'''
            # node_ids = torch.arange(self.args.graph_size, device=first_batch_X.device)
            # hash_indices = node_ids % self.codebook_size
            # self.hash_index = hash_indices

            ''' binary'''
            first_batch_X = first_batch_X.reshape(
                (-1, self.args.graph_size, self.args.gcn["in_channel"]))  # [bs, N, feature]
            hash_levels = self.hash_levels  # 2^7 = 128
            # V = first_batch_X.mean(dim=0).cpu()  # 树构建在CPU上更稳定
            ''' revise'''
            V = first_batch_X[0].cpu()  # 树构建在CPU上更稳定

            hash_map_rfb = _build_rfb_hash_map(V, hash_levels)
            hash_map = hash_map_rfb.to(first_batch_X.device)
            self.hash_index = hash_map  # [655], 值为 0-127

            # values, counts = torch.unique(hash_map, return_counts=True)
            if flag:
                self.N_old = len(self.hash_index)
                self.N_new = self.args.graph_size
            else:
                self.N_old = None


    def expand_adaptive_params(self, new_num_nodes):
        with torch.no_grad():
            self.hash_index = torch.zeros(new_num_nodes)
            self.num_nodes = new_num_nodes


    def get_fusion(self, data, adj):
        with torch.no_grad():
            N = adj.shape[0]
            x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))  # [bs, N, feature]
            B, N, T = x.shape
            hash_indices = self.hash_index.long()
            adaptive_params = self.prompt_codebook[hash_indices]  # Shape: [N, feature_dim]
            fusion = x + adaptive_params.unsqueeze(0).expand(B, *adaptive_params.shape)  # [bs, N, feature]
            return fusion


def _build_rfb_hash_map(V, levels):
    N, F = V.shape
    hash_map = torch.zeros(N, dtype=torch.long, device=V.device)

    if levels == 0:
        return hash_map

    dim_to_split = torch.argmax(torch.var(V, dim=0))  # index (0-11)

    median_val = torch.median(V[:, dim_to_split])
    # median_val = torch.mean(V[:, dim_to_split])

    # if levels > levels-1:
    #     median_val = torch.mean(V[:, dim_to_split])
    # else:
    #     median_val = torch.median(V[:, dim_to_split])

    left_indices = torch.where(V[:, dim_to_split] <= median_val)[0]
    right_indices = torch.where(V[:, dim_to_split] > median_val)[0]

    if len(left_indices) == 0 or len(right_indices) == 0:
        left_indices = torch.arange(N // 2, device=V.device)
        right_indices = torch.arange(N // 2, N, device=V.device)

    # (N_left) -> (N_left)
    left_hash_map = _build_rfb_hash_map(V[left_indices], levels - 1)
    # (N_right) -> (N_right)
    right_hash_map = _build_rfb_hash_map(V[right_indices], levels - 1)

    hash_map[left_indices] = left_hash_map
    hash_map[right_indices] = right_hash_map + (2 ** (levels - 1))

    return hash_map

