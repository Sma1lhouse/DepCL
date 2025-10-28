import numpy as np
import torch


def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(null_val)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def masked_mse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.mean(np.nan_to_num(mask * mse))


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))


def cal_metric(ground_truth, prediction, args):
    args.logger.info("[*] year {}, testing".format(args.year))
    mae_list, rmse_list, mape_list = [], [], []
    for i in range(1, 13):
        mae = masked_mae_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        rmse = masked_mse_np(ground_truth[:, :, :i], prediction[:, :, :i], 0) ** 0.5
        mape = masked_mape_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        mae_list.append(mae)
        rmse_list.append(rmse)
        mape_list.append(mape)
        if i==3 or i==6 or i==12:
            args.logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i,mae,rmse,mape))
            args.result[str(i)][" MAE"][args.year] = mae
            args.result[str(i)]["MAPE"][args.year] = mape
            args.result[str(i)]["RMSE"][args.year] = rmse
    args.result["Avg"][" MAE"][args.year] = np.mean(mae_list)
    args.result["Avg"]["RMSE"][args.year] = np.mean(rmse_list)
    args.result["Avg"]["MAPE"][args.year] = np.mean(mape_list)
    args.logger.info("T:Avg\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(np.mean(mae_list), np.mean(rmse_list), np.mean(mape_list)))


''' revise 研究节点公平性'''
def masked_mape_np_fair(y_true, y_pred, args, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()

        B, N, C = y_true.shape

        y_max = np.max(y_true)
        y_min = np.min(y_true)
        y_mean = np.mean(y_true)
        y_var = np.var(y_true)
        a1 = y_true

        # 655 715 786
        area = 655
        area2 = 715

        ''' 区域分布研究'''
        # sub = y_true[:, :area, :]
        sub = y_true[:, area:area2, :]

        a_max = np.max(sub)
        a_min = np.min(sub)
        a_mean = np.mean(sub)
        a_var = np.var(sub)


        new = y_true[:, area:, :]
        old = y_true[:, :area, :]
        a2 = new[new>y_mean]
        p1 = len(a2)/len(new.ravel())
        a3 = old[old>y_mean]
        p2 = len(a3)/len(old.ravel())


        per_node_mape = np.abs((y_pred - y_true) / y_true) * 100
        per_node_mape = np.nan_to_num(mask * per_node_mape)

        per_node_mae = np.abs(y_true - y_pred)
        per_node_mae = np.nan_to_num(mask * per_node_mae)

        # per_node_smape = 2 * np.abs(y_true - y_pred) / (np.abs(y_pred) + np.abs(y_true)) * 100
        per_node_smape = np.abs(y_true - y_pred) / (np.abs(y_pred) + np.abs(y_true)) * 100

        per_node_smape = np.nan_to_num(mask * per_node_smape)

        ''' 分区域节点smape'''
        previous_smape = np.mean(per_node_smape[:,:area,:])
        new_smape = np.mean(per_node_smape[:,area:,:])
        smape = np.mean(per_node_smape)
        previous_vari_smape = np.var(per_node_smape[:,:area,:])
        new_vari_smape = np.var(per_node_smape[:,area:,:])

        ''' 将mape换为smape'''
        # per_node_mape = per_node_smape
        per_node_mape = per_node_mae


        ''' 分区域节点mae'''
        previous_mae = np.mean(per_node_mae[:,:area,:])
        new_mae = np.mean(per_node_mae[:,area:,:])
        mae = np.mean(per_node_mae)
        previous_vari_mae = np.var(per_node_mae[:,:area,:])
        new_vari_mae = np.var(per_node_mae[:,area:,:])


        ''' 分区域节点mape'''
        previous_mape = np.mean(per_node_mape[:,:area,:])
        new_mape = np.mean(per_node_mape[:,area:,:])
        mape = np.mean(per_node_mape)
        previous_vari_mape = np.var(per_node_mape[:,:area,:])
        new_vari_mape = np.var(per_node_mape[:,area:,:])


        mini = np.min(per_node_mape)
        maxi = np.max(per_node_mape)
        meani = np.mean(per_node_mape)
        vari_mape = np.var(per_node_mape)
        vari_mae = np.var(per_node_mae)

        va = (per_node_mape - meani)**2
        va1 = np.mean(va)

        pre_va = np.mean(va[:,:area,:])
        aft_va = np.mean(va[:,area:,:])


        fore = per_node_mape[per_node_mape<meani]
        aft = per_node_mape[per_node_mape>meani]
        per = len(fore)/(len(fore) + len(aft))

        ''' per-node'''

        ''' 1'''
        # node_ = (np.mean(per_node_mape.reshape(N, -1), axis=-1)-meani)
        # node_[node_<0] = 0
        # pre_node_ = node_[:area]
        # a = np.sum([pre_node_>0])
        # aft_node_ = node_[area:]
        # b = np.sum([aft_node_>0])
        ''' 2'''
        node_ = (per_node_mape-meani).reshape(N, -1)
        node_[node_<0] = 0
        count = np.count_nonzero(node_, axis=-1)
        mean_count = np.mean(count)
        count = count-mean_count
        pre_node_ = count[:area]
        a = np.sum([pre_node_>0])
        aft_node_ = count[area:]
        b = np.sum([aft_node_>0])


        # pre_node_ = np.where(pre_node_ > 0, a, 0)
        # pre_bad_node_index = np.nonzero(pre_node_)
        # aft_node_ = np.where(aft_node_ > 0, a, 0)
        # new_bad_node_index = np.nonzero(aft_node_)

        # pre_bad_node_index = np.where(pre_node_ > 0)
        # new_bad_node_index = np.where(aft_node_ > 0)
        all_bad_node_count = np.where(count > 0)[0]
        pre_bad_node_index = all_bad_node_count[all_bad_node_count<area]
        new_bad_node_index = all_bad_node_count[all_bad_node_count>=area]
        adj = args.adj

        sub_adj = adj[pre_bad_node_index][:, area:]
        has_neighbor = (sub_adj > 0).any(dim=1)
        pre_count_has_neighbor = has_neighbor.sum().item()

        sub_adj = adj[new_bad_node_index][:, :area]
        has_neighbor = (sub_adj > 0).any(dim=1)
        new_count_has_neighbor = has_neighbor.sum().item()

        sub_adj = adj[pre_bad_node_index][:, new_bad_node_index]
        has_neighbor = (sub_adj > 0).any(dim=1)
        pre_new_count_has_neighbor = has_neighbor.sum().item()

        all_nodes = torch.arange(per_node_mape.shape[1])
        mask = torch.ones_like(all_nodes, dtype=torch.bool)  # 全 True
        mask[all_bad_node_count] = False
        mape_good = np.mean(per_node_mape[:, mask, :])
        mae_good = np.mean(per_node_mae[:, mask, :])

        lst = []
        for i in pre_bad_node_index:
            if (adj[[i]][:, new_bad_node_index] > 0).any(dim=1).sum().item() != 0:
                lst.append(i)
        mask = torch.ones_like(all_nodes, dtype=torch.bool)  # 全 True
        mask[np.array(lst)] = False
        mape_good2 = np.mean(per_node_mape[:, mask, :])
        mae_good2 = np.mean(per_node_mae[:, mask, :])

        remove_ = np.setdiff1d(all_bad_node_count, np.array(lst))
        mask = torch.ones_like(all_nodes, dtype=torch.bool)  # 全 True
        mask[remove_] = False
        mape_good3 = np.mean(per_node_mape[:, mask, :])
        mae_good3 = np.mean(per_node_mae[:, mask, :])

        remove_ = np.setdiff1d(all_bad_node_count, np.array(lst))
        mask = torch.ones_like(all_nodes, dtype=torch.bool)  # 全 True
        mask[remove_] = False
        mape_good4 = np.mean(per_node_mape[:, mask, :])
        mae_good4 = np.mean(per_node_mae[:, mask, :])

        ''' bad-node y'''
        bad_nodes = y_true[:, all_bad_node_count, :]
        ls = y_true[:,new_bad_node_index,:]
        ls2 = bad_nodes[:,-1,:]
        ls3 = bad_nodes[:,-2,:]
        m = np.mean(ls)
        n = np.mean(new)
        n2 = np.mean(old)
        n3 = np.mean(y_true)

        ''' bad-node mae'''
        lss = per_node_mae[:, new_bad_node_index, :]
        bad_nodes_mae = per_node_mae[:, all_bad_node_count, :]
        lss2 = np.mean(bad_nodes_mae[:,-1,:])
        lss3 = np.mean(bad_nodes_mae[:,-2,:])
        ms = np.mean(lss)
        ms2 = np.mean(per_node_mae[:, pre_bad_node_index, :])

        ''' area-node y mae'''
        ar1 = y_true[:, :655, :]
        ar2 = y_true[:, 655:715, :]
        ar3 = y_true[:, 715:786, :]
        ar4 = y_true[:, 786:, :]

        ar1 = np.mean(ar1)
        ar2 = np.mean(ar2)
        ar3 = np.mean(ar3)
        ar4 = np.mean(ar4)

        ar1e = per_node_mae[:, :655, :]
        ar2e = per_node_mae[:, 655:715, :]
        ar3e = per_node_mae[:, 715:786, :]
        ar4e = per_node_mae[:, 786:, :]
        ar1e = np.mean(ar1e)
        ar2e = np.mean(ar2e)
        ar3e = np.mean(ar3e)
        ar4e = np.mean(ar4e)

        ar1m = per_node_mape[:, :655, :]
        ar2m = per_node_mape[:, 655:715, :]
        ar3m = per_node_mape[:, 715:786, :]
        ar4m = per_node_mape[:, 786:, :]
        ar1m = np.mean(ar1m)
        ar2m = np.mean(ar2m)
        ar3m = np.mean(ar3m)
        ar4m = np.mean(ar4m)

        area1_index = all_bad_node_count[all_bad_node_count<655]
        condition = (all_bad_node_count>=655) & (all_bad_node_count<715)
        area2_index = all_bad_node_count[condition]
        condition = (all_bad_node_count >= 715) & (all_bad_node_count < area)
        area3_index = all_bad_node_count[condition]
        sub_adj = adj[area1_index][:, new_bad_node_index]
        has_neighbor = (sub_adj > 0).any(dim=1)
        pre_new_count_has_neighbor1 = has_neighbor.sum().item()

        sub_adj = adj[area2_index][:, new_bad_node_index]
        has_neighbor = (sub_adj > 0).any(dim=1)
        pre_new_count_has_neighbor2 = has_neighbor.sum().item()

        sub_adj = adj[area3_index][:, new_bad_node_index]
        has_neighbor = (sub_adj > 0).any(dim=1)
        pre_new_count_has_neighbor3 = has_neighbor.sum().item()



        a=1


        # pre_node_ = node_[:area, :]
        # pre_node_ind = pre_node_>0
        # pre_node_ = pre_node_[pre_node_>0]
        # aft_node_ = node_[area:, :]
        # aft_node_ = aft_node_[aft_node_>0]
        #
        # ''' previous node'''
        # previous_per_node_mape = per_node_mape[:,:area,:]
        # previous_mini = np.min(previous_per_node_mape)
        # previous_maxi = np.max(previous_per_node_mape)
        # previous_vari = np.var(previous_per_node_mape)
        #
        # previous_fore = previous_per_node_mape[previous_per_node_mape<meani]
        # previous_aft = previous_per_node_mape[previous_per_node_mape>meani]
        # previous_per = len(previous_fore)/(len(previous_fore) + len(previous_aft))
        #
        # ''' new node'''
        # new_per_node_mape = per_node_mape[:,area:,:]
        # new_mini = np.min(new_per_node_mape)
        # new_maxi = np.max(new_per_node_mape)
        # new_vari = np.var(new_per_node_mape)
        #
        # new_fore = new_per_node_mape[new_per_node_mape<meani]
        # new_aft = new_per_node_mape[new_per_node_mape>meani]
        # new_per = len(new_fore)/(len(new_fore) + len(new_aft))



        a=1
''' revise 研究节点公平性'''