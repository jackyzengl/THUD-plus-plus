import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from tqdm import tqdm
import time


def anorm(p1,p2):
    #求二范数的倒数; p1: [x,y];  p2: [x,y]
    NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
    if NORM ==0:
        return 0
    return 1/(NORM)
                
def seq_to_graph(seq_,seq_rel,norm_lap_matr = True):
    """
    args:
        seq_: [N,2,pred_len/obs_len]
        seq_rel: [N,2,pred_len/obs_len]
        norm_lap_matr: True
    return:
        V: [T, N, 2]
        A: [T, N, N], 加权的邻接矩阵，拉普拉斯正则化
    """
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    V = np.zeros((seq_len,max_nodes,2))  # [T, N, 2]
    A = np.zeros((seq_len,max_nodes,max_nodes))  # [T, N, N]
    for s in range(seq_len):  # T
        step_ = seq_[:,:,s] # [N, 2]
        step_rel = seq_rel[:,:,s] # [N, 2]
        for h in range(len(step_)):  # N
            V[s,h,:] = step_rel[h]
            A[s,h,h] = 1
            for k in range(h+1,len(step_)):
                l2_norm = anorm(step_rel[h],step_rel[k])
                A[s,h,k] = l2_norm
                A[s,k,h] = l2_norm
        if norm_lap_matr: 
            G = nx.from_numpy_array(A[s,:,:])  # A_s的邻接矩阵
            A[s,:,:] = nx.normalized_laplacian_matrix(G).toarray()
            
    return torch.from_numpy(V).type(torch.float),\
           torch.from_numpy(A).type(torch.float)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0
def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
        min_ped=1, delim='\t',norm_lap_matr = True):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []  # [N1,N2, ...] 每个序列样本的人数
        seq_list = []  # 将[[N1,2,seq_len], [N2,2,seq_len], ...]沿axis=0维度拼接的序列样本
        seq_list_rel = []  # 将[[N1,2,seq_len], [N2,2,seq_len], ...]沿axis=0维度拼接的序列样本
        loss_mask_list = []  # 将[[N1,seq_len], [N2,seq_len], ...]沿axis=0维度拼接，值全部置为1
        non_linear_ped = []
        for path in all_files:  # 每个txt文件
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()  # 帧序号的集合
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])  # [p, q, 4], p是独立帧的个数，q是每帧包含的人数

            # 为最大化利用数据集，每取seq_len帧作为一个sequence，就往后移动skip位(帧)，再取seq_len帧作为一个sequence。
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))  # math.ceil(): 天花板函数

            for idx in range(0, num_sequences * self.skip + 1, skip):  # 每组帧长度为seq_len的行人总和
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)  # [q1+q2+...+q_seqlen, 4]
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])  # 该组帧中独立行人的编号,数目为n,行人不一定覆盖全seq_len帧
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))  # [n,2，seq_len]
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))  # [n,2，seq_len]
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))  # [n,seq_len]
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):  # 组帧中的每个独立行人编号
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]  # [p1,4]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx  # 该行人在组帧中出现的第一帧的索引
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1

                    # 注意！！若行人出现总帧数少于obs_len + pred_len， 则忽略该行人
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq  # [2,seq_len]
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]  # [2,seq_len], 绝对坐标转相对坐标
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)  # 序列样本数
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)  # 将[N1,2,obs_len], [N2,2,obs_len], ...沿axis=0维度拼接的序列样本
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)  # 将[N1,2,pred_len], [N2,2,pred_len], ...沿axis=0维度拼接的序列样本
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)  # 将[N1,2,obs_len], [N2,2,obs_len], ...沿axis=0维度拼接的序列样本
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)  # 将[N1,2,pred_len], [N2,2,pred_len], ...沿axis=0维度拼接的序列样本
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)  # 将[N1,seq_len], [N2,seq_len], ...沿axis=0维度拼接，值全部置为1
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()  # [0, N1, N1+N2, N1+N2+N3, ...] 每个序列样本的人数
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])  # [num_seq, 2]
        ]
        # Convert to Graphs
        self.v_obs = []  # [[obs_len, N1, 2], [obs_len, N2, 2], ...]
        self.A_obs = []  # [[obs_len, N1, N1], [obs_len, N2, N2], ...]
        self.v_pred = []  # [[pred_len, N1, 2], [pred_len, N2, 2], ...]
        self.A_pred = []  # [[pred_len, N1, N1], [pred_len, N2, N2], ...]
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end)) 
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]

            v_,a_ = seq_to_graph(self.obs_traj[start:end,:],self.obs_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_,a_=seq_to_graph(self.pred_traj[start:end,:],self.pred_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        """
        return:
            self.obs_traj[start:end, :]: [N,2,obs_len]
            self.pred_traj[start:end, :]: [N,2,pred_len]
            self.obs_traj_rel[start:end, :]: [N,2,obs_len]
            self.pred_traj_rel[start:end, :]: [N,2,pred_len]
            self.non_linear_ped[start:end]: useless
            self.loss_mask[start:end, :]: [N,seq_len]
            self.v_obs[index]: [obs_len, N, 2]
            self.A_obs[index]: [obs_len, N, N]
            self.v_pred[index]: [pred_len, N, 2]
            self.A_pred[index]: [pred_len, N, N]
        """
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index]
        ]
        return out
