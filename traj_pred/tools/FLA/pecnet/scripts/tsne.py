import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader
import argparse
import copy
from sklearn.manifold import TSNE
from scipy.stats import chi2
from matplotlib.patches import Ellipse


sys.path.append("../utils/")
import matplotlib.pyplot as plt
import numpy as np
from models import *
from social_utils import *
from social_eth_ucy_utils import *
import yaml

parser = argparse.ArgumentParser(description='PECNet')

parser.add_argument('--num_workers', '-nw', type=int, default=0)
parser.add_argument('--gpu_index', '-gi', type=int, default=0)
parser.add_argument('--load_file', '-lf', default="PECNET_social_model_eth.pt")
parser.add_argument('--num_trajectories', '-nt', default=20)  # number of trajectories to sample
parser.add_argument('--verbose', '-v', action='store_true')
parser.add_argument('--root_path', '-rp', default="./")

args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
print(device)


def get_HighDimTensor(source_dataset, target_dataset, model, hyper_params):
    model.train()
    source_iter_traj = iter(source_dataset.trajectory_batches)
    source_iter_mask = iter(source_dataset.mask_batches)
    source_iter_pose = iter(source_dataset.initial_pos_batches)

    target_iter_traj = iter(target_dataset.trajectory_batches)
    target_iter_mask = iter(target_dataset.mask_batches)
    target_iter_pose = iter(target_dataset.initial_pos_batches)

    for batch_index in range(len_max):
        # Get data
        if batch_index % len_source == 0:
            del source_iter_traj
            del source_iter_mask
            del source_iter_pose
            source_iter_traj = iter(source_dataset.trajectory_batches)
            source_iter_mask = iter(source_dataset.mask_batches)
            source_iter_pose = iter(source_dataset.initial_pos_batches)
            # print('*** Source Iter is reset ***')
        if batch_index % len_target == 0:
            del target_iter_traj
            del target_iter_mask
            del target_iter_pose
            target_iter_traj = iter(target_dataset.trajectory_batches)
            target_iter_mask = iter(target_dataset.mask_batches)
            target_iter_pose = iter(target_dataset.initial_pos_batches)
            # print('*** Target Iter is reset ***')

        traj_s = torch.DoubleTensor(next(source_iter_traj)).to(device)
        mask_s = torch.DoubleTensor(next(source_iter_mask)).to(device)
        initial_pos_s = torch.DoubleTensor(next(source_iter_pose)).to(device)

        traj_t = torch.DoubleTensor(next(target_iter_traj)).to(device)
        mask_t = torch.DoubleTensor(next(target_iter_mask)).to(device)
        initial_pos_t = torch.DoubleTensor(next(target_iter_pose)).to(device)

        x_s = traj_s[:, :hyper_params['past_length'], :]  # [batchsize, 8, 2]
        y_s = traj_s[:, hyper_params['past_length']:, :]  # [batchsize, 12, 2]

        x_t = traj_t[:, :hyper_params['past_length'], :]  # [batchsize, 8, 2]
        y_t = traj_t[:, hyper_params['past_length']:, :]  # [batchsize, 12, 2]

        x_s = x_s.contiguous().view(-1, x_s.shape[1] * x_s.shape[2])  # (x,y,x,y ... )
        x_s = x_s.to(device)
        dest_s = y_s[:, -1, :].to(device)  # [bs, 2]
        future_s = y_s[:, :-1, :].contiguous().view(y_s.size(0), -1).to(device)  # [bs, (12-1)*2]

        x_t = x_t.contiguous().view(-1, x_t.shape[1] * x_t.shape[2])  # (x,y,x,y ... )
        x_t = x_t.to(device)
        dest_t = y_t[:, -1, :].to(device)  # [bs, 2]
        future_t = y_t[:, :-1, :].contiguous().view(y_t.size(0), -1).to(device)  # [bs, (12-1)*2]

        dest_recon_s, mu_s, var_s, interpolated_future_s, prediction_encoding_s = model.forward(x_s, initial_pos_s,
                                                                                                dest=dest_s,
                                                                                                mask=mask_s,
                                                                                                device=device)

        dest_recon_t, mu_t, var_t, interpolated_future_t, prediction_encoding_t = model.forward(x_t, initial_pos_t,
                                                                                                dest=dest_t,
                                                                                                mask=mask_t,
                                                                                                device=device)

        return prediction_encoding_s.detach().cpu().numpy(), prediction_encoding_t.detach().cpu().numpy()  # [bs, 2*fdim+2]


def visual(feat):  # [num, dim=64]
    ts = TSNE(n_components=2, random_state=0, perplexity=15)
    # ts = PCA(n_components=2)
    x_ts = ts.fit_transform(feat)
    # x_min, x_max = x_ts.min(0), x_ts.max(0)
    # x_final = (x_ts - x_min) / (x_max - x_min)
    x_final = x_ts
    return x_final


if __name__ == '__main__':
    dada_datasets = [
        'A2B', 'A2C', 'A2D', 'A2E',
        'B2A', 'B2C', 'B2D', 'B2E',
        'C2A', 'C2B', 'C2D', 'C2E',
        'D2A', 'D2B', 'D2C', 'D2E',
        'E2A', 'E2B', 'E2C', 'E2D']
    # dada_datasets = ['A2B']
    # dada_datasets = ['A2D', 'A2E', 'D2B', 'D2E', 'E2B']

    for subset in dada_datasets:
        args.load_file = f"PECNET_social_model_{subset}_DADA.pt"
        checkpoint = torch.load('../checkpoint_DADA/after_SocialPooling/{}'.format(args.load_file), map_location=device)
        hyper_params = checkpoint["hyper_params"]
        print(hyper_params)

        N = args.num_trajectories  # number of generated trajectories
        model = PECNet_FLA(hyper_params["enc_past_size"], hyper_params["enc_dest_size"],
                           hyper_params["enc_latent_size"],
                           hyper_params["dec_size"], hyper_params["predictor_hidden_size"],
                           hyper_params['non_local_theta_size'], hyper_params['non_local_phi_size'],
                           hyper_params['non_local_g_size'], hyper_params["fdim"], hyper_params["zdim"],
                           hyper_params["nonlocal_pools"], hyper_params['non_local_dim'], hyper_params["sigma"],
                           hyper_params["past_length"], hyper_params["future_length"], args.verbose)
        model = model.double().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        source_dataset = SocialDatasetETHUCY(set_name=subset, set_type="train", b_size=256, t_tresh=0, d_tresh=50)
        target_dataset = SocialDatasetETHUCY(set_name=subset, set_type="val", b_size=256, t_tresh=0, d_tresh=50)
        len_source, len_target = len(source_dataset.trajectory_batches), len(target_dataset.trajectory_batches)
        len_max = max(len_source, len_target)

        for traj in source_dataset.trajectory_batches:  # 把所有agent的轨迹的全减去第一帧
            traj -= traj[:, :1, :]
            traj *= hyper_params["data_scale"]
        for traj in source_dataset.trajectory_batches:  # 把所有agent的轨迹的全减去第一帧
            traj -= traj[:, :1, :]
            traj *= hyper_params["data_scale"]

        HighDim_s_real, HighDim_t = get_HighDimTensor(source_dataset, target_dataset, model, hyper_params)

        source_fake_dataset = SocialDatasetETHUCY(set_name=subset, set_type="train_fake", b_size=256, t_tresh=0, d_tresh=50)
        len_source_fake = len(source_fake_dataset.trajectory_batches)
        len_max = max(len_source_fake, len_target)

        for traj in source_fake_dataset.trajectory_batches:  # 把所有agent的轨迹的全减去第一帧
            traj -= traj[:, :1, :]
            traj *= hyper_params["data_scale"]

        HighDim_s_fake, _ = get_HighDimTensor(source_fake_dataset, target_dataset, model, hyper_params)

        print(subset, HighDim_t.shape, HighDim_s_fake.shape)
        LowDim_s_real, LowDim_t, LowDim_s_fake = visual(HighDim_s_real), visual(HighDim_t), visual(HighDim_s_fake)


        # fig, ax = plt.subplots()
        # colors = ['red', 'green', 'blue']
        # labels = ['Class A', 'Class B', 'Class C']
        # for i, dataset in enumerate([LowDim_s, LowDim_t]):
        #     # 计算每个类别的协方差矩阵和椭圆参数
        #     covariance_matrix = np.cov(dataset.T)
        #     center = np.mean(dataset, axis=0)
        #     eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        #     angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        #     width = 2 * np.sqrt(eigenvalues[0]) * np.sqrt(chi2.ppf(0.95, df=2))
        #     height = 2 * np.sqrt(eigenvalues[1]) * np.sqrt(chi2.ppf(0.95, df=2))
        #
        #     # 绘制样本分布的散点图，并使用自定义的颜色和标签
        #     ax.scatter(dataset[:, 0], dataset[:, 1], c=colors[i], label=labels[i])
        #
        #     # 绘制置信椭圆
        #     ellipse = Ellipse(xy=center, width=width, height=height, angle=angle, edgecolor=colors[i], facecolor='none')
        #     ax.add_patch(ellipse)
        #
        # ax.legend()
        # ax.set_xlabel('PC1')
        # ax.set_ylabel('PC2')
        # ax.set_title('Sample Distribution and Confidence Ellipses')
        #
        # plt.savefig(f'../DomainDiff_visualization/after_SocialPooling/{subset}.png', bbox_inches='tight', dpi=300)


        fig, ax = plt.subplots()
        ax.scatter(LowDim_s_real[:, 0], LowDim_s_real[:, 1], marker='.', c='red', s=14, label='source_origin')
        ax.scatter(LowDim_s_fake[:, 0], LowDim_s_fake[:, 1], marker='.', c='#89fe05', s=14, label='source_DLA')
        ax.scatter(LowDim_t[:, 0], LowDim_t[:, 1], marker='.', c='blue', s=14, label='target')

        # plt.title(subset, fontsize=12, fontweight='normal')
        ax.tick_params(axis='both', direction='in')
        # plt.xticks([])  # 去掉横坐标值
        # plt.yticks([])  # 去掉纵坐标值
        # ax.legend()
        plt.savefig(f'../DomainDiff_visualization/after_SocialPooling/{subset}_FLA.png', bbox_inches='tight', dpi=300)
