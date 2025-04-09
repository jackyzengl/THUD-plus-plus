import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import time
from torch.utils.data import DataLoader
import argparse
import copy

sys.path.append("../utils/")
import matplotlib.pyplot as plt
import numpy as np
from models import *
from social_utils import *
from social_eth_ucy_utils import *
import yaml

parser = argparse.ArgumentParser(description='PECNet')

parser.add_argument('--num_workers', '-nw', type=int, default=0)
parser.add_argument('--gpu_index', '-gi', type=int, default=1)
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


def test(test_dataset, model, best_of_n=1, hyper_params=None):
    model.eval()
    assert best_of_n >= 1 and type(best_of_n) == int
    test_loss = 0

    with torch.no_grad():
        total_seen = 0
        total_ade = 0
        total_fde = 0

        for i, (traj, mask, initial_pos) in enumerate(
                zip(test_dataset.trajectory_batches, test_dataset.mask_batches, test_dataset.initial_pos_batches)):
            total_seen += len(traj)
            traj, mask, initial_pos = torch.DoubleTensor(traj).to(device), torch.DoubleTensor(mask).to(
                device), torch.DoubleTensor(initial_pos).to(device)
            x = traj[:, :hyper_params["past_length"], :]  # [batchsize, 8, 2]
            y = traj[:, hyper_params["past_length"]:, :]  # [batchsize, 12, 2]
            y = y.cpu().numpy()
            # reshape the data
            x = x.contiguous().view(-1, x.shape[1] * x.shape[2])  # [batchsize, 8*2]
            x = x.to(device)

            future = y[:, :-1, :]  # [batchsize, 2]
            dest = y[:, -1, :]  # [batchsize, 2]
            all_l2_errors_dest = []
            all_guesses = []
            for index in range(best_of_n):
                dest_recon = model.forward(x, initial_pos, device=device)
                dest_recon = dest_recon.cpu().numpy()  # [bs, 2]
                all_guesses.append(dest_recon)

                l2error_sample = np.linalg.norm(dest_recon - dest, axis=1)  # [bs]
                all_l2_errors_dest.append(l2error_sample)

            all_l2_errors_dest = np.array(all_l2_errors_dest)
            all_guesses = np.array(all_guesses)
            # average error
            l2error_avg_dest = np.mean(all_l2_errors_dest)

            # choosing the best guess
            indices = np.argmin(all_l2_errors_dest, axis=0)  # [bs]

            best_guess_dest = all_guesses[indices, np.arange(x.shape[0]),
                              :]  # [bs, 2] 进行best_of_n次destination预测，取每个人的最优

            # taking the minimum error out of all guess
            l2error_dest = np.mean(np.min(all_l2_errors_dest, axis=0))  # 跟stgcnn一样，是作弊行为

            # back to torch land
            best_guess_dest = torch.DoubleTensor(best_guess_dest).to(device)

            # using the best guess for interpolation
            interpolated_future = model.predict(x, best_guess_dest, mask, initial_pos)
            interpolated_future = interpolated_future.cpu().numpy()  # [bs, 2 * (future_length - 1)]
            best_guess_dest = best_guess_dest.cpu().numpy()

            # final overall prediction
            predicted_future = np.concatenate((interpolated_future, best_guess_dest), axis=1)
            predicted_future = np.reshape(predicted_future,
                                          (-1, hyper_params["future_length"], 2))  # [bs, future_length, 2]

            # ADE error
            l2error_overall = np.mean(np.linalg.norm(y - predicted_future, axis=2))

            l2error_overall /= hyper_params["data_scale"]
            l2error_dest /= hyper_params["data_scale"]
            l2error_avg_dest /= hyper_params["data_scale"]

            total_ade += (l2error_overall * len(traj))
            total_fde += (l2error_dest * len(traj))

        # print('Test time error in destination best: {:0.3f} and mean: {:0.3f}'.format(l2error_dest, l2error_avg_dest))
        # print('Test time error overall (ADE) best: {:0.3f}'.format(l2error_overall))

    # return l2error_overall, l2error_dest, l2error_avg_dest
    return (total_ade / total_seen), (total_fde / total_seen), l2error_avg_dest

def main():
    Avg_ADE = 0
    Avg_FDE = 0
    subset = 'eth'
    args.load_file = f"PECNET_social_model_{subset}.pt"
    checkpoint = torch.load('../saved_models/{}'.format(args.load_file), map_location=device)
    hyper_params = checkpoint["hyper_params"]
    print(hyper_params)

    N = args.num_trajectories  # number of generated trajectories
    model = PECNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"],
                    hyper_params["dec_size"], hyper_params["predictor_hidden_size"],
                    hyper_params['non_local_theta_size'], hyper_params['non_local_phi_size'],
                    hyper_params['non_local_g_size'], hyper_params["fdim"], hyper_params["zdim"],
                    hyper_params["nonlocal_pools"], hyper_params['non_local_dim'], hyper_params["sigma"],
                    hyper_params["past_length"], hyper_params["future_length"], args.verbose)
    model = model.double().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    # test_dataset = SocialDataset(set_name="test", b_size=hyper_params["test_b_size"], t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"], verbose=args.verbose)
    test_dataset = SocialDatasetETHUCY(set_name='gym', set_type="test", b_size=4096, t_tresh=0, d_tresh=100)

    for traj in test_dataset.trajectory_batches:  # 把所有agent的轨迹的全减去第一帧
        traj -= traj[:, :1, :]
        traj *= hyper_params["data_scale"]

    # average ade/fde for k=20 (to account for variance in sampling)
    num_samples = 20
    average_ade, average_fde = 0, 0
    for i in range(num_samples):
        # print(i)
        test_loss, final_point_loss_best, final_point_loss_avg = test(test_dataset, model, best_of_n=N,
                                                                        hyper_params=hyper_params)
        average_ade += test_loss
        average_fde += final_point_loss_best

    print(f"{subset} ADE:", average_ade / num_samples)
    print(f"{subset} FDE:", average_fde / num_samples)
    print()

    Avg_ADE += average_ade / num_samples
    Avg_FDE += average_fde / num_samples

    # print(f"******************\nAvg_ADE: {Avg_ADE / 20}\nAvg_FDE: {Avg_FDE / 20}")


if __name__ == '__main__':
    start_time = time.time()
    main()
    print(f'use time: {time.time() - start_time}')



