import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader
import argparse
import time

# sys.path.append("../eth_ucy/")
# from social_eth_ucy_utils import *

sys.path.append("../utils/")
import yaml
from models import *
from social_eth_ucy_utils import *
import numpy as np
import pickle

parser = argparse.ArgumentParser(description='PECNet')

parser.add_argument('--num_workers', '-nw', type=int, default=0)
parser.add_argument('--gpu_index', '-gi', type=int, default=3)
parser.add_argument('--config_filename', '-cfn', type=str, default='optimal_eth_ucy.yaml')
parser.add_argument('--save_file', '-sf', type=str, default='PECNET_social_model_eth.pt')
parser.add_argument('--verbose', '-v', action='store_true')
parser.add_argument('--mlp_dim', type=int, default=68)

args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
print(device)

with open("../config/" + args.config_filename, 'r') as file:
    try:
        hyper_params = yaml.load(file, Loader=yaml.FullLoader)
    except:
        hyper_params = yaml.load(file)
file.close()
print(hyper_params)


def train(train_dataset, test_dataset):
    model.train()
    d_inlay.train()
    train_loss, g_loss, d_loss = 0, 0, 0
    total_rcl, total_kld, total_adl = 0, 0, 0
    criterion = nn.MSELoss()

    source_iter_traj = iter(train_dataset.trajectory_batches)
    source_iter_mask = iter(train_dataset.mask_batches)
    source_iter_pose = iter(train_dataset.initial_pos_batches)

    target_iter_traj = iter(test_dataset.trajectory_batches)
    target_iter_mask = iter(test_dataset.mask_batches)
    target_iter_pose = iter(test_dataset.initial_pos_batches)

    for batch_index in range(len_max):
        # Get data
        if batch_index % len_source == 0:
            del source_iter_traj
            del source_iter_mask
            del source_iter_pose
            source_iter_traj = iter(train_dataset.trajectory_batches)
            source_iter_mask = iter(train_dataset.mask_batches)
            source_iter_pose = iter(train_dataset.initial_pos_batches)
            # print('*** Source Iter is reset ***')
        if batch_index % len_target == 0:
            del target_iter_traj
            del target_iter_mask
            del target_iter_pose
            target_iter_traj = iter(test_dataset.trajectory_batches)
            target_iter_mask = iter(test_dataset.mask_batches)
            target_iter_pose = iter(test_dataset.initial_pos_batches)
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

        dest_recon_s, mu_s, var_s, interpolated_future_s, _ = model.forward(x_s, initial_pos_s,
                                                                                                dest=dest_s,
                                                                                                mask=mask_s,
                                                                                                device=device)
        _, _, _, _, prediction_encoding_s = model.forward(x_s, initial_pos_s,
                                                                            dest=dest_s,
                                                                            mask=mask_s,
                                                                            device=device)
        score_fake = d_inlay(prediction_encoding_s)

        dest_recon_t, mu_t, var_t, interpolated_future_t, _ = model.forward(x_t, initial_pos_t,
                                                                                                dest=dest_t,
                                                                                                mask=mask_t,
                                                                                                device=device)
        _, _, _, _, prediction_encoding_t = model.forward(x_t, initial_pos_t,
                                                                            dest=dest_t,
                                                                            mask=mask_t,
                                                                            device=device)
        score_real = d_inlay(prediction_encoding_t)

        # 1.Discriminator
        d_loss_real, d_loss_fake = gan_d_loss(score_real, score_fake, mode='mse')
        d_inlay_loss = d_loss_real + d_loss_fake
        d_inlay_loss *= 0.05

        # 2.Generator
        g_inlay_loss = gan_g_loss(score_fake, mode='mse')
        g_inlay_loss *= 0.05

        # 3.Predictor
        rcl, kld, adl = calculate_loss(dest_s, dest_recon_s, mu_s, var_s, criterion, future_s, interpolated_future_s)
        loss = rcl + kld * hyper_params["kld_reg"] + adl * hyper_params["adl_reg"]


        # Metrics
        train_loss += loss.item()
        d_loss += d_inlay_loss
        g_loss += g_inlay_loss
        total_rcl += rcl.item()
        total_kld += kld.item()
        total_adl += adl.item()


        # 2.Generator
        optimizer.zero_grad()
        # g_inlay_loss = g_inlay_loss/g_inlay_loss.detach()*loss.detach()
        g_inlay_loss.backward(retain_graph=True)
        # print(f'&&&&&&&& g_loss:{model.encoder_past.layers[0].weight.grad} &&&&&&&&&&&&&&&&&&&&&&&&\n')
        optimizer.step()

        # 1.Discriminator
        optimizer_d_inlay.zero_grad()
        # d_inlay_loss = d_inlay_loss/d_inlay_loss.detach()*loss.detach()
        d_inlay_loss.backward()
        optimizer_d_inlay.step()

        # 3.predictor
        optimizer.zero_grad()
        # loss = loss/loss.detach()
        loss.backward()
        optimizer.step()
        # print(f'&&&&&&&& pred_loss:{model.encoder_past.layers[0].weight.grad} &&&&&&&&&&&&&&&&&&&&&&&&\n')

    return train_loss, d_loss, g_loss, total_rcl, total_kld, total_adl


def test(test_dataset, best_of_n=1):
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
            x = traj[:, :hyper_params['past_length'], :]
            y = traj[:, hyper_params['past_length']:, :]
            y = y.cpu().numpy()

            # reshape the data
            x = x.view(-1, x.shape[1] * x.shape[2])
            x = x.to(device)

            dest = y[:, -1, :]
            all_l2_errors_dest = []
            all_guesses = []
            for _ in range(best_of_n):
                dest_recon = model.forward(x, initial_pos, device=device)
                dest_recon = dest_recon.cpu().numpy()
                all_guesses.append(dest_recon)

                l2error_sample = np.linalg.norm(dest_recon - dest, axis=1)
                all_l2_errors_dest.append(l2error_sample)

            all_l2_errors_dest = np.array(all_l2_errors_dest)
            all_guesses = np.array(all_guesses)

            # average error
            l2error_avg_dest = np.mean(all_l2_errors_dest)

            # choosing the best guess
            indices = np.argmin(all_l2_errors_dest, axis=0)

            best_guess_dest = all_guesses[indices, np.arange(x.shape[0]), :]

            # taking the minimum error out of all guess
            l2error_dest = np.mean(np.min(all_l2_errors_dest, axis=0))

            best_guess_dest = torch.DoubleTensor(best_guess_dest).to(device)

            # using the best guess for interpolation
            interpolated_future = model.predict(x, best_guess_dest, mask, initial_pos)
            interpolated_future = interpolated_future.cpu().numpy()
            best_guess_dest = best_guess_dest.cpu().numpy()

            # final overall prediction
            predicted_future = np.concatenate((interpolated_future, best_guess_dest), axis=1)
            predicted_future = np.reshape(predicted_future, (-1, hyper_params['future_length'], 2))  # making sure
            # ADE error
            l2error_overall = np.mean(np.linalg.norm(y - predicted_future, axis=2))

            l2error_overall /= hyper_params["data_scale"]
            l2error_dest /= hyper_params["data_scale"]
            l2error_avg_dest /= hyper_params["data_scale"]

            total_ade += (l2error_overall * len(traj))
            total_fde += (l2error_dest * len(traj))

            print(
                'Test time error in destination best: {:0.3f} and mean: {:0.3f}'.format(l2error_dest, l2error_avg_dest))
            print('Test time error overall (ADE) best: {:0.3f}'.format(l2error_overall))

    return (total_ade / total_seen), (total_fde / total_seen), l2error_avg_dest


if __name__ == '__main__':
    start_time = time.time()
    dada_datasets = [
        'A2B', 'A2C', 'A2D', 'A2E',
        'B2A', 'B2C', 'B2D', 'B2E',
        'C2A', 'C2B', 'C2D', 'C2E',
        'D2A', 'D2B', 'D2C', 'D2E',
        'E2A', 'E2B', 'E2C', 'E2D']
    # dada_datasets = ['A2B']
    for subset in dada_datasets:
        model = PECNet_FLA(hyper_params["enc_past_size"], hyper_params["enc_dest_size"],
                           hyper_params["enc_latent_size"],
                           hyper_params["dec_size"], hyper_params["predictor_hidden_size"],
                           hyper_params['non_local_theta_size'], hyper_params['non_local_phi_size'],
                           hyper_params['non_local_g_size'], hyper_params["fdim"], hyper_params["zdim"],
                           hyper_params["nonlocal_pools"], hyper_params['non_local_dim'], hyper_params["sigma"],
                           hyper_params["past_length"], hyper_params["future_length"], args.verbose)
        model = model.double().to(device)

        d_inlay = Discriminator_inlay(2 * hyper_params["fdim"] + 2, 4 * hyper_params["fdim"] + 4).double().to(device)  # before/after SocialPooling
        # d_inlay = Discriminator_inlay(hyper_params["fdim"]+hyper_params['zdim'], 2*hyper_params["fdim"]+2*hyper_params['zdim']).double().to(device)  # before DLatent
        # d_inlay = Discriminator_inlay(2*hyper_params["fdim"], 4*hyper_params["fdim"]).double().to(device)  # before ELatent

        # Training settings
        optimizer = optim.Adam(model.parameters(), lr=hyper_params["learning_rate"])
        optimizer_d_inlay = optim.Adam(d_inlay.parameters(), lr=hyper_params["learning_rate"])

        train_dataset = SocialDatasetETHUCY(set_name=subset, set_type='train_origin', b_size=hyper_params["train_b_size"],
                                            t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"])
        test_dataset = SocialDatasetETHUCY(set_name=subset, set_type='val', b_size=hyper_params["test_b_size"],
                                           t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"])
        len_source, len_target = len(train_dataset.trajectory_batches), len(test_dataset.trajectory_batches)
        len_max = max(len_source, len_target)

        # shift origin and scale data
        for traj in train_dataset.trajectory_batches:
            traj -= traj[:, :1, :]
            traj *= hyper_params["data_scale"]
        for traj in test_dataset.trajectory_batches:
            traj -= traj[:, :1, :]
            traj *= hyper_params["data_scale"]

        best_test_loss = 50  # start saving after this threshold
        best_endpoint_loss = 50
        metrics = {'train_loss': [], 'd_loss': [], 'g_loss': []}
        N = hyper_params["n_values"]
        args.save_file = f'PECNET_social_model_{subset}.pt'

        for e in range(hyper_params['num_epochs']):
            train_loss, d_loss, g_loss, rcl, kld, adl = train(train_dataset, test_dataset)
            metrics['train_loss'].append(train_loss)
            metrics['d_loss'].append(d_loss)
            metrics['g_loss'].append(g_loss)

            test_loss, final_point_loss_best, final_point_loss_avg = test(test_dataset, best_of_n=N)

            print()
            print("Epoch: ", e)
            if best_test_loss > test_loss:
                print('################## BEST PERFORMANCE {:0.2f} ########'.format(test_loss))
                best_test_loss = test_loss
                if best_test_loss < 10.25:
                    save_path = '../checkpoint_FLA/' + args.save_file
                    torch.save({
                        'hyper_params': hyper_params,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metrics': metrics
                    }, save_path)
                    print("Saved model to:\n{}".format(save_path))


            if final_point_loss_best < best_endpoint_loss:
                best_endpoint_loss = final_point_loss_best

            print(subset)
            print("Train Loss", train_loss)
            print("Discriminator d_Loss", d_loss)
            print("Discriminator g_Loss", g_loss)
            # print("RCL", rcl)
            # print("KLD", kld)
            # print("ADL", adl)
            print("Test ADE", test_loss)
            print("Test Average FDE (Across  all samples)", final_point_loss_avg)
            # print("Test Min FDE", final_point_loss_best)
            print("Test Best ADE Loss So Far (N = {})".format(N), best_test_loss)
            print("Test Best Min FDE (N = {})".format(N), best_endpoint_loss)

    print(f'use time: {time.time() - start_time}')


# def solve_multi_loss(pred_grad, g_grad):
#     alpha = (torch.sum(g_grad) - torch.sum(pred_grad*g_grad)) / torch.sum((pred_grad-g_grad)**2)  # 基于向量
#     # alpha = (g_grad - pred_grad*g_grad) / ((pred_grad-g_grad)**2)  # 基于元素
#
#     alpha = torch.max(torch.min(alpha, torch.tensor(1)), torch.tensor(0))
#
#     return alpha*pred_grad + (torch.tensor(1)-alpha)*g_grad
#
#
# def train(train_dataset, test_dataset):
#     model.train()
#     d_inlay.train()
#     train_loss, g_loss, d_loss = 0, 0, 0
#     total_rcl, total_kld, total_adl = 0, 0, 0
#     criterion = nn.MSELoss()
#
#     source_iter_traj = iter(train_dataset.trajectory_batches)
#     source_iter_mask = iter(train_dataset.mask_batches)
#     source_iter_pose = iter(train_dataset.initial_pos_batches)
#
#     target_iter_traj = iter(test_dataset.trajectory_batches)
#     target_iter_mask = iter(test_dataset.mask_batches)
#     target_iter_pose = iter(test_dataset.initial_pos_batches)
#
#     for batch_index in range(len_max):
#         # Get data
#         if batch_index % len_source == 0:
#             del source_iter_traj
#             del source_iter_mask
#             del source_iter_pose
#             source_iter_traj = iter(train_dataset.trajectory_batches)
#             source_iter_mask = iter(train_dataset.mask_batches)
#             source_iter_pose = iter(train_dataset.initial_pos_batches)
#             # print('*** Source Iter is reset ***')
#         if batch_index % len_target == 0:
#             del target_iter_traj
#             del target_iter_mask
#             del target_iter_pose
#             target_iter_traj = iter(test_dataset.trajectory_batches)
#             target_iter_mask = iter(test_dataset.mask_batches)
#             target_iter_pose = iter(test_dataset.initial_pos_batches)
#             # print('*** Target Iter is reset ***')
#
#         traj_s = torch.DoubleTensor(next(source_iter_traj)).to(device)
#         mask_s = torch.DoubleTensor(next(source_iter_mask)).to(device)
#         initial_pos_s = torch.DoubleTensor(next(source_iter_pose)).to(device)
#
#         traj_t = torch.DoubleTensor(next(target_iter_traj)).to(device)
#         mask_t = torch.DoubleTensor(next(target_iter_mask)).to(device)
#         initial_pos_t = torch.DoubleTensor(next(target_iter_pose)).to(device)
#
#         x_s = traj_s[:, :hyper_params['past_length'], :]  # [batchsize, 8, 2]
#         y_s = traj_s[:, hyper_params['past_length']:, :]  # [batchsize, 12, 2]
#
#         x_t = traj_t[:, :hyper_params['past_length'], :]  # [batchsize, 8, 2]
#         y_t = traj_t[:, hyper_params['past_length']:, :]  # [batchsize, 12, 2]
#
#         x_s = x_s.contiguous().view(-1, x_s.shape[1] * x_s.shape[2])  # (x,y,x,y ... )
#         x_s = x_s.to(device)
#         dest_s = y_s[:, -1, :].to(device)  # [bs, 2]
#         future_s = y_s[:, :-1, :].contiguous().view(y_s.size(0), -1).to(device)  # [bs, (12-1)*2]
#
#         x_t = x_t.contiguous().view(-1, x_t.shape[1] * x_t.shape[2])  # (x,y,x,y ... )
#         x_t = x_t.to(device)
#         dest_t = y_t[:, -1, :].to(device)  # [bs, 2]
#         future_t = y_t[:, :-1, :].contiguous().view(y_t.size(0), -1).to(device)  # [bs, (12-1)*2]
#
#         dest_recon_s, mu_s, var_s, interpolated_future_s, _ = model.forward(x_s, initial_pos_s,
#                                                                             dest=dest_s,
#                                                                             mask=mask_s,
#                                                                             device=device)
#         _, _, _, _, prediction_encoding_s = model.forward(x_s, initial_pos_s,
#                                                           dest=dest_s,
#                                                           mask=mask_s,
#                                                           device=device)
#         score_fake = d_inlay(prediction_encoding_s)
#
#         dest_recon_t, mu_t, var_t, interpolated_future_t, _ = model.forward(x_t, initial_pos_t,
#                                                                             dest=dest_t,
#                                                                             mask=mask_t,
#                                                                             device=device)
#         _, _, _, _, prediction_encoding_t = model.forward(x_t, initial_pos_t,
#                                                           dest=dest_t,
#                                                           mask=mask_t,
#                                                           device=device)
#         score_real = d_inlay(prediction_encoding_t)
#
#         # 1.Discriminator
#         d_loss_real, d_loss_fake = gan_d_loss(score_real, score_fake, mode='mse')
#         d_inlay_loss = d_loss_real + d_loss_fake
#         d_inlay_loss *= 1.
#
#         # 2.Generator
#         g_inlay_loss = gan_g_loss(score_fake, mode='mse')
#         g_inlay_loss *= 1.
#
#         # 3.Predictor
#         rcl, kld, adl = calculate_loss(dest_s, dest_recon_s, mu_s, var_s, criterion, future_s, interpolated_future_s)
#         loss = rcl + kld * hyper_params["kld_reg"] + adl * hyper_params["adl_reg"]
#
#         # Metrics
#         train_loss += loss.item()
#         d_loss += d_inlay_loss
#         g_loss += g_inlay_loss
#         total_rcl += rcl.item()
#         total_kld += kld.item()
#         total_adl += adl.item()
#
#         # 2.Generator
#         optimizer_encoder.zero_grad()
#         g_inlay_loss.backward(retain_graph=True)
#         # print(f'&&&&&&&& g_loss:{model.encoder_past.layers[0].weight.grad} &&&&&&&&&&&&&&&&&&&&&&&&\n')
#         # optimizer_encoder.step()
#         encoder_module_g_grad = [[] for _ in range(encoder_module_num)]  # [[grad_00, grad_01, ...], [grad10, grad11, ...], ...]
#         for encoder_module_idx in range(encoder_module_num):
#             encoder_module = encoder_modules[encoder_module_idx]
#             layer_num = len(encoder_module.layers)
#             for layer_idx in range(layer_num):
#                 encoder_module_g_grad[encoder_module_idx].append(encoder_module.layers[layer_idx].weight.grad.clone().detach())
#                 # print('&&&&&&&&&&&&&&&&&&&&&&&&&', encoder_module_g_grad[encoder_module_idx][layer_idx].shape)
#
#         # 1.Discriminator
#         optimizer_d_inlay.zero_grad()
#         d_inlay_loss.backward()
#         optimizer_d_inlay.step()
#
#         # 3.predictor
#         optimizer_encoder.zero_grad()
#         optimizer_decoder.zero_grad()
#         loss.backward()
#         optimizer_decoder.step()
#
#         encoder_module_pred_grad = [[] for _ in range(encoder_module_num)]  # [[grad_00, grad_01, ...], [grad10, grad11, ...], ...]
#         for encoder_module_idx in range(encoder_module_num):
#             encoder_module = encoder_modules[encoder_module_idx]
#             layer_num = len(encoder_module.layers)
#             for layer_idx in range(layer_num):
#                 encoder_module_pred_grad[encoder_module_idx].append(encoder_module.layers[layer_idx].weight.grad.clone().detach())
#                 encoder_module.layers[layer_idx].weight.grad = solve_multi_loss(
#                                                                 encoder_module_pred_grad[encoder_module_idx][layer_idx],
#                                                                 encoder_module_g_grad[encoder_module_idx][layer_idx])
#                 # encoder_module.layers[layer_idx].weight.grad = encoder_module_pred_grad[encoder_module_idx][layer_idx]
#                 # encoder_module.layers[layer_idx].weight.grad = encoder_module.layers[layer_idx].weight.grad.clone().detach()
#
#                 # pred_grad = encoder_module_pred_grad[encoder_module_idx][layer_idx]
#                 # g_grad = encoder_module_g_grad[encoder_module_idx][layer_idx]
#                 # alpha = (torch.sum(g_grad) - torch.sum(pred_grad * g_grad)) / torch.sum((pred_grad - g_grad) ** 2)  # 基于向量
#                 # # alpha = (g_grad - pred_grad*g_grad) / ((pred_grad-g_grad)**2)  # 基于元素
#                 # alpha = torch.max(torch.min(alpha, torch.tensor(1)), torch.tensor(0))
#                 #
#                 # encoder_module.layers[layer_idx].weight.grad = alpha*pred_grad + (1-alpha)*g_grad
#
#         optimizer_encoder.step()
#
#         # print(f'&&&&&&&& pred_loss:{model.encoder_past.layers[0].weight.grad} &&&&&&&&&&&&&&&&&&&&&&&&\n')
#
#     return train_loss, d_loss, g_loss, total_rcl, total_kld, total_adl
#
#
# def test(test_dataset, best_of_n=1):
#     model.eval()
#     assert best_of_n >= 1 and type(best_of_n) == int
#     test_loss = 0
#
#     with torch.no_grad():
#         total_seen = 0
#         total_ade = 0
#         total_fde = 0
#         for i, (traj, mask, initial_pos) in enumerate(
#                 zip(test_dataset.trajectory_batches, test_dataset.mask_batches, test_dataset.initial_pos_batches)):
#             total_seen += len(traj)
#             traj, mask, initial_pos = torch.DoubleTensor(traj).to(device), torch.DoubleTensor(mask).to(
#                 device), torch.DoubleTensor(initial_pos).to(device)
#             x = traj[:, :hyper_params['past_length'], :]
#             y = traj[:, hyper_params['past_length']:, :]
#             y = y.cpu().numpy()
#
#             # reshape the data
#             x = x.view(-1, x.shape[1] * x.shape[2])
#             x = x.to(device)
#
#             dest = y[:, -1, :]
#             all_l2_errors_dest = []
#             all_guesses = []
#             for _ in range(best_of_n):
#                 dest_recon = model.forward(x, initial_pos, device=device)
#                 dest_recon = dest_recon.cpu().numpy()
#                 all_guesses.append(dest_recon)
#
#                 l2error_sample = np.linalg.norm(dest_recon - dest, axis=1)
#                 all_l2_errors_dest.append(l2error_sample)
#
#             all_l2_errors_dest = np.array(all_l2_errors_dest)
#             all_guesses = np.array(all_guesses)
#
#             # average error
#             l2error_avg_dest = np.mean(all_l2_errors_dest)
#
#             # choosing the best guess
#             indices = np.argmin(all_l2_errors_dest, axis=0)
#
#             best_guess_dest = all_guesses[indices, np.arange(x.shape[0]), :]
#
#             # taking the minimum error out of all guess
#             l2error_dest = np.mean(np.min(all_l2_errors_dest, axis=0))
#
#             best_guess_dest = torch.DoubleTensor(best_guess_dest).to(device)
#
#             # using the best guess for interpolation
#             interpolated_future = model.predict(x, best_guess_dest, mask, initial_pos)
#             interpolated_future = interpolated_future.cpu().numpy()
#             best_guess_dest = best_guess_dest.cpu().numpy()
#
#             # final overall prediction
#             predicted_future = np.concatenate((interpolated_future, best_guess_dest), axis=1)
#             predicted_future = np.reshape(predicted_future, (-1, hyper_params['future_length'], 2))  # making sure
#             # ADE error
#             l2error_overall = np.mean(np.linalg.norm(y - predicted_future, axis=2))
#
#             l2error_overall /= hyper_params["data_scale"]
#             l2error_dest /= hyper_params["data_scale"]
#             l2error_avg_dest /= hyper_params["data_scale"]
#
#             total_ade += (l2error_overall * len(traj))
#             total_fde += (l2error_dest * len(traj))
#
#             print(
#                 'Test time error in destination best: {:0.3f} and mean: {:0.3f}'.format(l2error_dest, l2error_avg_dest))
#             print('Test time error overall (ADE) best: {:0.3f}'.format(l2error_overall))
#
#     return (total_ade / total_seen), (total_fde / total_seen), l2error_avg_dest
#
#
# if __name__ == '__main__':
#     start_time = time.time()
#     # dada_datasets = [
#     #     'A2B', 'A2C', 'A2D', 'A2E',
#     #     'B2A', 'B2C', 'B2D', 'B2E',
#     #
#     #     'D2A', 'D2B', 'D2C', 'D2E',
#     #     'E2A', 'E2B', 'E2C', 'E2D']
#     # dada_datasets = ['C2A', 'C2B', 'C2D', 'C2E']
#     # dada_datasets = ['A2D', 'A2E', 'C2B', 'C2E', 'D2B', 'D2E', 'E2B']
#     # dada_datasets = ['A2D', 'A2E', 'D2B', 'D2E', 'E2B']
#     dada_datasets = ['A2B']
#     for subset in dada_datasets:
#         model = PECNet_FLA(hyper_params["enc_past_size"], hyper_params["enc_dest_size"],
#                            hyper_params["enc_latent_size"],
#                            hyper_params["dec_size"], hyper_params["predictor_hidden_size"],
#                            hyper_params['non_local_theta_size'], hyper_params['non_local_phi_size'],
#                            hyper_params['non_local_g_size'], hyper_params["fdim"], hyper_params["zdim"],
#                            hyper_params["nonlocal_pools"], hyper_params['non_local_dim'], hyper_params["sigma"],
#                            hyper_params["past_length"], hyper_params["future_length"], args.verbose)
#         model = model.double().to(device)
#
#         encoder_modules = [model.encoder_past,
#                            model.encoder_dest,
#                            model.encoder_latent,
#                            model.decoder,
#                            model.non_local_g,
#                            model.non_local_theta,
#                            model.non_local_phi]
#         encoder_module_num = len(encoder_modules)
#
#         decoder_modules = [model.predictor]
#
#         d_inlay = Discriminator_inlay(2 * hyper_params["fdim"] + 2, 4 * hyper_params["fdim"] + 4).double().to(device)  # before/after SocialPooling
#         # d_inlay = Discriminator_inlay(hyper_params["fdim"]+hyper_params['zdim'], 2*hyper_params["fdim"]+2*hyper_params['zdim']).double().to(device)  # before DLatent
#         # d_inlay = Discriminator_inlay(2*hyper_params["fdim"], 4*hyper_params["fdim"]).double().to(device)  # before ELatent
#
#         # Training settings
#         optimizer_encoder = optim.Adam([{'params': encoder_module.parameters()} for encoder_module in encoder_modules],
#                                        lr=hyper_params["learning_rate"])
#         optimizer_decoder = optim.Adam([{'params': decoder_module.parameters()} for decoder_module in decoder_modules],
#                                        lr=hyper_params["learning_rate"])
#         optimizer_d_inlay = optim.Adam(d_inlay.parameters(), lr=hyper_params["learning_rate"])
#
#         train_dataset = SocialDatasetETHUCY(set_name=subset, set_type='train', b_size=hyper_params["train_b_size"],
#                                             t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"])
#         test_dataset = SocialDatasetETHUCY(set_name=subset, set_type='val', b_size=hyper_params["test_b_size"],
#                                            t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"])
#         len_source, len_target = len(train_dataset.trajectory_batches), len(test_dataset.trajectory_batches)
#         len_max = max(len_source, len_target)
#
#         # shift origin and scale data
#         for traj in train_dataset.trajectory_batches:
#             traj -= traj[:, :1, :]
#             traj *= hyper_params["data_scale"]
#         for traj in test_dataset.trajectory_batches:
#             traj -= traj[:, :1, :]
#             traj *= hyper_params["data_scale"]
#
#         best_test_loss = 50  # start saving after this threshold
#         best_endpoint_loss = 50
#         metrics = {'train_loss': [], 'd_loss': [], 'g_loss': []}
#         N = hyper_params["n_values"]
#         args.save_file = f'PECNET_social_model_{subset}_DADA.pt'
#
#         for e in range(hyper_params['num_epochs']):
#             train_loss, d_loss, g_loss, rcl, kld, adl = train(train_dataset, test_dataset)
#             metrics['train_loss'].append(train_loss)
#             metrics['d_loss'].append(d_loss)
#             metrics['g_loss'].append(g_loss)
#
#             test_loss, final_point_loss_best, final_point_loss_avg = test(test_dataset, best_of_n=N)
#
#             print()
#             print("Epoch: ", e)
#             if best_test_loss > test_loss:
#                 print('################## BEST PERFORMANCE {:0.2f} ########'.format(test_loss))
#                 best_test_loss = test_loss
#                 if best_test_loss < 10.25:
#                     save_path = '../checkpoint_DADA/after_SocialPooling_DomainDiffVisual/' + args.save_file
#                     torch.save({
#                         'hyper_params': hyper_params,
#                         'model_state_dict': model.state_dict(),
#                         'optimizer_encoder_state_dict': optimizer_encoder.state_dict(),
#                         'optimizer_decoder_state_dict': optimizer_decoder.state_dict(),
#                         'metrics': metrics
#                     }, save_path)
#                     print("Saved model to:\n{}".format(save_path))
#
#             # if e == 150:
#             #     print('################## BEST PERFORMANCE {:0.2f} ########'.format(test_loss))
#             #     best_test_loss = test_loss
#             #     if True:
#             #         save_path = '../checkpoint_DADA/after_SocialPooling/' + args.save_file
#             #         torch.save({
#             #             'hyper_params': hyper_params,
#             #             'model_state_dict': model.state_dict(),
#             #             'optimizer_encoder_state_dict': optimizer_encoder.state_dict(),
#             #             'optimizer_decoder_state_dict': optimizer_decoder.state_dict(),
#             #             'metrics': metrics
#             #         }, save_path)
#             #         print("Saved model to:\n{}".format(save_path))
#
#
#             if final_point_loss_best < best_endpoint_loss:
#                 best_endpoint_loss = final_point_loss_best
#
#             print(subset)
#             print("Train Loss", train_loss)
#             print("Discriminator d_Loss", d_loss)
#             print("Discriminator g_Loss", g_loss)
#             print("RCL", rcl)
#             print("KLD", kld)
#             print("ADL", adl)
#             print("Test ADE", test_loss)
#             print("Test Average FDE (Across  all samples)", final_point_loss_avg)
#             # print("Test Min FDE", final_point_loss_best)
#             print("Test Best ADE Loss So Far (N = {})".format(N), best_test_loss)
#             print("Test Best Min FDE (N = {})".format(N), best_endpoint_loss)
#
#     print(f'use time: {time.time() - start_time}')
