import pickle
import numpy as np
from IPython import embed
import csv
import torch
from torch import nn
from torch.utils import data
import os
import sys
sys.path.append('../')
# from eth_ucy_utils import *

def get_pickle_path(set_name, set_type):
	_dir = os.path.dirname(__file__)
	if _dir:
		return _dir + '/datasets/{0}/{1}/{0}_{1}.p'.format(set_name, set_type)
	else:
		return './datasets/{0}/{1}/{0}_{1}.p'.format(set_name, set_type)

def find_min_time(t1, t2):
	'''given two time frame arrays, find then min dist between starts'''
	min_d = 999999999
	for t in t2:
		if abs(t1[0]-t)<min_d:
			min_d = abs(t1[0]-t)

	for t in t1:
		if abs(t2[0]-t)<min_d:
			min_d = abs(t2[0]-t)

	return min_d

def find_min_dist(p1x, p1y, p2x, p2y):
	'''given two time frame arrays, find then min dist between starts'''
	min_d = 999999999
	for i in range(len(p1x)):
		for j in range(len(p1x)):
			if ((p2x[i]-p1x[j])**2 + (p2y[i]-p1y[j])**2)**0.5 < min_d:
				min_d = ((p2x[i]-p1x[j])**2 + (p2y[i]-p1y[j])**2)**0.5

	return min_d

def social_and_temporal_filter(p1_key, p2_key, all_data_dict, time_thresh=10, dist_tresh=2):
	p1_traj, p2_traj = np.array(all_data_dict[p1_key]), np.array(all_data_dict[p2_key])
	p1_time, p2_time = p1_traj[:,0], p2_traj[:,0]
	p1_x, p2_x = p1_traj[:,2], p2_traj[:,2]
	p1_y, p2_y = p1_traj[:,3], p2_traj[:,3]


	if all_data_dict[p1_key][0][4]!=all_data_dict[p2_key][0][4]: #adding the condition that they must be from the same environment
		return False
	if all_data_dict[p1_key][0][1]==all_data_dict[p2_key][0][1]: #if they are the same person id, no self loops
		return False
	if find_min_time(p1_time, p2_time)>time_thresh:
		return False
	if find_min_dist(p1_x, p1_y, p2_x, p2_y)>dist_tresh:
		return False

	return True

def mark_similar(mask, sim_list):
	for i in range(len(sim_list)):
		for j in range(len(sim_list)):
			mask[sim_list[i]][sim_list[j]] = 1

def socially_pickle_data(batch_size=512, time_thresh=0, dist_tresh=10):
	print("pickling...")
	for scene in ['eth', 'univ', 'zara1', 'zara2', 'hotel']:
		for j in ['test']:
			path = get_pickle_path(scene, j)
			data = pickle.load(open(path, "rb"))

			full_dataset = []
			full_masks = []

			current_batch = []
			mask_batch = [[0 for i in range(int(batch_size*2))] for j in range(int(batch_size*2))]
			current_size = 0
			social_id = 0

			data_by_id = {}
			person_id = 0
			for d in data:
				data_by_id[person_id] = d
				person_id += 1

			all_data_dict = data_by_id.copy()

			print("Total People: ", len(list(data_by_id.keys())))
			while len(list(data_by_id.keys()))>0:
				print(len(list(data_by_id.keys())))
				related_list = []
				curr_keys = list(data_by_id.keys())

				if current_size<batch_size:
					pass
				else:
					full_dataset.append(current_batch.copy())
					mask_batch = np.array(mask_batch)
					full_masks.append(mask_batch[0:len(current_batch), 0:len(current_batch)])
					current_size = 0
					social_id = 0
					current_batch = []
					mask_batch = [[0 for i in range(int(batch_size*2))] for j in range(int(batch_size*2))]

				current_batch.append((all_data_dict[curr_keys[0]]))
				related_list.append(current_size)
				current_size+=1
				del data_by_id[curr_keys[0]]

				for i in range(1, len(curr_keys)):
					if social_and_temporal_filter(curr_keys[0], curr_keys[i], all_data_dict, time_thresh=time_thresh, dist_tresh=dist_tresh):
						current_batch.append((all_data_dict[curr_keys[i]]))
						related_list.append(current_size)
						current_size+=1
						del data_by_id[curr_keys[i]]

				mark_similar(mask_batch, related_list)
				social_id +=1

			full_dataset.append(current_batch)
			mask_batch = np.array(mask_batch)
			full_masks.append(mask_batch[0:len(current_batch),0:len(current_batch)])

			all_data = [full_dataset, full_masks]
			save_name = "social_eth_ucy_dataset/social_" + str(scene) + "_" + str(j) + "_" + str(batch_size) + "_" + str(time_thresh) + "_" + str(dist_tresh) + ".pickle"
			with open(save_name, 'wb') as f:
				pickle.dump(all_data, f)

# socially_pickle_data(batch_size=4096, time_thresh=0, dist_tresh=50)
# socially_pickle_data(batch_size=256, time_thresh=0, dist_tresh=50)

def initial_pos(traj_batches):
	batches = []
	for b in traj_batches:
		starting_pos = b[:,7,:].copy()/1000 #starting pos is end of past, start of future. scaled down.
		batches.append(starting_pos)

	return batches

def calculate_loss(x, reconstructed_x, mean, log_var, criterion, future, interpolated_future):
	# reconstruction loss
	RCL_dest = criterion(x, reconstructed_x)

	ADL_traj = criterion(future, interpolated_future) # better with l2 loss

	# kl divergence loss
	KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

	return RCL_dest, KLD, ADL_traj

# loss of G
def gan_g_loss(scores_fake,  # [nums, 1]
               mode='mse'):
    if mode == 'bce':
        loss_func = nn.BCEWithLogitsLoss()
    elif mode == 'mse':
        loss_func = nn.MSELoss()

    y_real = torch.ones_like(scores_fake)  # * random.uniform(0.7, 1.2)
    return loss_func(scores_fake, y_real)  # make the fake more real, [1]

# loss of D
def gan_d_loss(scores_real,  # [nums, 1]
               scores_fake,  # [nums, 1]
               mode='mse'):
    if mode == 'bce':
        loss_func = nn.BCEWithLogitsLoss()
    elif mode == 'mse':
        loss_func = nn.MSELoss()

    y_real = torch.ones_like(scores_real)  # * random.uniform(0.7, 1.2)  # [nums] ~ U(0.7, 1.2)
    y_fake = torch.zeros_like(scores_fake)  # * random.uniform(0, 0.3)  # [nums] = [0.0, 0.0, ..., 0.0]

    loss_real = loss_func(scores_real, y_real)  # make the real more real
    loss_fake = loss_func(scores_fake, y_fake)  # make the fake faker

    return loss_real, loss_fake  # [1], [1]


class SocialDatasetETHUCY(data.Dataset):
	def __init__(self, set_name=None, set_type=None, b_size=512, t_tresh=0, d_tresh=10):
		'Initialization'
		load_name = "../social_pool_data/social_" + set_name + "_" + set_type + "_" + str(b_size) + "_" + str(t_tresh) + "_" + str(d_tresh) + ".pickle"
		with open(load_name, 'rb') as f:
			data = pickle.load(f)

		traj, masks = data
		traj_new = []
		for t in traj:
			t = np.array(t)
			t = t[:,:,2:4]
			traj_new.append(t)

			if set_type=="train":
				#augment training set with reversed tracklets...
				reverse_t = np.flip(t, axis=1).copy()
				traj_new.append(reverse_t)

		#comment
		masks_new = []
		for m in masks:
			masks_new.append(m)

			if set_type=="train":
				#add second time for the reversed tracklets...
				masks_new.append(m)

		traj_new = np.array(traj_new)
		masks_new = np.array(masks_new)
		self.trajectory_batches = traj_new.copy()  # [num_of_batch, batch_size, 20, 2]
		self.mask_batches = masks_new.copy()

		#for relative positioning
		self.initial_pos_batches = np.array(initial_pos(self.trajectory_batches))  # [num_of_batch, batch_size, 2], it is scaled py 1/1000

		print("Initialized social dataloader for ucy/eth...")


if __name__ == '__main__':
	print(os.path.dirname(__file__))