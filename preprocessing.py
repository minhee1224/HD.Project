# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 20:20:36 2023

@author: mleem
"""

from mat73 import loadmat
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torchvision import transforms
from torch.utils.data import Subset, Dataset, DataLoader, WeightedRandomSampler

from model import *
from config import *

# 마지막에 time profiling 진행하고 code optimization 진행
path = "./65 isometric hand gestures/s1.mat"
# load data as a form of dictionary
mat_data_1 = loadmat(path)
#######################################################################
####### DATA DESCRIPTIONS #############################################
#######################################################################
# The number of data: 6752691
# adjusted_label: classify data by 16 basic dof
# adjusted_class: 1 ~ 65 classes, 0 for rest
# repetition -> do not assign another feature!! (temporal decision)
# Do not reflect outlier processing!!
#######################################################################
label_1 = torch.Tensor(mat_data_1['adjusted_class'])
label_1 = label_1.unsqueeze(1)
extensors_1 = torch.Tensor(mat_data_1['emg_extensors'])
flexors_1 = torch.Tensor(mat_data_1['emg_flexors'])
########################################################################
# Sliding Window Normalization
# Window size: 200 ms, 410 samples
# Overlapping size: 150 ms, 307 samples
# Time advance: 100ms, 205 samples
######################################
######################################
# PARAMETERS
sampling_freq = 2048
window_size = 200 # ms
overlapping_ratio = float(0.75)
time_advance = 100 # ms
total_sample = len(label_1)
######################################
######################################
# implementation
increment = float(1000/sampling_freq) # ms
window_sample = int(np.floor(float(window_size/increment)))
predict_sample = int(np.floor(float(time_advance/increment)))
overlapping_sample = int(np.floor(float((1 - overlapping_ratio)*window_sample)))
predict_num = int(np.floor(
    (total_sample - window_sample - predict_sample) / overlapping_sample))
total_length = int(window_sample + predict_sample +\
    overlapping_sample*predict_num)
#####################################
# sliding window division
# emg
emg_indexing = np.arange(0, total_length - predict_sample)
t_emg_indexing = torch.from_numpy(emg_indexing)
# label
label_indexing = np.arange(window_sample, total_length)
t_label_indexing = torch.from_numpy(label_indexing)
# indexing
label_1 = torch.index_select(label_1, 0, t_label_indexing)
extensors_1 = torch.index_select(extensors_1, 0, t_emg_indexing)
flexors_1 = torch.index_select(flexors_1, 0, t_emg_indexing)
# for loop - window division
w_extensors = extensors_1.unfold(0, window_sample, overlapping_sample)
w_flexors = flexors_1.unfold(0, window_sample, overlapping_sample)
w_label = label_1.unfold(0, predict_sample, overlapping_sample)
# permute for dimension matching
w_extensors = w_extensors.permute(3, 0, 1, 2)
w_flexors = w_flexors.permute(3, 0, 1, 2)
w_label = w_label.permute(0, 2, 1)
######################################
# window-wise, electrode-wise normalization (mean, std)
# mean, std calculation
extensors_mean = torch.mean(w_extensors, dim=0)
extensors_std = torch.std(w_extensors, dim=0)
flexors_mean = torch.mean(w_flexors, dim=0)
flexors_std = torch.std(w_flexors, dim=0)
# normalize mean 0, std 1
normalize_extensors = transforms.Normalize(mean = extensors_mean,
                                           std = extensors_std)
normalize_flexors = transforms.Normalize(mean = flexors_mean,
                                         std = flexors_std)
norm_extensors = normalize_extensors(w_extensors)
norm_flexors = normalize_flexors(w_flexors)
#######################################
# Re-labelling
w_label = w_label.squeeze(2)
re_w_label = np.array([])
for l in w_label:
    nonzero_len = len((l > 0).nonzero())
    if nonzero_len != 0:
        if nonzero_len == len(l):
            re_w_label = np.append(re_w_label, np.array([int(l[0])]))
        if nonzero_len < len(l):
            nonzero_ratio = float(nonzero_len/len(l))
            if l[0] > 0:
                re_w_label = np.append(re_w_label, np.array([int(l[0])]))
            else:
                re_w_label = np.append(re_w_label,
                                       np.array([int(l[len(l) - 1])]))
    else:
        re_w_label = np.append(re_w_label, np.array([int(0.0)]))
#######################################
# Dataloader (train, test)
w_extensors = w_extensors.permute(1, 0, 2, 3)
w_flexors = w_flexors.permute(1, 0, 2, 3)
w_emg = torch.cat([w_extensors, w_flexors], dim=3)
re_w_label = torch.IntTensor(re_w_label)
# # dataset
# dataset = CustomDataset(w_emg, re_w_label)
# # dataset division
# dataset_size = len(dataset)
# SSS = StratifiedShuffleSplit(n_splits=1, test_size=TEST_RATIO,
#                                random_state=RANDOM_STATE)
# for train_index, test_index in SSS.split(
#         np.arange(len(dataset)), np.array(dataset[:][1])):
#     pass
# train_dataset = Subset(dataset, train_index)
# test_dataset = Subset(dataset, test_index)
# #######################################
# # same number data for each label
# counter = Counter(re_w_label)
# # min_num = counter[sorted(counter, key=counter.get)[0]]
# label_weights = [len(re_w_label) / list(counter.values())[i]
#                  for i in range(len(counter))]
# train_weights = [label_weights[int(re_w_label[i])] for i in train_index]
# test_weights = [label_weights[int(re_w_label[i])] for i in test_index]
# train_sampler = WeightedRandomSampler(torch.DoubleTensor(train_weights),
#                                       len(train_index))
# test_sampler = WeightedRandomSampler(torch.DoubleTensor(test_weights),
#                                      len(test_index))
# # dataloader implementation
# train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
#                               sampler = train_sampler, drop_last=True,
#                               collate_fn=dataset.collate_fn)
# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
#                              sampler = test_sampler, drop_last=True,
#                              collate_fn=dataset.collate_fn)

###############################################################################
###############################################################################
# TRAINING
#######################################
opt = arg_parse()
wandb.init(project=opt.prj_name, entity='minheelee', config=vars(opt))
config=wandb.config
# Hyperparameter setting
# TEST_RATIO = 0.25
# RANDOM_STATE = 42
# BATCH_SIZE = 16
# EPOCHS = 300
# LR = 0.001
# N_WARMUP_STEPS = 10
# DECAY_RATE = 0.98
# DROPOUT_P = 0.1

# Device setting
DEVICE = torch.device('cuda') \
    if torch.cuda.is_available else torch.device('cpu')
print("Using PyTorch version: {}, Device: {}".format(
    torch.__version__, DEVICE))

# Model parameter setting
model_dir = "./ViT_model/ViT_LR%s.pt" % str(opt.LR)
# MODEL_DIM = 32
# HIDDEN_DIM = 128
# N_CLASS = 66 # FIXED
# N_HEAD = 8
# N_LAYER = 12
N_PATCH = window_sample # FIXED


test_loss, history = ViTtraining(model_dir, emg=w_emg, label=re_w_label,
                                 patch_size=int(8), MODEL_DIM=opt.model_dim,
                                 HIDDEN_DIM=opt.hidden_dim, N_CLASS=opt.n_class,
                                 N_HEAD=opt.n_head, N_LAYER=opt.n_layer,
                                 N_PATCH=N_PATCH, DROPOUT_P=opt.dropout_p,
                                 TEST_RATIO=opt.TEST_RATIO,
                                 RANDOM_STATE=opt.RANDOM_STATE,
                                 BATCH_SIZE=opt.BATCH_SIZE, EPOCHS=opt.EPOCHS,
                                 DEVICE=DEVICE, LR=opt.LR,
                                 N_WARMUP_STEPS=opt.n_warmup_steps,
                                 DECAY_RATE=opt.decay_rate)
plot_history(history, "ViT_LR_%s_" % str(opt.LR))
