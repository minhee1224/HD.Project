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


def emg_preprocessing(mat_path):
    # 마지막에 time profiling 진행하고 code optimization 진행
    path = mat_path
    # load data as a form of dictionary
    mat_data_1 = loadmat(path)
    #######################################################################
    ####### DATA DESCRIPTIONS #############################################
    #######################################################################
    # The number of data: 6752691 (s1 기준)
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
    norm_extensors = norm_extensors.permute(1, 0, 2, 3)
    norm_flexors = norm_flexors.permute(1, 0, 2, 3)
    w_emg = torch.cat([norm_extensors, norm_flexors], dim=3)
    re_w_label = torch.IntTensor(re_w_label)

    return window_sample, w_emg, re_w_label
