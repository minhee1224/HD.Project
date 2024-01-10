# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 20:20:36 2023

@author: mleem
"""

import scipy.io as sio
import mat73
import h5py
import mat4py
import os
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
    mat_data_1 = mat73.loadmat(path)
    # mat_data_1 = mat4py.loadmat(path)
    # mat_data_1 = sio.loadmat(path)
    # read by h5py
    # with h5py.File(path, mode='r+') as F:
    #     label_1 = torch.Tensor(np.array(F['adjusted_class']))
    #     label_1 = label_1.unsqueeze(1)
    #     extensors_1 = torch.Tensor(np.array(F['emg_extensors']))
    #     flexors_1 = torch.Tensor(np.array(F['emg_flexors']))
    #######################################################################
    ####### DATA DESCRIPTIONS #############################################
    #######################################################################
    # The number of data: 6752691 (s1 기준)
    # adjusted_label: classify data by 16 basic dof
    # adjusted_class: 1 ~ 65 classes, 0 for rest
    # repetition -> do not assign another feature!! (temporal decision)
    # Do not reflect outlier processing!!
    #######################################################################
    # class label
    # label_1 = torch.Tensor(mat_data_1['adjusted_class'])
    # label_1 = label_1.unsqueeze(1)
    # force label
    label_f_1 = torch.Tensor(mat_data_1['force'])
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
    total_sample = len(label_f_1)
    label_size = 2.5 # ms (200 Hz 기준)
    label_width_ratio = float(int(label_size)/int(time_advance))
    ######################################
    ######################################
    # implementation
    increment = float(1000/sampling_freq) # ms
    avg_sample = int(np.floor(float((2*label_size)/increment)))
    window_sample = int(np.floor(float(window_size/increment)))
    predict_sample = int(np.floor(float(time_advance/increment)))
    overlapping_sample = int(np.floor(float((1 - overlapping_ratio)*window_sample)))
    predict_num = int(np.floor(
        (total_sample - window_sample - predict_sample *
         (1 + label_width_ratio)) / overlapping_sample))
    total_length = int(window_sample + predict_sample *
                       (1 + label_width_ratio) +
                       overlapping_sample*predict_num)
    predict_force_sample = total_length - overlapping_sample * predict_num -\
        window_sample
    #####################################
    # sliding window division
    # emg
    # t_emg_indexing = torch.from_numpy(
    #     np.arange(0, total_length - predict_sample))
    # label
    # t_label_indexing = torch.from_numpy(np.arange(window_sample, total_length))
    # indexing
    # label_1 = torch.index_select(label_1, 0, torch.from_numpy(
    #     np.arange(window_sample, total_length)))
    # force_label 
    label_f_1 = torch.index_select(label_f_1, 0, torch.from_numpy(
        np.arange(window_sample, total_length)))
    extensors_1 = torch.index_select(extensors_1, 0, torch.from_numpy(
        np.arange(0, total_length - predict_force_sample)))
    flexors_1 = torch.index_select(flexors_1, 0, torch.from_numpy(
        np.arange(0, total_length - predict_force_sample)))
    # for loop - window division
    extensors_1 = extensors_1.unfold(0, window_sample, overlapping_sample)
    flexors_1 = flexors_1.unfold(0, window_sample, overlapping_sample)
    label_f_1 = label_f_1.unfold(0, predict_force_sample, overlapping_sample)
    # permute for dimension matching
    extensors_1 = extensors_1.permute(3, 0, 1, 2)
    flexors_1 = flexors_1.permute(3, 0, 1, 2)
    label_f_1 = label_f_1.permute(2, 0, 1)
    ######################################
    # window-wise, electrode-wise normalization (mean, std)
    # mean, std calculation 
    # normalize mean 0, std 1
    normalize_extensors = transforms.Normalize(mean = torch.mean(
        extensors_1, dim=0), std = torch.std(extensors_1, dim=0))
    normalize_flexors = transforms.Normalize(mean = torch.mean(
        flexors_1, dim=0), std = torch.std(flexors_1, dim=0))
    extensors_1 = normalize_extensors(extensors_1)
    flexors_1 = normalize_flexors(flexors_1)
    #######################################
    # Re-labelling (force average) ## 여기 다시 보기!!!!!!
    avg_w_label = label_f_1[-avg_sample:, :, :]
    avg_w_label = torch.mean(avg_w_label, dim=0)
    #######################################
    # Dataloader (train, test)
    extensors_1 = extensors_1.permute(1, 0, 2, 3)
    flexors_1 = flexors_1.permute(1, 0, 2, 3)
    w_emg = torch.cat([extensors_1, flexors_1], dim=3)

    return window_sample, w_emg, avg_w_label
