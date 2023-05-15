# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:04:58 2023

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
import wandb

from preprocessing import *
from config import *
from model import *


def main(window_sample, w_emg, re_w_label):

    # path = "./isometric_hand_gestures/s1.mat"
    # window_sample, w_emg, re_w_label = emg_preprocessing(path)
    #########################################
    #########################################
    # TRAINING
    #######################################
    # LR_list = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    # for LR in LR_list:

    # opt = arg_parse()
    wandb.init(project="HD_ViT", entity='minheelee', config=hyperparameter_defaults)
    w_config=wandb.config
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
    model_dir = "./ViT_model/ViT_LR%s.pt" % str(w_config.LR)
    # MODEL_DIM = 32
    # HIDDEN_DIM = 128
    # N_CLASS = 66 # FIXED
    # N_HEAD = 8
    # N_LAYER = 12
    N_PATCH = window_sample # FIXED


    test_loss, history = ViTtraining(w_config, model_dir, emg=w_emg,
                                     label=re_w_label, patch_size=int(8),
                                     MODEL_DIM=w_config.model_dim,
                                     HIDDEN_DIM=w_config.hidden_dim,
                                     N_CLASS=w_config.n_class,
                                     N_HEAD=w_config.n_head,
                                     N_LAYER=w_config.n_layer,
                                     N_PATCH=N_PATCH,
                                     DROPOUT_P=w_config.dropout_p,
                                     TEST_RATIO=w_config.TEST_RATIO,
                                     RANDOM_STATE=w_config.RANDOM_STATE,
                                     BATCH_SIZE=w_config.BATCH_SIZE,
                                     EPOCHS=w_config.EPOCHS,
                                     DEVICE=DEVICE, LR=w_config.LR,
                                     N_WARMUP_STEPS=w_config.n_warmup_steps,
                                     DECAY_RATE=w_config.decay_rate)
    plot_history(history, "ViT_LR_%s_" % str(w_config.LR))
    wandb.run.finish()

if __name__ == "__main__":

    path = "./isometric_hand_gestures/s1.mat"
    window_sample, w_emg, re_w_label = emg_preprocessing(path)
    
    sweep_id = wandb.sweep(sweep_config, project="HD_ViT", entity="minheelee")
    wandb.agent(sweep_id, main(window_sample, w_emg, re_w_label), count=10)
    # main(window_sample, w_emg, re_w_label)
