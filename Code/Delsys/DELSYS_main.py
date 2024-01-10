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

from DELSYS_preprocessing import *
from DELSYS_model import *
from DELSYS_model_case2 import *
from DELSYS_config import *


def DELSYS_main(wandb_set, window_sample, w_emg, w_label):

    #########################################
    #########################################
    # TRAINING
    #######################################
    df_loss = pd.DataFrame(columns=["LR", "loss", "test_loss"])
    if wandb_set == True:
        wandb.init(project="HD_ViT", entity='minheelee',
                    config=DELSYS_hyperparameter_defaults)
        w_config=wandb.config
    else:
        w_config=DELSYS_hyperparameter_defaults

    # Device setting
    DEVICE = torch.device('cuda') \
        if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(
        torch.__version__, DEVICE))

    if wandb_set == True:
        # Model parameter setting
        model_dir = "./ViT_model/ViT_DELSYS_angle_LR%s_HD%s.pt" %\
            (str(w_config.LR), str(w_config.hidden_dim))
        N_PATCH = window_sample # FIXED
    
        test_loss, history, final_test_loss =\
            ViT_DELSYStraining(w_config, wandb_set, model_dir, emg=w_emg, label=w_label,
                              patch_size=int(1), MODEL_DIM=w_config.model_dim,
                              HIDDEN_DIM=w_config.hidden_dim,
                              HIDDEN1_DIM=w_config.hidden1_dim,
                              HIDDEN2_DIM=w_config.hidden2_dim,
                              N_OUTPUT=w_config.n_output,
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
    
        df_loss = pd.concat([df_loss,
                             pd.DataFrame([{'LR': w_config.LR,
                                            'loss': test_loss,
                                            'test_loss': final_test_loss}])
                             ], ignore_index=True)
        print(df_loss)
        plot_history(history, "ViT_DELSYS_angle_LR_%s_" % str(w_config.LR))
        wandb.run.finish()
    else:
        # Model parameter setting
        model_dir = "./ViT_model/ViT_DELSYS_angle_LR%s_HD%s.pt" %\
            (str(w_config["LR"]), str(w_config["hidden_dim"]))
        N_PATCH = window_sample # FIXED
    
        test_loss, history, final_test_loss =\
            ViT_DELSYStraining(w_config, wandb_set, model_dir, emg=w_emg, label=w_label,
                              patch_size=int(1), MODEL_DIM=w_config["model_dim"],
                              HIDDEN_DIM=w_config["hidden_dim"],
                              HIDDEN1_DIM=w_config["hidden1_dim"],
                              HIDDEN2_DIM=w_config["hidden2_dim"],
                              N_OUTPUT=w_config["n_output"],
                              N_HEAD=w_config["n_head"],
                              N_LAYER=w_config["n_layer"],
                              N_PATCH=N_PATCH,
                              DROPOUT_P=w_config["dropout_p"],
                              TEST_RATIO=w_config["TEST_RATIO"],
                              RANDOM_STATE=w_config["RANDOM_STATE"],
                              BATCH_SIZE=w_config["BATCH_SIZE"],
                              EPOCHS=w_config["EPOCHS"],
                              DEVICE=DEVICE, LR=w_config["LR"], 
                              N_WARMUP_STEPS=w_config["n_warmup_steps"],
                              DECAY_RATE=w_config["decay_rate"])
    
        df_loss = pd.concat([df_loss,
                             pd.DataFrame([{'LR': w_config["LR"],
                                            'loss': test_loss,
                                            'test_loss': final_test_loss}])
                             ], ignore_index=True)
        print(df_loss)
        plot_history(history, "ViT_DELSYS_angle_LR_%s_" % str(w_config["LR"]))


def DELSYS_strain_case2_main(wandb_set, window_sample, strain_window_sample,
                             w_emg, w_strain, w_label):

    #########################################
    #########################################
    # TRAINING
    #######################################
    df_loss = pd.DataFrame(columns=["LR", "loss", "test_loss"])
    if wandb_set == True:
        wandb.init(project="HD_ViT", entity='minheelee',
                    config=DELSYS_case2_hyperparameter_defaults)
        w_config=wandb.config
    else:
        w_config=DELSYS_case2_hyperparameter_defaults

    # Device setting
    DEVICE = torch.device('cuda') \
        if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(
        torch.__version__, DEVICE))

    if wandb_set == True:
        # Model parameter setting
        model_dir = "./ViT_model/s1_DELSYS_case2/ViT_angle_LR%s_HD%s.pt" %\
            (str(w_config.LR), str(w_config.hidden_dim))
        N_PATCH = window_sample # FIXED
        N_STRAIN_PATCH = strain_window_sample # FIXED

        test_loss, history, final_test_loss =\
            ViT_DELSYS_case2_training(w_config, wandb_set, model_dir, emg=w_emg,
                                      strain=w_strain, label=w_label,
                                      patch_size=int(1),
                                      MODEL_DIM=w_config.model_dim,
                                      HIDDEN_DIM=w_config.hidden_dim,
                                      HIDDEN1_DIM=w_config.hidden1_dim,
                                      HIDDEN2_DIM=w_config.hidden2_dim,
                                      N_OUTPUT=w_config.n_output,
                                      N_HEAD=w_config.n_head,
                                      N_LAYER=w_config.n_layer,
                                      N_PATCH=N_PATCH,
                                      N_STRAIN_PATCH=N_STRAIN_PATCH,
                                      DROPOUT_P=w_config.dropout_p,
                                      TEST_RATIO=w_config.TEST_RATIO,
                                      RANDOM_STATE=w_config.RANDOM_STATE,
                                      BATCH_SIZE=w_config.BATCH_SIZE,
                                      EPOCHS=w_config.EPOCHS,
                                      DEVICE=DEVICE, LR=w_config.LR, 
                                      N_WARMUP_STEPS=w_config.n_warmup_steps,
                                      DECAY_RATE=w_config.decay_rate)

        df_loss = pd.concat([df_loss,
                              pd.DataFrame([{'LR': w_config.LR,
                                            'loss': test_loss,
                                            'test_loss': final_test_loss}])
                              ], ignore_index=True)
        print(df_loss)
        # plot_history(history, "ViT_DELSYS_case2_angle_LR_%s_" %\
        #              str(w_config.LR))
        wandb.run.finish()
    else:
        # Model parameter setting
        model_dir = "./ViT_model/s1_DELSYS_case2/ViT_angle_LR%s_HD%s.pt" %\
            (str(w_config["LR"]), str(w_config["hidden_dim"]))
        N_PATCH = window_sample # FIXED
        N_STRAIN_PATCH = strain_window_sample # FIXED
    
        test_loss, history, final_test_loss =\
            ViT_DELSYS_case2_training(w_config, wandb_set, model_dir, emg=w_emg,
                                      strain=w_strain, label=w_label,
                                      patch_size=int(1),
                                      MODEL_DIM=w_config["model_dim"],
                                      HIDDEN_DIM=w_config["hidden_dim"],
                                      HIDDEN1_DIM=w_config["hidden1_dim"],
                                      HIDDEN2_DIM=w_config["hidden2_dim"],
                                      N_OUTPUT=w_config["n_output"],
                                      N_HEAD=w_config["n_head"],
                                      N_LAYER=w_config["n_layer"],
                                      N_PATCH=N_PATCH,
                                      N_STRAIN_PATCH=N_STRAIN_PATCH,
                                      DROPOUT_P=w_config["dropout_p"],
                                      TEST_RATIO=w_config["TEST_RATIO"],
                                      RANDOM_STATE=w_config["RANDOM_STATE"],
                                      BATCH_SIZE=w_config["BATCH_SIZE"],
                                      EPOCHS=w_config["EPOCHS"],
                                      DEVICE=DEVICE, LR=w_config["LR"], 
                                      N_WARMUP_STEPS=w_config["n_warmup_steps"],
                                      DECAY_RATE=w_config["decay_rate"])
    
        df_loss = pd.concat([df_loss,
                              pd.DataFrame([{'LR': w_config["LR"],
                                            'loss': test_loss,
                                            'test_loss': final_test_loss}])
                              ], ignore_index=True)
        print(df_loss)
        # plot_history(history, "ViT_DELSYS_case2_angle_LR_%s_" %\
        #              str(w_config["LR"]))


if __name__ == "__main__":

    ########################################################################
    # FOR SWEEP
    ########################################################################
    # ANGLE REGRESSION
    ########################################################################
    # EMG only
    # window_sample, emg, angle = DELSYS_emg_preprocessing()
    ########################################################################
    # CASE 2
    window_sample, strain_window_sample, emg, strain, angle =\
        DELSYS_emg_strain_preprocessing()
    # WANDB setup
    wandb_set = True
    # wandb_set = False
    ########################################################################
    # EMG only
    # DELSYS_main(wandb_set, window_sample, emg, angle)
    ########################################################################
    # CASE 2
    DELSYS_strain_case2_main(wandb_set, window_sample, strain_window_sample,
                             emg, strain, angle)
