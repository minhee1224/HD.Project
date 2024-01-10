# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:18:42 2023

@author: minhee
"""

import os
import argparse
import ast
import torch
import wandb


DELSYS_hyperparameter_defaults = {
    # Training
    'RANDOM_STATE': 42,
    'TEST_RATIO': 0.25,
    'BATCH_SIZE': 16,
    'EPOCHS': 300,
    'LR': 0.001,
    # wandb & logging
    'prj_name': "HD_ViT",
    'log_interval': 5,
    # Model
    'model_dim': 32,
    'hidden_dim': 64,
    'hidden1_dim': 16,
    'hidden2_dim': 4,
    'n_output': 1,
    'n_head': 8,
    'n_layer': 10,
    'dropout_p': 0.2,
    'model_dir': "./ViT_model/ViT_test.pt",
    # Scheduler
    'n_warmup_steps': 10,
    'decay_rate': 0.99
    }

DELSYS_case2_hyperparameter_defaults = {
    # Training
    'RANDOM_STATE': 42,
    'TEST_RATIO': 0.25,
    'BATCH_SIZE': 16,
    'EPOCHS': 300,
    'LR': 0.001,
    # wandb & logging
    'prj_name': "HD_ViT",
    'log_interval': 5,
    # Model
    'model_dim': 32,
    'hidden_dim': 64,
    'hidden1_dim': 16,
    'hidden2_dim': 4,
    'n_output': 1,
    'n_head': 8,
    'n_layer': 10,
    'dropout_p': 0.2,
    'model_dir': "./ViT_model/ViT_test.pt",
    # Scheduler
    'n_warmup_steps': 10,
    'decay_rate': 0.99
    }