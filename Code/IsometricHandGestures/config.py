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


hyperparameter_defaults = {
    # Training
    'RANDOM_STATE': 42,
    'TEST_RATIO': 0.25,
    'BATCH_SIZE': 16,
    'EPOCHS': 300,
    'LR': 0.0001,
    # wandb & logging
    'prj_name': "HD_ViT",
    'log_interval': 5,
    # Model
    'model_dim': 32,
    'hidden_dim': 128,
    'n_class': 66,
    'n_head': 8,
    'n_layer': 12,
    'n_patch': 409,
    'dropout_p': 0.1,
    'model_dir': "./ViT_model/ViT_test.pt",
    # Scheduler
    'n_warmup_steps': 10,
    'decay_rate': 0.98
    }


force_hyperparameter_defaults = {
    # Training
    'RANDOM_STATE': 42,
    'TEST_RATIO': 0.25,
    'BATCH_SIZE': 16,
    'EPOCHS': 300,
    'LR': 0.0005,
    # wandb & logging
    'prj_name': "HD_ViT",
    'log_interval': 5,
    # Model
    'model_dim': 32,
    'hidden_dim': 128,
    'hidden1_dim': 256,
    'hidden2_dim': 256,
    'n_output': 9,
    'n_head': 32,
    'n_layer': 10,
    'n_patch': 409,
    'dropout_p': 0.2,
    'model_dir': "./ViT_model/ViT_test.pt",
    # Scheduler
    'n_warmup_steps': 10,
    'decay_rate': 0.99
    }


sweep_config = {
    'name': 'ViT_subject2_sweep_230529',
    'method': 'random',
    'metric': {
        'name': 'Validation Accuracy',
        'goal': 'maximize'
        },
    'parameters': {
        'LR': {
            'values': [0.00001, 0.00005, 0.0001, 0.0005]
            },
        'n_warmup_steps': {
            'values': [5, 10, 15]
            },
        'decay_rate': {
            'values': [0.98, 0.985, 0.99, 0.995]
            },
        # Model
        'model_dim': {
            'values': [32, 64, 128]
            },
        'hidden_dim': {
            'values': [32, 64, 128, 256]
            },
        'n_layer': {
            'values': [10, 12, 14]
            },
        'n_head': {
            'values': [8, 16, 32]
            },
        'dropout_p': {
            'values': [0.1, 0.2, 0.3]
            },
        # Default
        'n_patch': {
            'values': [409]
            },
        'n_class':{
            'values': [66]
            },
        'EPOCHS': {
            'values': [300]
            },
        'BATCH_SIZE': {
            'values': [16]
            },
        'TEST_RATIO': {
            'values': [0.25]
            },
        'RANDOM_STATE': {
            'values': [42]
            },
        'prj_name': {
            'values': ["HD_ViT"]
            },
        'log_interval': {
            'values': [5]
            },
        },
    }


force_sweep_config = {
    'name': 'ViT_force_sweep_230710',
    'method': 'random',
    'metric': {
        'name': 'Validation Loss',
        'goal': 'minimize'
        },
    'parameters': {
        'LR': {
            'values': [0.00001, 0.00005, 0.0001, 0.0005]
            },
        'n_warmup_steps': {
            'values': [5, 10, 15]
            },
        'decay_rate': {
            'values': [0.98, 0.985, 0.99, 0.995]
            },
        # Model
        'model_dim': {
            'values': [32, 64, 128]
            },
        'hidden_dim': {
            'values': [32, 64, 128, 256]
            },
        'hidden1_dim': {
            'values': [32, 64, 128, 256]
            },
        'hidden2_dim': {
            'values': [32, 64, 128, 256]
            },
        'n_layer': {
            'values': [10, 12, 14]
            },
        'n_head': {
            'values': [8, 16, 32]
            },
        'dropout_p': {
            'values': [0.1, 0.2, 0.3]
            },
        # Default
        'n_patch': {
            'values': [409]
            },
        'n_output':{
            'values': [9]
            },
        'EPOCHS': {
            'values': [300]
            },
        'BATCH_SIZE': {
            'values': [16]
            },
        'TEST_RATIO': {
            'values': [0.25]
            },
        'RANDOM_STATE': {
            'values': [42]
            },
        'prj_name': {
            'values': ["HD_ViT"]
            },
        'log_interval': {
            'values': [5]
            },
        },
    }


# def arg_parse():
#     parser = argparse.ArgumentParser()

#     # Model
#     parser.add_argument("--model_dim", type=int, default=32)
#     parser.add_argument("--hidden_dim", type=int, default=128)
#     parser.add_argument("--n_class", type=int, default=66)
#     parser.add_argument("--n_head", type=int, default=8)
#     parser.add_argument("--n_layer", type=int, default=12)
#     parser.add_argument("--n_patch", type=int, default=409)
#     parser.add_argument("--dropout_p", type=float, default=0.1)
#     parser.add_argument("--model_dir", type=str,
#                         default="./ViT_model/ViT_test.pt")

#     # Scheduler
#     parser.add_argument("--n_warmup_steps", type=int, default=10)
#     parser.add_argument("--decay_rate", type=float, default=0.98)

#     # Training
#     parser.add_argument("--RANDOM_STATE", type=int, default=42)
#     parser.add_argument("--TEST_RATIO", type=float, default=0.25)
#     parser.add_argument("--BATCH_SIZE", type=int, default=16)
#     parser.add_argument("--EPOCHS", type=int, default=300)
#     parser.add_argument("--LR", type=float, default=0.0001)

#     # wandb & logging
#     parser.add_argument("--prj_name", type=str, default="HD_ViT")
#     parser.add_argument("--log_interval", type=int, default=5)
#     parser.add_argument("--sample_save_dir", type=str, default="./results/")
#     parser.add_argument("--checkpoint_dir", type=str, default="./weights/")
#     parser.add_argument("--resume_from", action="store_true")

#     opt = parser.parse_args()

#     return opt


# checkpoint
def save_checkpoint(checkpoint, checkpoint_dir, file_name):
    os.makedirs(checkpoint_dir, exist_ok=True)

    output_path = os.path.join(checkpoint_dir, file_name)
    torch.save(checkpoint, output_path)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, rank=-1):
    # load model if resume_from is set
    if rank != -1: # distributed
        map_location = {"cuda:%d" % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    else:
        checkpoint = torch.load(checkpoint_path)    

    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch']

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return model, optimizer, scheduler, start_epoch
