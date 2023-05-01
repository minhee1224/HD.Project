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


def arg_parse():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--model_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_class", type=int, default=66)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_patch", type=int, default=409)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--model_dir", type=str,
                        default="./ViT_model/ViT_test.pt")

    # Scheduler
    parser.add_argument("--n_warmup_steps", type=int, default=10)
    parser.add_argument("--decay_rate", type=float, default=0.98)

    # Training
    parser.add_argument("--RANDOM_STATE", type=int, default=42)
    parser.add_argument("--TEST_RATIO", type=float, default=0.25)
    parser.add_argument("--BATCH_SIZE", type=int, default=16)
    parser.add_argument("--EPOCHS", type=int, default=300)
    parser.add_argument("--LR", type=float, default=0.0001)

    # wandb & logging
    parser.add_argument("--prj_name", type=str, default="HD_ViT")
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--sample_save_dir", type=str, default="./results/")
    parser.add_argument("--checkpoint_dir", type=str, default="./weights/")
    parser.add_argument("--resume_from", action="store_true")

    opt = parser.parse_args()

    return opt


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
