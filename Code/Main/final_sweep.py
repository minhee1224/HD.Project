# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 22:22:03 2023

@author: minhee
"""

import pickle
import torch
import torch.nn as nn
import time
import wandb

from final_function import *
from final_training import *

wandb.login(key="sample")  # Use private key input
# Sweep configuration
sweep_config = {
    'name' : 'HD_ViT_sweep_231114',
    'method' : 'random',
    'metric' : {
        'goal' : 'minimize',
        'name' : 'Test Loss'
        },
    'parameters' : {
        # 'learning_rate' : {
        #     'max' : 0.01,
        #     'min': 0.00001
        #     },
        # 'model_dim' : {
        #     'distribution' : 'q_uniform',
        #     'min' : 4,
        #     'max' : 256,
        #     'q' : 1
        #     },
        # 'hidden_dim' : {
        #     'distribution' : 'q_uniform',
        #     'min' : 32,
        #     'max' : 256,
        #     'q' : 1
            
        #     },
        # 'hidden1_dim' : {
        #     'distribution' : 'q_uniform',
        #     'min' : 32,
        #     'max' : 256,
        #     'q' : 1
        #     },
        # 'hidden2_dim' : {
        #     'distribution' : 'q_uniform',
        #     'min' : 4,
        #     'max' : 64,
        #     'q' : 1
        #     },
        'learning_rate' : {
            'values': [0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001]
            },
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
            'values': [4, 8, 16, 32, 64]
            },
        'n_layers': {
            'values': [8, 10, 12, 14]
            },
        'n_heads': {
            'values': [4, 8, 16, 32]
            },
        'dropout_p': {
            'values': [0.1, 0.2, 0.3]
            },
        'patch_size': {'values':[20]},
        'n_output': {'values':[1]},
        'epochs': {'values':[300]}
        }
    }


def run_sweep(config=None):
    #############################
    # Path Reading
    #############################
    train_emg_input_path = './Test_data/final/1114_version/train_emg_all.p'
    train_strain_label_path = './Test_data/final/1114_version/train_delta_strain_all.p'

    test_emg_input_path = './Test_data/final/1114_version/test_emg_all.p'
    test_strain_label_path = './Test_data/final/1114_version/test_delta_strain_all.p'

    validate_emg_input_path = './Test_data/final/1114_version/validation_emg_all.p'
    validate_strain_label_path = './Test_data/final/1114_version/validation_delta_strain_all.p'

    random_emg_input_path = './Test_data/final/1114_version/random_emg_all.p'
    random_strain_label_path = './Test_data/final/1114_version/random_delta_strain_all.p'
    #############################
    # Data Reading
    #############################
    # EMG: N * 400 * 8
    # Strain input: N * 400 * 1
    # Strain label: N
    train_emg_input = pickle.load(open(train_emg_input_path, "rb"))
    train_strain_label = pickle.load(open(train_strain_label_path, "rb"))

    test_emg_input = pickle.load(open(test_emg_input_path, "rb"))
    test_strain_label = pickle.load(open(test_strain_label_path, "rb"))

    validation_emg_input = pickle.load(open(validate_emg_input_path, "rb"))
    validation_strain_label = pickle.load(open(validate_strain_label_path, "rb"))

    random_emg_input = pickle.load(open(random_emg_input_path, "rb"))
    random_strain_label = pickle.load(open(random_strain_label_path, "rb"))
    #############################
    wandb_set = True
    window_sample = 400  # 20 * 20
    # Final dimension fitting
    ##################################################
    # EMG input: N * window_sample *
    # hyperparameter_training["emg_height_number"] *
    # hyperparameter_training["emg_width_number"]
    # N * 8 * 20 * 20
    train_emg_input = train_emg_input.permute(0, 2, 1).reshape(-1, 8, 20, 20)
    test_emg_input = test_emg_input.permute(0, 2, 1).reshape(-1, 8, 20, 20)
    train_emg_input = torch.cat((train_emg_input, test_emg_input), dim=0)

    validation_emg_input =\
        validation_emg_input.permute(0, 2, 1).reshape(-1, 8, 20, 20)
    random_emg_input =\
        random_emg_input.permute(0, 2, 1).reshape(-1, 8, 20, 20)

    ##################################################
    # Strain label: N * 1
    train_strain_label = train_strain_label.unsqueeze(1)
    test_strain_label = test_strain_label.unsqueeze(1)
    train_strain_label = torch.cat((train_strain_label, test_strain_label), dim=0)

    validation_strain_label = validation_strain_label.unsqueeze(1)
    random_strain_label = random_strain_label.unsqueeze(1)
    ##################################################
    # Dataset_CASE1
    # Batch * 9 * 20 * 20
    # train_input = torch.cat([train_emg_input, train_strain_input], dim=1)
    # test_input = torch.cat([test_emg_input, test_strain_input], dim=1)
    train_label = train_strain_label
    # test_label = test_strain_label
    validation_label = validation_strain_label
    random_label = random_strain_label

    # train_dataset = CustomDataset_case1(train_input, train_label)
    # test_dataset = CustomDataset_case1(test_input, test_label)
    ##################################################
    # Dataset_CASE2
    # Batch * 3 * 80 * 40
    train_dataset = CustomDataset_case1(train_emg_input, train_label)
    # test_dataset = CustomDataset_case1(test_emg_input, test_label)
    validation_dataset = CustomDataset_case1(
        validation_emg_input, validation_label)
    random_dataset = CustomDataset_case1(random_emg_input, random_label)
    ##################################################
    # Dataloader
    train_dataloader = DataLoader(
        train_dataset, batch_size=16,
        drop_last=True,
        collate_fn=train_dataset.collate_fn
        )
    # test_dataloader = DataLoader(
    #     test_dataset, batch_size=1,
    #     drop_last=True,
    #     collate_fn=test_dataset.collate_fn
    #     )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=1,
        drop_last=True,
        collate_fn=validation_dataset.collate_fn
        )
    random_dataloader = DataLoader(
        random_dataset, batch_size=1,
        drop_last=True,
        collate_fn=random_dataset.collate_fn
        )
    #####################################################
    # Device setting
    device = torch.device('cuda') \
        if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(
        torch.__version__, device)
        )
    
    
    wandb.init(config=config,project="ViT",entity='kjhy1336')
    w_config = wandb.config
    args = {
        'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
        'learning_rate' : w_config.learning_rate,
        'model_dim' : w_config.model_dim,
        'hidden_dim' : w_config.hidden_dim,
        'hidden1_dim' : w_config.hidden1_dim,
        'hidden2_dim' : w_config.hidden2_dim,
        'n_layers' : w_config.n_layers,
        'n_heads' : w_config.n_heads,
        'dropout_p' : w_config.dropout_p,
        'patch_size': w_config.patch_size,
        'n_output': w_config.n_output,
        'epochs': w_config.epochs
    }
    # model_save_dir = "./ViT_model/CYJ_data/" +\
    #     "231114_(delta_strain_all_dataset)_ViT_LR%f" +\
    #         "_model_dim%d_hidden_dim%d_hidden1_dim%d_" +\
    #             "hidden2_dim%d_layer%d_head%d.pt" % (
    #                 float(args['learning_rate']),
    #                 int(args['model_dim']),
    #                 int(args['hidden_dim']),
    #                 int(args['hidden1_dim']),
    #                 int(args['hidden2_dim']),
    #                 int(args['n_layers']),
    #                 int(args['n_heads']))
    model_save_dir = "./ViT_model/CYJ_data/" +\
        "231114_(delta_strain_all_dataset)_ViT_LR%s.pt" % str(args['learning_rate'])
    # Define model
    model = ViT_Regression(
        p=args['patch_size'],
        model_dim=args['model_dim'],
        hidden_dim=args['hidden_dim'],
        hidden1_dim=args['hidden1_dim'],
        hidden2_dim=args['hidden2_dim'],
        n_output=args['n_output'],
        n_heads=args['n_heads'],
        n_layers=args['n_layers'],
        n_patches=8,
        dropout_p=args['dropout_p'],
        pool='cls').to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        args['learning_rate'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min',
        factor=0.99, patience=3)
    criterion = RMSELoss()
    #############################
    main_patience = 0
    best_loss = 1000

    for epoch in range(1, args['epochs'] + 1):
        train_loss = train(
            model, train_dataloader, optimizer, scheduler,
            epoch, criterion, device, args,
            wandb_set=True
            )
        test_loss, _, _ = sweep_evaluate(
            model, validation_dataloader, random_dataloader, criterion,
            device, args, wandb_set=True
            )

        lr = optimizer.param_groups[0]['lr']
        print("\n[EPOCH: {:2d}], \tModel: ViT, \tLR: {:8.6f}, ".format(
            epoch, lr
            ) + "\tTrain Loss: {:8.6f}, \tTest Loss: {:8.6f} \n".format(
                train_loss, test_loss)
            )

        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            main_patience = 0
        else:
            main_patience += 1
            if main_patience >= 5:
                break

    # ViT regression model save
    if model_save_dir is not None:
        torch.save(model.state_dict(), model_save_dir)

    summary(model, (8, 20, 20))


sweep_id = wandb.sweep(sweep_config,project="ViT",entity='kjhy1336')
wandb.agent(
    sweep_id,
    run_sweep,
    count=7)