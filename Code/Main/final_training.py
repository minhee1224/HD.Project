# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:38:18 2023

@author: minhee
"""

import pickle
from torchsummary import summary
# import librosa
# from mne.time_frequency import tfr_array_morlet

from final_function import *


hyperparameter_training = {
    # Training
    'pool': 'cls',  # 'cls' or 'mean'
    'test_size': 0.35,
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': [0.0001],
    # wandb & logging
    'prj_name': "HD_ViT",
    'log_interval': 5,
    # Model
    'patch_size': 20,  # p
    'model_dim': [256],
    'hidden_dim': [256],
    'hidden1_dim': [128],
    'hidden2_dim': [16],
    'n_output': 1,
    'n_heads': 8,
    'n_layers': [10],
    'dropout_p': 0.2,
    'model_save_dir': "./ViT_model/2310012_test1_ViT.pt",
    # Scheduler
    'n_warmup_steps': 10,
    'decay_rate': 0.5,
    # preprocessing parameters
    # "emg_height_number": 2,  # 4
    # "emg_width_number": 4,  # 8
    # "strain_height_number": 1,
    # "strain_width_number": 1,
    "sampling_freq": 2000,  # hz
    "window_size": 200,  # ms
    "overlapping_ratio": 0.75,
    "time_advance": 100,  # ms
    "label_half_size": 5,  # ms
    "lowest_freq": 20,
    "highest_freq": 300,
    "bandpass_order": 4,
    "smoothing_window": 10.0  # ms
    }

hyperparameter_training_sweep = {
    'name': 'ViT_sweep_231031',
    'method': 'random',
    'metric': {
        'name': 'Validation Loss',
        'goal': 'minimize'
        },
    'parameters': {
        'learning_rate': {
            'values': [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005]
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
            'values': [4, 8, 16, 32, 64]
            },
        'n_layers': {
            'values': [4, 6, 8, 10, 12, 14]
            },
        'n_heads': {
            'values': [4, 8, 16, 32]
            },
        'dropout_p': {
            'values': [0.1, 0.2, 0.3]
            },
        # Default
        'pool': {
            'values': ['mean']
            },
        'batch_size': {
            'values': [16]
            },
        'n_output':{
            'values': [1]
            },
        'epochs': {
            'values': [300]
            },
        'patch_size': {
            'values': [2]
            },
        'emg_height_number': {
            'values': [2]
            },
        'emg_width_number': {
            'values': [4]
            },
        'strain_height_number': {
            'values': [1]
            },
        'strain_width_number': {
            'values': [1]
            },
        'prj_name': {
            'values': ["HD_ViT"]
            },
        'log_interval': {
            'values': [5]
            },
        },
    }

def sweep_main(
        training_sweep, df_loss,
        window_sample, device, wandb_set,
        train_dataloader, test_dataloader
        ):
    model_save_dir = "./ViT_model/CYJ_data/" +\
        "231031_trial1(30_dataset)_ViT_LR%s.pt" %\
            str(training_sweep['learning_rate'])
    history_title = "231031_trial1(30_dataset)_ViT_LR%s" %\
        str(training_sweep['learning_rate'])
    # Define model
    model = ViT_Regression(
        p=training_sweep['patch_size'],
        model_dim=training_sweep['model_dim'],
        hidden_dim=training_sweep['hidden_dim'],
        hidden1_dim=training_sweep['hidden1_dim'],
        hidden2_dim=training_sweep['hidden2_dim'],
        n_output=training_sweep['n_output'],
        n_heads=training_sweep['n_heads'],
        n_layers=training_sweep['n_layers'],
        n_patches=window_sample,
        dropout_p=training_sweep['dropout_p'],
        training_phase='p',
        pool='mean',
        drop_hidden=True
        ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        training_sweep['learning_rate'])
    scheduler = ScheduleOptim(
        optimizer,
        training_sweep['n_warmup_steps'],
        training_sweep['decay_rate'])
    criterion = RMSELoss()
    #############################
    patience = 0
    best_loss = 1000
    history = {'train_loss': [], 'test_loss': [], 'lr': []}

    for epoch in range(1, training_sweep['epochs'] + 1):
        train_loss = train(
            model, train_dataloader, scheduler,
            epoch, criterion, device, training_sweep,
            wandb_set
            )
        test_loss = evaluate(
            model, test_dataloader, criterion,
            device, training_sweep, wandb_set
            )
        lr = scheduler.get_lr()
        print("\n[EPOCH: {:2d}], \tModel: ViT, \tLR: {:8.6f}, ".format(
            epoch, lr
            ) + "\tTrain Loss: {:8.6f}, \tTest Loss: {:8.6f} \n".format(
                train_loss, test_loss
                )
            )

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['lr'].append(lr)

        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience = 0
        else:
            patience += 1
            if patience >= 15:
                break

    # ViT regression model save
    if model_save_dir is not None:
        torch.save(model.state_dict(), model_save_dir)

    df_loss = pd.concat([
        df_loss,
        pd.DataFrame([{
            'learning_rate': training_sweep['learning_rate'],
            'loss': test_loss,
            'test_loss': 0}]
            )
        ], ignore_index=True)
    print(df_loss)
    # plot_history(history, history_title)

if __name__ == "__main__":

    #############################
    # Path Reading
    #############################
    train_emg_input_path = './Test_data/final/1114_version/train_emg_all.p'
    # train_strain_input_path = './Test_data/final/1108_version/train_pre_strain_evenly_distributed.p'
    train_strain_label_path = './Test_data/final/1114_version/train_delta_strain_all.p'

    test_emg_input_path = './Test_data/final/1114_version/test_emg_all.p'
    # test_strain_input_path = './Test_data/final/1108_version/test_pre_strain_evenly_distributed.p'
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
    # train_strain_input = pickle.load(open(train_strain_input_path, "rb"))
    train_strain_label = pickle.load(open(train_strain_label_path, "rb"))

    test_emg_input = pickle.load(open(test_emg_input_path, "rb"))
    # test_strain_input = pickle.load(open(test_strain_input_path, "rb"))
    test_strain_label = pickle.load(open(test_strain_label_path, "rb"))
    
    validation_emg_input = pickle.load(open(validate_emg_input_path, "rb"))
    validation_strain_label = pickle.load(open(validate_strain_label_path, "rb"))
    
    random_emg_input = pickle.load(open(random_emg_input_path, "rb"))
    random_strain_label = pickle.load(open(random_strain_label_path, "rb"))
    #############################
    wandb_set = False
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

    # gaf = GramianAngularField(image_size=20)
    # mtf = MarkovTransitionField(image_size=20, n_bins=10)
    # train_N = train_emg_input.size(0)
    # test_N = test_emg_input.size(0)

    # train_emg_after_gaf = torch.Tensor()
    # test_emg_after_gaf = torch.Tensor()
    # train_emg_after_mtf = torch.Tensor()
    # test_emg_after_mtf = torch.Tensor()

    # for i in np.arange(train_N):
    #     print('emg train')
    #     print(i)
    #     train_data_gaf = torch.tensor(
    #         gaf.fit_transform(
    #             train_emg_input[i][:][:].numpy())).unsqueeze(0)
    #     train_data_mtf = torch.tensor(
    #         mtf.fit_transform(
    #             train_emg_input[i][:][:].numpy())).unsqueeze(0)
    #     train_emg_after_gaf = torch.cat(
    #         [train_emg_after_gaf, train_data_gaf], dim=0)
    #     train_emg_after_mtf = torch.cat(
    #         [train_emg_after_mtf, train_data_mtf], dim=0)

    # for i in np.arange(test_N):
    #     print('emg test')
    #     print(i)
    #     test_data_gaf = torch.tensor(
    #         gaf.fit_transform(
    #             test_emg_input[i][:][:].numpy())).unsqueeze(0)
    #     test_data_mtf = torch.tensor(
    #         mtf.fit_transform(
    #             test_emg_input[i][:][:].numpy())).unsqueeze(0)
    #     test_emg_after_gaf = torch.cat(
    #         [test_emg_after_gaf, test_data_gaf], dim=0)
    #     test_emg_after_mtf = torch.cat(
    #         [test_emg_after_mtf, test_data_mtf], dim=0)

    # train_emg_input = train_emg_after_gaf
    # test_emg_input = test_emg_after_gaf
    ##################################################
    # Strain input: N * window_sample *
    # hyperparameter_training["strain_height_number"] *
    # hyperparameter_training["strain_width_number"]
    # N * 400 -> N * 8 * 8 * 1
    ####################################################
    # train_strain_N = train_strain_input.size(0)
    # test_strain_N = test_strain_input.size(0)
    # train_strain_after_gaf = torch.Tensor()
    # train_strain_after_mtf = torch.Tensor()
    # test_strain_after_gaf = torch.Tensor()
    # test_strain_after_mtf = torch.Tensor()

    # for i in np.arange(train_strain_N):
    #     print('strain train')
    #     print(i)
    #     train_data_gaf = torch.tensor(
    #         gaf.fit_transform(
    #             train_strain_input[i][:].numpy().reshape(1, -1)))
    #     train_data_mtf = torch.tensor(
    #         mtf.fit_transform(
    #             train_strain_input[i][:].numpy().reshape(1, -1)))
    #     train_strain_after_gaf = torch.cat(
    #         [train_strain_after_gaf, train_data_gaf], dim=0)
    #     train_strain_after_mtf = torch.cat(
    #         [train_strain_after_mtf, train_data_mtf], dim=0)

    # for i in np.arange(test_strain_N):
    #     print('strain test')
    #     print(i)
    #     test_data_gaf = torch.tensor(
    #         gaf.fit_transform(
    #             test_strain_input[i][:].numpy().reshape(1, -1)))
    #     test_data_mtf = torch.tensor(
    #         mtf.fit_transform(
    #             test_strain_input[i][:].numpy().reshape(1, -1)))
    #     test_strain_after_gaf = torch.cat(
    #         [test_strain_after_gaf, test_data_gaf], dim=0)
    #     test_strain_after_mtf = torch.cat(
    #         [test_strain_after_mtf, test_data_mtf], dim=0)
        
    # train_strain_after_gaf = train_strain_after_gaf.unsqueeze(1)
    # train_strain_after_mtf = train_strain_after_mtf.unsqueeze(1)
    # test_strain_after_gaf = test_strain_after_gaf.unsqueeze(1)
    # test_strain_after_mtf = test_strain_after_mtf.unsqueeze(1)

    # train_strain_input = train_strain_after_gaf
    # test_strain_input = test_strain_after_gaf
    
    ############################################################
    ############################################################
    # Make image (CH 3)
    ############################################################
    # Train
    # train_emg_after_gaf = train_emg_after_gaf.permute(1, 0, 2, 3)
    # train_emg_after_mtf = train_emg_after_mtf.permute(1, 0, 2, 3)
    # train_strain_after_gaf = train_strain_after_gaf.permute(1, 0, 2, 3)
    # # EMG, GASF, B * 20 * 40
    # train_input_image_ch1_row1 = torch.cat(
    #     [
    #         train_emg_after_gaf[0],
    #         train_emg_after_gaf[1]
    #         ], dim=2)
    # train_input_image_ch1_row2 = torch.cat(
    #     [
    #         train_emg_after_gaf[2],
    #         train_emg_after_gaf[3]
    #         ], dim=2)
    # train_input_image_ch1_row3 = torch.cat(
    #     [
    #         train_emg_after_gaf[4],
    #         train_emg_after_gaf[5]
    #         ], dim=2)
    # train_input_image_ch1_row4 = torch.cat(
    #     [
    #         train_emg_after_gaf[6],
    #         train_emg_after_gaf[7]
    #         ], dim=2)
    # # EMG, GASF, B * 80 * 40
    # train_input_image_ch1 = torch.cat(
    #     [
    #         train_input_image_ch1_row1,
    #         train_input_image_ch1_row2,
    #         train_input_image_ch1_row3,
    #         train_input_image_ch1_row4
    #         ], dim=1)
    # # EMG, MTF, B * 20 * 40
    # train_input_image_ch2_row1 = torch.cat(
    #     [
    #         train_emg_after_mtf[0],
    #         train_emg_after_mtf[1]
    #         ], dim=2)
    # train_input_image_ch2_row2 = torch.cat(
    #     [
    #         train_emg_after_mtf[2],
    #         train_emg_after_mtf[3]
    #         ], dim=2)
    # train_input_image_ch2_row3 = torch.cat(
    #     [
    #         train_emg_after_mtf[4],
    #         train_emg_after_mtf[5]
    #         ], dim=2)
    # train_input_image_ch2_row4 = torch.cat(
    #     [
    #         train_emg_after_mtf[6],
    #         train_emg_after_mtf[7]
    #         ], dim=2)
    # # EMG, MTF, B * 80 * 40
    # train_input_image_ch2 = torch.cat(
    #     [
    #         train_input_image_ch2_row1,
    #         train_input_image_ch2_row2,
    #         train_input_image_ch2_row3,
    #         train_input_image_ch2_row4
    #         ], dim=1)
    # # Strain, GASF, B * 80 * 40
    # train_input_image_ch3_row = torch.cat(
    #     [
    #         train_strain_after_gaf[0],
    #         train_strain_after_gaf[0]
    #         ], dim=2)
    # train_input_image_ch3 = torch.cat(
    #     [
    #         train_input_image_ch3_row,
    #         train_input_image_ch3_row,
    #         train_input_image_ch3_row,
    #         train_input_image_ch3_row
    #     ], dim=1)
    # # B * 3 * 80 * 40
    # train_input_image = torch.stack(
    #     [
    #         train_input_image_ch1,
    #         train_input_image_ch2,
    #         train_input_image_ch3
    #         ], dim=1)
    
    ############################################################
    # Test
    # test_emg_after_gaf = test_emg_after_gaf.permute(1, 0, 2, 3)
    # test_emg_after_mtf = test_emg_after_mtf.permute(1, 0, 2, 3)
    # test_strain_after_gaf = test_strain_after_gaf.permute(1, 0, 2, 3)
    # # EMG, GASF, B * 20 * 40
    # test_input_image_ch1_row1 = torch.cat(
    #     [
    #         test_emg_after_gaf[0],
    #         test_emg_after_gaf[1]
    #         ], dim=2)
    # test_input_image_ch1_row2 = torch.cat(
    #     [
    #         test_emg_after_gaf[2],
    #         test_emg_after_gaf[3]
    #         ], dim=2)
    # test_input_image_ch1_row3 = torch.cat(
    #     [
    #         test_emg_after_gaf[4],
    #         test_emg_after_gaf[5]
    #         ], dim=2)
    # test_input_image_ch1_row4 = torch.cat(
    #     [
    #         test_emg_after_gaf[6],
    #         test_emg_after_gaf[7]
    #         ], dim=2)
    # # EMG, GASF, B * 80 * 40
    # test_input_image_ch1 = torch.cat(
    #     [
    #         test_input_image_ch1_row1,
    #         test_input_image_ch1_row2,
    #         test_input_image_ch1_row3,
    #         test_input_image_ch1_row4
    #         ], dim=1)
    # # EMG, MTF, B * 20 * 40
    # test_input_image_ch2_row1 = torch.cat(
    #     [
    #         test_emg_after_mtf[0],
    #         test_emg_after_mtf[1]
    #         ], dim=2)
    # test_input_image_ch2_row2 = torch.cat(
    #     [
    #         test_emg_after_mtf[2],
    #         test_emg_after_mtf[3]
    #         ], dim=2)
    # test_input_image_ch2_row3 = torch.cat(
    #     [
    #         test_emg_after_mtf[4],
    #         test_emg_after_mtf[5]
    #         ], dim=2)
    # test_input_image_ch2_row4 = torch.cat(
    #     [
    #         test_emg_after_mtf[6],
    #         test_emg_after_mtf[7]
    #         ], dim=2)
    # # EMG, MTF, B * 80 * 40
    # test_input_image_ch2 = torch.cat(
    #     [
    #         test_input_image_ch2_row1,
    #         test_input_image_ch2_row2,
    #         test_input_image_ch2_row3,
    #         test_input_image_ch2_row4
    #         ], dim=1)
    # # Strain, GASF, B * 80 * 40
    # test_input_image_ch3_row = torch.cat(
    #     [
    #         test_strain_after_gaf[0],
    #         test_strain_after_gaf[0]
    #         ], dim=2)
    # test_input_image_ch3 = torch.cat(
    #     [
    #         test_input_image_ch3_row,
    #         test_input_image_ch3_row,
    #         test_input_image_ch3_row,
    #         test_input_image_ch3_row
    #     ], dim=1)
    # # B * 3 * 80 * 40
    # test_input_image = torch.stack(
    #     [
    #         test_input_image_ch1,
    #         test_input_image_ch2,
    #         test_input_image_ch3
    #         ], dim=1)
    # train_strain_input = torch.mul(train_strain_input, 1)
    # test_strain_input = torch.mul(test_strain_input, 1)
    ##################################################
    # Strain label: N * 1
    train_strain_label = train_strain_label.unsqueeze(1)
    test_strain_label = test_strain_label.unsqueeze(1)
    train_strain_label = torch.cat(
        (train_strain_label, test_strain_label), dim=0)
    
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
    # Dataset
    # train_dataset = CustomDataset(
    #     train_emg_input, train_strain_input, train_strain_label)
    # test_dataset = CustomDataset(
    #     test_emg_input, test_strain_input, test_strain_label)
    ##################################################
    # Dataloader
    train_dataloader = DataLoader(
        train_dataset, batch_size=32,
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
    #####################################################
    # SWEEP
    #####################################################
    # df_loss = pd.DataFrame(columns=["learning_rate", "loss", "test_loss"])
    # wandb.init(
    #     project="HD_ViT", entity='minheelee', config=hyperparameter_training)
    # sweep_id = wandb.sweep(
    #     hyperparameter_training_sweep, project="HD_ViT", entity="minheelee")
    # wandb.agent(
    #     sweep_id,
    #     sweep_main(
    #         hyperparameter_training,
    #         df_loss,
    #         window_sample,
    #         device,
    #         wandb_set,
    #         train_dataloader,
    #         test_dataloader), count=20)
    #####################################################
    # Main
    # ViT training
    df_loss = pd.DataFrame(columns=[
        "learning_rate", "train_loss", "test_loss",
        "validation_loss", "random_loss"])
    # [0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    model_num = 1
    model_num_list = np.arange(model_num)
    for LR in hyperparameter_training['learning_rate']:
        for model_number in model_num_list:
            model_save_dir = "./ViT_model/CYJ_data/" +\
                "231115_delta_strain_all_dataset_ViT_LR%s_model%s.pt" % (
                    str(LR),
                    str(hyperparameter_training['model_dim'][model_number]))
            history_title = "231115_delta_strain_all_dataset_ViT_LR%s_model%s"\
                % (str(LR),
                   str(hyperparameter_training['model_dim'][model_number]))
            # Define model
            model = ViT_Regression(
                p=hyperparameter_training['patch_size'],
                model_dim=hyperparameter_training['model_dim'][model_number],
                hidden_dim=hyperparameter_training['hidden_dim'][model_number],
                hidden1_dim=hyperparameter_training['hidden1_dim'][model_number],
                hidden2_dim=hyperparameter_training['hidden2_dim'][model_number],
                n_output=hyperparameter_training['n_output'],
                n_heads=hyperparameter_training['n_heads'],
                n_layers=hyperparameter_training['n_layers'][model_number],
                n_patches=8,
                dropout_p=hyperparameter_training['dropout_p'],
                training_phase='p',
                pool=hyperparameter_training['pool'],
                drop_hidden=True
                ).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                LR)
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(
            #     optimizer=optimizer,
            #     milestones=[20, 50, 100, 150, 200, 250, 280],
            #     gamma=hyperparameter_training['decay_rate'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min',
                factor=hyperparameter_training['decay_rate'], patience=3,
                verbose=True)
            # scheduler = ScheduleOptim(
            #     optimizer,
            #     hyperparameter_training['n_warmup_steps'],
            #     hyperparameter_training['decay_rate'])
            criterion = RMSELoss()
            #############################
            main_patience = 0
            best_loss = 1000
            history = {'train_loss': [], 'test_loss': [],
                       'validation_loss': [], 'random_loss': [], 'lr': []}
    
            for epoch in range(1, hyperparameter_training['epochs'] + 1):
                train_loss = train(
                    model, train_dataloader, optimizer, scheduler,
                    epoch, criterion, device, hyperparameter_training,
                    wandb_set=False
                    )
                # test_loss = evaluate(
                #     model, test_dataloader, criterion,
                #     device, hyperparameter_training, wandb_set
                #     )
                # validation_loss = evaluate(
                #     model, validation_dataloader, criterion,
                #     device, hyperparameter_training, wandb_set
                #     )
                # random_loss = evaluate(
                #     model, random_dataloader, criterion,
                #     device, hyperparameter_training, wandb_set
                #     )
                test_loss, validation_loss, random_loss = sweep_evaluate(
                    model, validation_dataloader, random_dataloader, criterion,
                    device, hyperparameter_training, wandb_set=False
                    )
    
                # lr = scheduler.get_lr()
                # lr = scheduler.get_last_lr()[0]
                lr = optimizer.param_groups[0]['lr']
                print("\r[EPOCH: {:2d}], \tModel: ViT, \tLR: {:8.6f}, ".format(
                    epoch, lr
                    ) + "\tTrain Loss: {:8.6f}, \tTest Loss: {:8.6f}".format(
                        train_loss, test_loss) +
                        "\tValidation Loss: {:8.6f}, \tRandom Loss: {:8.6f}".format(
                            validation_loss, random_loss
                            )
                    )
    
                history['train_loss'].append(train_loss)
                history['test_loss'].append(test_loss)
                history['validation_loss'].append(validation_loss)
                history['random_loss'].append(random_loss)
                history['lr'].append(lr)
    
                # # Early stopping
                # if test_loss < best_loss:
                #     best_loss = test_loss
                #     main_patience = 0
                # else:
                #     main_patience += 1
                #     if main_patience >= 5:
                #         break
    
            # ViT regression model save
            if model_save_dir is not None:
                torch.save(model.state_dict(), model_save_dir)
    
            df_loss = pd.concat([
                df_loss,
                pd.DataFrame([{
                    'learning_rate': LR,
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'validation_loss': validation_loss,
                    'random_loss': random_loss}]
                    )
                ], ignore_index=True)
            print(df_loss)
            print(history)
            plot_history(history, history_title)
            summary(model, (8, 20, 20))
