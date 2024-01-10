# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 12:09:45 2023

@author: minhee
"""

from final_function import *
from final_training import *

#################################################################
# Model load
#################################################################
model_path = './ViT_model/CYJ_data/' +\
    '231115_delta_strain_all_dataset_ViT_LR0.0001_model256.pt'
device = torch.device('cuda')
model = ViT_Regression(
    p=hyperparameter_training['patch_size'],
    model_dim=hyperparameter_training['model_dim'][0],
    hidden_dim=hyperparameter_training['hidden_dim'][0],
    hidden1_dim=hyperparameter_training['hidden1_dim'][0],
    hidden2_dim=hyperparameter_training['hidden2_dim'][0],
    n_output=hyperparameter_training['n_output'],
    n_heads=hyperparameter_training['n_heads'],
    n_layers=hyperparameter_training['n_layers'][0],
    n_patches=8,
    dropout_p=hyperparameter_training['dropout_p'],
    pool=hyperparameter_training['pool']
    ).to(device)
model.load_state_dict(
    torch.load(model_path, map_location=device))
model.eval()

#################################################################
# Data load
#################################################################
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
validation_emg_input = pickle.load(open(validate_emg_input_path, "rb"))
validation_strain_label = pickle.load(open(validate_strain_label_path, "rb"))

random_emg_input = pickle.load(open(random_emg_input_path, "rb"))
random_strain_label = pickle.load(open(random_strain_label_path, "rb"))
#############################
wandb_set = False
window_sample = 400  # 20 * 20
##################################################
# N * 8 * 20 * 20
validation_emg_input =\
    validation_emg_input.permute(0, 2, 1).reshape(-1, 8, 20, 20)
random_emg_input =\
    random_emg_input.permute(0, 2, 1).reshape(-1, 8, 20, 20)

##################################################
# Strain label: N * 1
validation_strain_label = validation_strain_label.unsqueeze(1)
random_strain_label = random_strain_label.unsqueeze(1)

validation_label = validation_strain_label
random_label = random_strain_label

##################################################
# Dataset_CASE2
# Batch * 3 * 80 * 40
validation_dataset = CustomDataset_case1(
    validation_emg_input, validation_label)
random_dataset = CustomDataset_case1(random_emg_input, random_label)
##################################################
# Dataloader
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
#####################################################
# Main
# ViT Inference
criterion = RMSELoss()

test_loss, validation_loss, random_loss, history, inference_time, total_num =\
    print_evaluate(
        model, validation_dataloader, random_dataloader, criterion,
        device, hyperparameter_training, wandb_set=False
        )

# print
print("inference time:")  # per sample
print(inference_time / total_num)

# random dataset
random_range = np.max(np.abs(history["random_true"])) - \
    np.min(np.abs(history["random_true"]))
random_nRMSE_mean = (np.mean(history["random_RMSE"])/random_range)*100
random_nRMSE_max = (np.max(history["random_RMSE"])/random_range)*100

random_allowable = 0.25*random_range
random_RMSE = pd.Series(history["random_RMSE"])
random_CI = len(random_RMSE[random_RMSE<=random_allowable])/len(random_RMSE)

print(random_nRMSE_mean)
print(random_nRMSE_max)
print(random_CI)

# validation dataset
validation_range = np.max(np.abs(history["validation_true"])) - \
    np.min(np.abs(history["validation_true"]))
validation_nRMSE_mean = (np.mean(history["validation_RMSE"])/validation_range)*100
validation_nRMSE_max = (np.max(history["validation_RMSE"])/validation_range)*100

validation_allowable = 0.25*validation_range
validation_RMSE = pd.Series(history["validation_RMSE"])
validation_CI = len(validation_RMSE[validation_RMSE<=validation_allowable])/len(validation_RMSE)

print(validation_nRMSE_mean)
print(validation_nRMSE_max)
print(validation_CI)

# Graph
fontdict = {
    'fontname':'Arial',
    'fontsize':20,
    'style':'normal'
    }
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20
plt.rcParams['font.style'] = 'normal'

# validation graph
plt.scatter(history["validation_true"],history["validation_prediction"])
plt.xlim((-0.035,0.035))
plt.ylim((-0.035,0.035))
plt.xlabel("True strain [mV]",**fontdict)
plt.ylabel("Predicted strain [mV]",**fontdict)
plt.show()

# random graph
plt.scatter(history["random_true"],history["random_prediction"])
plt.xlim((-0.035,0.035))
plt.ylim((-0.035,0.035))
plt.xlabel("True strain [mV]",**fontdict)
plt.ylabel("Predicted strain [mV]",**fontdict)
plt.show()
