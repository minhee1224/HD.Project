# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 19:39:26 2023

@author: minhee
"""

from final_function import *
from final_training import *


# class DataPredictor:
#     def __init__(self, test_dataloader, model_path, opt, device):
#         self.model_path = model_path
#         self.test_dataloader = test_dataloader
#         self.opt = opt
#         self.device = device
#         self.model_load()
#         self.data_loader()

#     def model_load(self):
#         model = ViT_Regression(
#             p=self.opt["patch_size"],
#             model_dim=self.opt["model_dim"],
#             hidden_dim=self.opt["hidden_dim"],
#             hidden1_dim=self.opt["hidden1_dim"],
#             hidden2_dim=self.opt["hidden2_dim"],
#             n_output=self.opt["n_output"],
#             n_heads=self.opt["n_heads"],
#             n_layers=self.opt["n_layers"],
#             n_patches=int(np.floor(float(
#                 self.opt["window_size"] /
#                 float(1000.0 / self.opt["sampling_freq"])
#                 ))),
#             dropout_p=self.opt["dropout_p"],
#             training_phase='p',
#             pool='mean',
#             drop_hidden=True
#             )
#         model.load_state_dict(
#             torch.load(self.model_path, map_location=self.device)
#             )
#         model.to(self.device)
#         self.model = model
#         self.criterion = RMSELoss()

#     def data_loader(self):
#         test_dataloader = DataLoader(
#             self.test_dataset,
#             batch_size=1,
#             drop_last=True,
#             collate_fn=CustomDataset()
#             )
#         self.test_dataloader = test_dataloader

#     def prediction(self):
#         test_loss = []
#         self.model.eval()
#         with torch.no_grad():
#             for idx, (emg, strain, label) in\
#                 tqdm(enumerate(self.test_dataloader)):
#                     emg = emg.to(self.device)
#                     strain = strain.to(self.device)
#                     label = label.to(torch.float32)
#                     label = label.tp(self.device)
#                     output = self.model(emg, strain)
#                     loss = self.criterion(output, label)
#                     test_loss.append(loss)
#         return test_loss

model_path = './ViT_model/CYJ_data/231108_trial1(delta_strain_all_dataset)_ViT_LR0.0001.pt'
device = torch.device('cuda')
model = ViT_Regression(
    p=hyperparameter_training['patch_size'],
    model_dim=hyperparameter_training['model_dim'],
    hidden_dim=hyperparameter_training['hidden_dim'],
    hidden1_dim=hyperparameter_training['hidden1_dim'],
    hidden2_dim=hyperparameter_training['hidden2_dim'],
    n_output=hyperparameter_training['n_output'],
    n_heads=hyperparameter_training['n_heads'],
    n_layers=hyperparameter_training['n_layers'],
    n_patches=8,
    dropout_p=hyperparameter_training['dropout_p'],
    pool='mean'
    ).to(device)
model.load_state_dict(
    torch.load(model_path, map_location=device))
model.eval()

# 1*400*2*4
# 1*400*1*1
with torch.no_grad():
    traced_model = torch.jit.trace(
        model,
        example_inputs=torch.randn((1, 8, 20, 20)).float().to(device)
            )
traced_model.save("./ViT_model/traced_script_module/231108_ViT2.pt")
