# -*- coding: utf-8 -*-
"""
Created on Mon May  1 13:38:19 2023

@author: mleem
"""

from mat73 import loadmat
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing import *
from config import *
from model import *
from main import *

class DataPredictor:
    def __init__(self, start_time, model_path, opt, device):
        self.start_time = start_time
        self.model_path = model_path
        self.opt = opt
        self.device = device
        self.model_load()

    def model_load(self):
        # model load
        model = ViT(p=int(8), model_dim=self.opt["model_dim"],
                    hidden_dim=self.opt["hidden_dim"],
                    n_class=self.opt["n_class"],
                    n_heads=self.opt["n_head"],
                    n_layers=self.opt["n_layer"],
                    n_patches=self.opt["n_patch"],
                    dropout_p=self.opt["dropout_p"])
        model.load_state_dict(torch.load(self.model_path))
        model.to(self.device)
        self.model = model

    def data_transform(self, x):
        x = torch.FloatTensor(x)
        x = torch.stack([x], dim=0).float()
        return x

    def prediction(self, x):
        self.model.eval()
        with torch.no_grad():
            x = self.data_transform(x)
            x = x.to(self.device)
            output = self.model(x)
        return output

if __name__ == "__main__":

    subject_list = ["4"]
    for sub in subject_list:

        # data_path = "./isometric_hand_gestures/s1.mat"
        # window_sample, w_emg, label = emg_preprocessing(data_path)
        emg_path = "./isometric_hand_gestures/s%s_emg.npy" % str(sub)
        label_path = "./isometric_hand_gestures/s%s_label.npy" % str(sub)

        w_emg = torch.from_numpy(np.load(emg_path))
        re_w_label = torch.from_numpy(np.load(label_path))
        
        n_correct = int(0)

        # emg shape: data_length * window_sample * 8 * 16
        model_path_0 = "./ViT_model/ViT_s%s_LR0.0001.pt" % str(sub)
        device_0 = torch.device('cuda')
        opt = hyperparameter_defaults
        start_time = time.time()
        predictor0 = DataPredictor(start_time, model_path_0, opt, device_0)

        for data_num, x_emg in enumerate(w_emg):
            mid_time = time.time()
            out_0 = predictor0.prediction(x_emg)
            _, pred_0 = torch.max(out_0, 1)
            pred_0 = pred_0.cpu().numpy()[0]
            if pred_0 == re_w_label[data_num].numpy():
                n_correct += 1
            
        end_time = time.time()
        print((end_time - mid_time))
        print((end_time - mid_time)/len(w_emg))
        print((end_time - start_time))
        print((end_time - start_time)/len(w_emg))
        print("Accuracy: %.6f" % float(n_correct/len(w_emg)))
