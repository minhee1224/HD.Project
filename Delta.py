# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 22:06:28 2023

@author: srbl
"""
import pickle
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import torchinfo

# Load emg & labels for training and test
train_emg_filePath = \
    'C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\train_emg_all.p'
test_emg_filePath = \
    'C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\test_emg_all.p'
train_delta_strain_filePath = \
    'C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\train_delta_strain_all.p'
test_delta_strain_filePath = \
    'C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\test_delta_strain_all.p'
with open(train_emg_filePath, 'rb') as f:
  train_emg = pickle.load(f)
with open(train_delta_strain_filePath, 'rb') as f:
  train_strain = pickle.load(f)
with open(test_emg_filePath, 'rb') as f:
  test_emg = pickle.load(f)
with open(test_delta_strain_filePath, 'rb') as f:
  test_strain = pickle.load(f)
validation_emg_filePath = \
    'C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\validation_emg_all.p'
validation_delta_strain_filePath = \
    'C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\validation_delta_strain_all.p'
with open(validation_emg_filePath, 'rb') as f:
  validation_emg = pickle.load(f)
with open(validation_delta_strain_filePath, 'rb') as f:
  validation_strain = pickle.load(f)
random_emg_filePath = \
    'C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\random_emg_all.p'
random_delta_strain_filePath = \
    'C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\random_delta_strain_all.p'
with open(random_emg_filePath, 'rb') as f:
  random_emg = pickle.load(f)
with open(random_delta_strain_filePath, 'rb') as f:
  random_strain = pickle.load(f)
train_emg = torch.cat((train_emg,test_emg),dim=0)
train_strain = torch.cat((train_strain,test_strain),dim=0)
  
# Dataset
def SweepDataset(args):
  batch_size = args['batch_size']
  train_data = TensorDataset(train_emg, train_strain)
  train_loader = DataLoader(train_data, shuffle = True, batch_size = batch_size, drop_last = True)
  validation_data = TensorDataset(validation_emg, validation_strain)
  validation_loader = DataLoader(validation_data, shuffle = False, batch_size = 1)
  random_data = TensorDataset(random_emg, random_strain)
  random_loader = DataLoader(random_data, shuffle = False, batch_size = 1)

  return train_loader, validation_loader, random_loader

# Model
class HD(nn.Module):
    def __init__(self,args):
        super(HD, self).__init__()
        
        self.device = args['device']
        self.channel1 = args['channel1']
        self.channel2 = args['channel2']
        self.channel3 = args['channel3']
        self.input_dim = args['input_dim']
        self.hidden_dim = args['hidden_dim']
        self.layers = args['gru_layers']
        self.dropout = args['dropout']
        self.fc2_dim = args['fc2_dim']
        
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(8,self.channel1,kernel_size=(1,5),stride=(1,5)),                    # c1*20*4
            nn.ReLU(),
            nn.Conv2d(self.channel1,self.channel2,kernel_size=(1,4)),                     # c2*20*1
            nn.ReLU(),
            nn.BatchNorm2d(self.channel2),
            nn.ReLU()
            )
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(self.channel2,self.channel3,kernel_size=4),                         # c3*16
            nn.ReLU(),
            nn.BatchNorm1d(self.channel3),
            nn.ReLU()
            )
        self.fc1 = nn.Sequential(
            nn.Linear(self.channel3, self.input_dim),
            nn.ReLU()
            )
        self.gru = nn.GRU(self.input_dim,self.hidden_dim,self.layers,batch_first=True,dropout=self.dropout)
        self.fc2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.fc2_dim),
            nn.ReLU()
            )
        self.fc3 = nn.Linear(self.fc2_dim,1)
        
    def forward(self,emg):
        emg = emg.transpose(1,2).reshape(-1,8,20,20)
        out = self.conv_layer1(emg)
        out = out.squeeze(3)
        out = self.conv_layer2(out)
        out = out.transpose(1,2)
        out = self.fc1(out)
        out, h = self.gru(out)
        out = out[:,-1]
        out = self.fc2(out)
        out = self.fc3(out)
        out = out.squeeze(1)
        
        return out

# Train
def train(args, loader):
  model = HD(args).to(args['device'])
  loss_function = nn.MSELoss()
  model_optim = torch.optim.Adam(model.parameters(), lr = args['learning_rate'])
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optim, mode = 'min', factor = 0.5, patience = 3, verbose = True)
  model.train()

  print('Training start.')
  total_time = 0
  epochs = args['epochs']

  for epoch in range(1, epochs + 1):
    avg_loss_by_epoch = []
    start = time.time()
    avg_loss = 0
    for emg, angle in loader:
      emg = emg.to(args['device']).float()
      angle = angle.to(args['device']).float()
      model.zero_grad()
      
      out = model(emg)
      loss = torch.sqrt(loss_function(out, angle))
      loss.backward()
      model_optim.step()
      avg_loss += loss.item()

    end = time.time()
    total_time += end - start
    total_loss = avg_loss / len(loader)
    scheduler.step(total_loss)
    avg_loss_by_epoch.append(total_loss)
    print(f'{epoch} / {epochs}. Time : {end - start}. Total loss : {total_loss}')

  print(f'Training completed! Total training time : {total_time}')
  torch.save(model.state_dict(), 'C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\model_1113_cnngru_nowave.pt')

# Test
def test(args, validation_loader, random_loader):
  model = HD(args).to(args['device'])
  model.load_state_dict(torch.load('C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\model_1113_cnngru_nowave.pt'))
  loss_func = nn.MSELoss()
  model.eval()
  fontdict = {
      'fontname':'Arial',
      'fontsize':20,
      'style':'normal'
      }
  plt.rcParams['font.family'] = 'Arial'
  plt.rcParams['font.size'] = 20
  plt.rcParams['font.style'] = 'normal'
  print('Test Start!')

  validation_error = np.zeros(len(validation_loader))
  validation_predicted = np.zeros(len(validation_loader))
  validation_truth = np.zeros(len(validation_loader))
  validation_loss, validation_total = 0, 0
  validation_minmax = torch.abs(validation_strain)
  validation_minmax = (torch.max(validation_minmax) - torch.min(validation_minmax)).item()
  
  start = time.time()
  for emg, angle in validation_loader:
     emg = emg.to(args['device']).float()
     angle = angle.to(args['device']).float()
     out = model(emg)
     loss = torch.sqrt(loss_func(out, angle)).item()
     validation_loss += loss
     validation_predicted[validation_total] = out.item()
     validation_truth[validation_total] = angle.item()
     validation_error[validation_total] = 100*(loss/validation_minmax)
     validation_total += 1
  end1 = time.time()
  random_loss, random_total = 0, 0
  random_minmax = torch.abs(random_strain)
  random_minmax = (torch.max(random_minmax)-torch.min(random_minmax)).item()
  random_error = np.zeros(len(random_loader))
  random_predicted = np.zeros(len(random_loader))
  random_truth = np.zeros(len(random_loader))
  for emg, angle in random_loader:
    emg = emg.to(args['device']).float()
    angle = angle.to(args['device']).float()
    out = model(emg)
    loss = torch.sqrt(loss_func(out, angle)).item()
    random_loss += loss
    random_predicted[random_total] = out.item()
    random_truth[random_total] = angle.item()
    random_error[random_total] = 100*(loss/random_minmax)
    random_total += 1
  end2 = time.time()
  validation_average_loss = validation_loss / validation_total
  validation_nrmse_max = np.max(validation_error)
  validation_nrmse_avg = np.mean(validation_error)
  random_nrmse_max = np.max(random_error)
  random_nrmse_avg = np.mean(random_error)
  plt.scatter(validation_truth,validation_predicted)
  plt.xlim((-0.035,0.035))
  plt.ylim((-0.035,0.035))
  plt.xlabel("True strain [mV]",**fontdict)
  plt.ylabel("Predicted strain [mV]",**fontdict)
  plt.show()
  print(f'Time : {end1 - start}, Average loss : {validation_average_loss}')
  print(f'Max: {validation_nrmse_max}, Avg: {validation_nrmse_avg}')
  random_average_loss = random_loss / random_total
  plt.scatter(random_truth,random_predicted)
  plt.xlim((-0.035,0.035))
  plt.ylim((-0.035,0.035))
  plt.xlabel("True strain [mV]",**fontdict)
  plt.ylabel("Predicted strain [mV]",**fontdict)
  plt.show()
  print(f'Time : {end2 - end1}, Average loss : {random_average_loss}')
  print(f'Max: {random_nrmse_max}, Avg: {random_nrmse_avg}')
  torchinfo.summary(model, (1,400,8), verbose=2)
  
args = {
    'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
    'channel1' : 43,
    'channel2' : 79,
    'channel3' : 340,
    'input_dim' : 391,
    'hidden_dim' : 308,
    'gru_layers' : 1,
    'dropout' : 0.2,
    'fc2_dim' : 27,
    'learning_rate' : 0.001053,
    'batch_size' : 64,
    'epochs' : 100
}

def trace(args):
    model = HD(args).to(args['device'])
    model.load_state_dict(torch.load('C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\model_1108_cnngru.pt'))
    model.eval()
    with torch.no_grad():
        example = torch.rand(1,200,8).to(args['device'])
        traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save('C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\traced_model_1108_cnngru.pt')
train_loader, validation_loader, random_loader = SweepDataset(args)
train(args, train_loader)
test(args, validation_loader, random_loader)
# trace(args)