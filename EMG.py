#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive, files
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import random
import math
from torch.utils.data import TensorDataset, DataLoader
import pickle

# Google drive mount
drive.mount('/content/drive')

'''
# Load raw data
raw_filePath = '/content/drive/MyDrive/s1.mat'
with open(raw_filePath, 'r') as f:
    Subject_1 = h5py.File(raw_filePath, 'r')
extensor = torch.tensor(np.array(Subject_1.get('emg_extensors')))                   # [8, 8, X]
flexor = torch.tensor(np.array(Subject_1.get('emg_flexors')))                       # [8, 8, X]
labels = torch.tensor(np.array(Subject_1.get('adjusted_class'))).unsqueeze(0)       # [1, 1, X]
del Subject_1

# Decrease the sampling rate: 2048Hz -> 1024Hz
halve = nn.AvgPool1d(2, stride = 2)
max = nn.MaxPool1d(2, stride = 2)
extensor = halve(extensor).transpose(1,2).transpose(0,1)                            # [X, 8, 8]
flexor = halve(flexor).transpose(1,2).transpose(0,1)                                # [X, 8, 8]
labels = max(labels.float()).transpose(1,2).transpose(0,1).to(torch.long)           # [X, 1, 1]

# RMS
sampling_rate = 1024        # [Hz]
window_size = (75 * sampling_rate) // 1000
def sEMG_rms(raw_data):
    tmp = torch.zeros(window_size + raw_data.size(0), raw_data.size(1), raw_data.size(2))
    post_data = torch.zeros_like(raw_data)
    tmp[window_size:] = raw_data
    for i in range(window_size, tmp.size(0)):
        post_data[i - window_size, :, :] = tmp[i - window_size : i].pow(2).mean(0).pow(0.5)
    del tmp
    return post_data
extensor = sEMG_rms(extensor).unsqueeze(1)      # [X, 1, 8, 8]
flexor = sEMG_rms(flexor).unsqueeze(1)          # [X, 1, 8, 8]
emg = torch.cat((extensor, flexor), 1)          # [X, 2, 8, 8]
del extensor
del flexor

# Slice by sequence length
sampling_rate = 1024        # [Hz]
sequence_length = (200 * sampling_rate) // 1000
overlapping = sequence_length // 4
emg = emg[:emg.size(0) - sequence_length]
labels = labels[sequence_length:]
emg = np.array(emg.unfold(0, sequence_length, overlapping).transpose(3, 4).transpose(2, 3).transpose(1, 2))
labels = np.array(labels.unfold(0, sequence_length, overlapping).squeeze(1).squeeze(1))
i = 0
indices = []
while i < len(labels):
    if labels[i, 0] != labels[i, sequence_length - 1]:
        indices.append(i)
    i += 1
emg = np.delete(emg, indices, 0)
labels = np.delete(labels, indices, 0)
emg = torch.FloatTensor(emg)
labels = torch.LongTensor(labels)[:,0]
del indices
del sampling_rate
del overlapping
del i

# Train / Test data slice
class_list = []
for i in range(66):
    class_list.append([])
for i in range(labels.size(0)):
    class_list[labels[i]].append(i)
test_list = []
for i in range(66):
    test_list.append([])
for i in range(66):
    test_list[i] = random.sample(class_list[i], len(class_list[i]) // 5)
for i in range(1, 66):
    for j in range(len(test_list[i])):
        test_list[0].append(test_list[i][j])
test_list[0].sort()
test_emg = emg[torch.tensor(test_list[0])]
labels = labels.tolist()
for i in range(len(test_list[0])):
    labels.remove(labels[test_list[0][i] - i])
train_emg = emg[torch.tensor(labels)]
test_labels = torch.LongTensor(test_list[0])
train_labels = torch.LongTensor(labels)
del emg
del labels
del class_list
del test_list
'''
sequence_length = 204
# Load emg & labels for training and test
train_emg_filePath = '/content/drive/MyDrive/train_emg.p'
test_emg_filePath = '/content/drive/MyDrive/test_emg.p'
train_labels_filePath = '/content/drive/MyDrive/train_labels.p'
test_labels_filePath = '/content/drive/MyDrive/test_labels.p'
with open(train_emg_filePath, 'rb') as f:
  train_emg = pickle.load(f)
with open(test_emg_filePath, 'rb') as f:
  test_emg = pickle.load(f)
with open(train_labels_filePath, 'rb') as f:
  train_labels = pickle.load(f)
with open(test_labels_filePath, 'rb') as f:
  test_labels = pickle.load(f)

# wandb
get_ipython().system('pip install wandb --upgrade')
import wandb
wandb.login()

# Sweep_Configuration
sweep_config = {
    'name' : 'bayes_test',
    'method' : 'random',
    'metric' : {
        'name' : 'accuracy',
        'goal' : 'maximize'
    },
    'parameters' : {
        'dropout' : {
            'values' : [0.3, 0.4]
        },
        'batch_size' : {
            'values' : [32, 64]
        },
        'num_features' : {
            'distribution' : 'q_log_uniform',
            'q' : 1,
            'min' : math.log(128),
            'max' : math.log(256)
        },
        'hidden_features' : {
            'distribution' : 'q_log_uniform',
            'q' : 1,
            'min' : math.log(128),
            'max' : math.log(256)
        },
        'n_layers' : {
            'values' : [3, 4]
        }
    }
}

# Dataset
def SweepDataset(args):
  batch_size = args['batch_size']
  train_data = TensorDataset(train_emg, train_labels)
  train_loader = DataLoader(train_data, shuffle = True, batch_size = batch_size, drop_last = True)
  test_data = TensorDataset(test_emg, test_labels)
  test_loader = DataLoader(test_data, shuffle = False, batch_size = 1)

  return train_loader, test_loader

# Model
class EMGRU(nn.Module):
  def __init__(self, args):
    super(EMGRU, self).__init__()

    self.device = args['device']
    self.input_dim = args['num_features']
    self.hidden_dim = args['hidden_features']
    self.n_layers = args['n_layers']
    self.output_dim = args['output_dim']
    self.sequence_length = args['sequence_length']
    self.dropout = args['dropout']
    self.conv_layer = nn.Sequential(
        nn.Conv2d(2, 2, kernel_size = 5, padding = 2),
        nn.BatchNorm2d(2),
        nn.ReLU(),
        nn.Conv2d(2, 2, kernel_size = 5, padding = 2),
        nn.BatchNorm2d(2),
        nn.ReLU()
    )
    self.fc1 = nn.Sequential(
        nn.Linear(128, self.input_dim),
        nn.BatchNorm1d(self.input_dim),
        nn.ReLU()
    )
    self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers, batch_first = True, dropout = self.dropout)
    self.fc2 = nn.Sequential(
        nn.ReLU(),
        nn.Linear(self.hidden_dim, self.output_dim)
    )

    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal(m.weight.data)
        m.bias.data.zero_()

  def forward(self, x, h):
    x = x.reshape(-1, 2, 8, 8)
    out = self.conv_layer(x)
    out = self.fc1(out.flatten(start_dim=1))
    out, h = self.gru(out.reshape(-1, self.sequence_length, self.input_dim), h)
    out = self.fc2(out[:, -1])
    return out, h

  def init_hidden(self, batch_size):
    for m in self.modules():
      if isinstance(m, nn.GRU):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden

# Train / Test
def train(args, loader, wandb):
  model = EMGRU(args).to(args['device'])
  loss_function = nn.CrossEntropyLoss()
  wandb.watch(model, loss_function, log = 'all', log_freq = 10)
  optim = torch.optim.Adam(model.parameters(), lr = 0.001)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode = 'min', factor = 0.5, patience = 2, verbose = True)
  model.train()

  print('Training start.')
  total_time = 0
  epochs = 100
  batch_size = args['batch_size']

  for epoch in range(1, epochs + 1):
    avg_loss_by_epoch = []
    start = time.time()
    h = model.init_hidden(batch_size)
    avg_loss = 0
    for x, y_true in loader:
      h = h.data
      model.zero_grad()

      out, h = model(x.to(args['device']).float(), h)
      loss = loss_function(out, y_true.to(args['device']).to(torch.long))
      loss.backward()
      optim.step()
      avg_loss += loss.item()

    end = time.time()
    total_time += end - start
    total_loss = avg_loss / len(loader)
    if total_loss < 1e-5:
      break
    scheduler.step(total_loss)
    avg_loss_by_epoch.append(total_loss)
    print(f'{epoch} / {epochs}. Time : {end - start}. Total loss : {total_loss}')

  print(f'Training completed! Total training time : {total_time}')
  torch.save(model.state_dict(), '/content/drive/MyDrive/model.pt')

def test(args, loader, wandb):
  model = EMGRU(args).to(args['device'])
  model.load_state_dict(torch.load('/content/drive/MyDrive/model.pt'))
  loss_func = nn.CrossEntropyLoss()
  model.eval()

  print('Test Start!')
  start = time.time()

  test_loss, correct, total = 0, 0, 0
  for x, y_true in loader:
      h = model.init_hidden(1)
      out, h = model(x.to(args['device']).float(), h)
      test_loss += loss_func(out, y_true.to(args['device']).to(torch.long)).item()
      pred = out.max(1, keepdim = True)[1]
      correct += pred.eq(y_true.to(args['device']).to(torch.long).view_as(pred)).sum().item()
      total += 1
  end = time.time()
  average_loss = test_loss / total
  accuracy = 100 * correct / total
  wandb.log({'accuracy' : accuracy})

  print(f'Time : {end - start}, Average loss : {average_loss}, Accuracy :{correct} / {total}, {accuracy}')

# Sweep
def run_sweep(config=None):
  wandb.init(config = config)

  w_config = wandb.config

  args = {
      'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
      'num_features' : w_config.num_features,
      'hidden_features' : w_config.hidden_features,
      'n_layers' : w_config.n_layers,
      'output_dim' : 66,
      'sequence_length' : sequence_length,
      'dropout' : w_config.dropout,
      'batch_size' : w_config.batch_size,
  }

  train_loader, test_loader = SweepDataset(args)
  train(args, train_loader, wandb)
  test(args, test_loader, wandb)

# Wandb running
sweep_id = wandb.sweep(sweep_config, project = 'emg-gru', entity = 'kjhy133676')
wandb.agent(sweep_id, run_sweep, count = 3)

