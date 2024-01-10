# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 22:57:58 2023

@author: minhee
"""

from mat73 import loadmat
from collections import Counter, OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb


from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# def make_patches(emg_input, p=int(8)):
#     ## Generate image patches in a serial manner
#     ############################################
#     ## input
#     # emg_input: (window_sample) * 8 * 16
#     # p (int): size of patch (= 8)
#     ## output
#     # patches (np.ndarray): p * p * C
#     emg_input = emg_input.unfold(2, 8, 8)
#     emg_input = emg_input.permute(2, 0, 1, 3) # 2 * window_sample * 8 * 8
#     patch_size = int(p)
#     patch_num = int(emg_input[0])


class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class Embeddings(nn.Module):
    def __init__(self, p:int, input_dim:int, model_dim:int,
                 n_patches:int, n_strain_patches:int, dropout_p:float):
        # input_dim = h * w = 1 * 1
        # model_dim = D
        # projected_dim = N * D
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.n_patches = n_patches
        self.n_strain_patches = n_strain_patches
        self.dropout_p = dropout_p
        self.to_patch_embedding = mySequential(
            OrderedDict(
                {
                    "rearrange": Rearrange('b n (h1 p1) (w1 p2) -> b n (h1 w1 p1 p2)', p1 = p, p2 = p),
                    "projection": nn.Linear(self.input_dim, self.model_dim)
                })
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.model_dim)) # 1 * 1 * D
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_patches + 1, self.model_dim)) # 1 * (N+1) * D
        self.strain_emb = nn.Parameter(torch.zeros(1, self.n_patches - self.n_strain_patches + 1, self.model_dim)) # 1 * (N-Ns+1) * D
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input1, input2):
        e_projection = self.to_patch_embedding(input1) # b * N * D
        s_projection = self.to_patch_embedding(input2) # b * Ns * D
        b, _, _ = e_projection.shape
        cls_token = self.cls_token.repeat(b, 1, 1) # b * 1 * D
        strain_emb = self.strain_emb.repeat(b, 1, 1) # b * (N-Ns+1) * D
        s_emb = torch.cat((strain_emb, s_projection), dim=1) # b * (N+1) * D
        patch_emb = torch.cat((cls_token, e_projection), dim=1) # b * (N+1) * D

        return self.dropout(self.pos_emb + s_emb + patch_emb)


class MultiHeadSelfAttentionLayer(nn.Module):
    def __init__(self, model_dim:int, n_heads:int, dropout_p:float, drop_hidden:bool):
        # model_dim: D
        # n_heads
        # dropout_p: a probability of a dropout masking
        super().__init__()

        self.model_dim = model_dim
        self.n_heads = n_heads
        self.dropout_p = dropout_p
        self.drop_hidden = drop_hidden
        self.scaling = (self.model_dim / self.n_heads) ** (1/2)

        self.norm = nn.LayerNorm(self.model_dim)
        self.linear_qkv = nn.Linear(self.model_dim, 3*self.model_dim, bias=False)
        self.projection = nn.Identity() if self.drop_hidden else mySequential(nn.Linear(self.model_dim, self.model_dim), nn.Dropout(self.dropout_p))

    def forward(self, z):
        b, n, _ = z.shape
        qkv = self.linear_qkv(self.norm(z))
        qkv_destack = rearrange(qkv, "b n (h d qkv_n) -> (qkv_n) b h n d", h=self.n_heads, qkv_n = 3)
        query, key, value = qkv_destack[0], qkv_destack[1], qkv_destack[2]
        qk_att = torch.einsum('bhqd, bhkd -> bhqk', query, key)
        att = F.softmax(qk_att / self.scaling, dim=-1)

        if self.drop_hidden:
            att = F.dropout(att, p=self.dropout_p)
        
        out = torch.einsum('bhal, bhlv -> bhav', att, value)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)

        return z + out


class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, model_dim, hidden_dim, dropout_p):
        super().__init__()

        self.model_dim = model_dim
        self.hidden_dim = hidden_dim

        self.norm = nn.LayerNorm(self.model_dim)
        self.fc1 = nn.Linear(self.model_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.model_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)

        self.block = mySequential(
            self.norm,
            self.fc1,
            self.activation,
            self.dropout,
            self.fc2,
            self.dropout
        )

    def forward(self, z):
        return z + self.block(z)


class Encoder(nn.Module):
    def __init__(self, n_layers, model_dim, n_heads, hidden_dim, dropout_p, drop_hidden):
        super().__init__()

        self.n_layers = n_layers
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.drop_hidden = drop_hidden

        layers = []

        for _ in range(self.n_layers):
            layers.append(
                mySequential(
                    MultiHeadSelfAttentionLayer(self.model_dim, self.n_heads, self.dropout_p, self.drop_hidden),
                    PositionWiseFeedForwardLayer(self.model_dim, self.hidden_dim, self.dropout_p)
                )
            )
        self.encoder = mySequential(*layers)

    def forward(self, z):
        return self.encoder(z)


class RegressionHead(nn.Module):
    def __init__(self, model_dim, n_output, hidden1_dim, hidden2_dim,
                 training_phase, pool:str):
        super().__init__()

        self.model_dim = model_dim
        self.n_output = n_output
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.training_phase = training_phase

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.pool = pool

        self.norm = nn.LayerNorm(self.model_dim)
        self.hidden1 = nn.Linear(self.model_dim, self.hidden1_dim)
        self.hidden2 = nn.Linear(self.hidden1_dim, self.hidden2_dim)
        self.hidden = nn.Linear(self.hidden2_dim, self.n_output)
        self.block = mySequential(self.norm, self.hidden1,
                                   self.hidden2, self.hidden)

    def forward(self, encoder_output):
        y = encoder_output.mean(dim=1) if self.pool == 'mean' else encoder_output[:, 0]

        return self.block(y)


class ViT_Regression(nn.Module):
    def __init__(self, p, model_dim, hidden_dim, hidden1_dim, hidden2_dim,
                 n_output, n_heads, n_layers, n_patches, n_strain_patches,
                 dropout_p=0.1, training_phase='p', pool='mean',
                 drop_hidden=True):
        super().__init__()
        input_dim = (p**2)

        self.vit_regress = mySequential(
            OrderedDict({
                "embedding": Embeddings(p, input_dim, model_dim,
                                        n_patches, n_strain_patches, dropout_p),
                "encoder": Encoder(n_layers, model_dim, n_heads,
                                   hidden_dim, dropout_p, drop_hidden),
                "r_head": RegressionHead(model_dim, n_output, hidden1_dim,
                                         hidden2_dim, training_phase, pool)
            })
        )

    def forward(self, input1, input2):
        return self.vit_regress(input1, input2)


class ScheduleOptim():
    def __init__(self, optimizer, n_warmup_steps=10, decay_rate=0.9):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.decay = decay_rate
        self.n_steps = 0
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.current_lr = optimizer.param_groups[0]['lr']

    def zero_grad(self):
        self._optimizer.zero_grad()

    def step(self):
        self._optimizer.step()

    def get_lr(self):
        return self.current_lr

    def update(self):
        if self.n_steps < self.n_warmup_steps:
            lr = self.n_steps / self.n_warmup_steps * self.initial_lr
        elif self.n_steps == self.n_warmup_steps:
            lr = self.initial_lr
        else:
            lr = self.current_lr * self.decay

        self.current_lr = lr
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

        self.n_steps += 1


def train(model, train_loader, scheduler, epoch, criterion, device,
          opt, wandb_set=False):
    model.train()
    train_loss = 0.0

    tqdm_bar = tqdm(enumerate(train_loader))

    for batch_idx, (emg, strain, label) in tqdm_bar:
        emg = emg.to(device)
        strain = strain.to(device)
        label = label.to(torch.float32)
        # label = torch.round(label, decimals=1)
        # print("label")
        # print(label)
        label = label.to(device)

        scheduler.zero_grad()
        output = model(emg, strain)
        # print("output")
        # print(output)
        loss = criterion(output, label)
        train_loss += loss.item()

        loss.backward()
        scheduler.step()
        if wandb_set == True:
            wandb.log(
                {
                    "Learning Rate": scheduler.get_lr()
                }
            )
        tqdm_bar.set_description(
            "Epoch {} batch {} - train loss: {:.6f}".format(
                epoch, (batch_idx), loss.item()))
        if wandb_set == True:
            if (opt.log_interval > 0) and ((batch_idx + 1) % opt.log_interval == 0):
                wandb.log(
                    {
                        "Training Loss": round(train_loss/opt.log_interval, 6)    
                    }
                )
    scheduler.update()
    train_loss /= len(train_loader.dataset)

    return train_loss


def evaluate(model, test_loader, criterion, device, opt, wandb_set=False):
    model.eval()
    test_loss = 0.0

    tqdm_bar = tqdm(enumerate(test_loader))
    
    with torch.no_grad():
        for batch_idx, (emg, strain, label) in tqdm_bar:
            emg = emg.to(device)
            strain = strain.to(device)
            label = label.to(torch.float32)
            # label = torch.round(label, decimals=1)
            print("label")
            print(label)
            label = label.to(device)

            output = model(emg, strain)
            print("output")
            print(output)
            loss = criterion(output, label)
            test_loss += loss.item()

            tqdm_bar.set_description(
                "Validation step: {} || Validation loss: {:.6f}".format(
                    (batch_idx + 1)/len(test_loader), loss.item()))
            if wandb_set == True:
                # wandb logging
                wandb.log(
                    {
                        'Validation Loss':
                            round(loss.item(), 6),
                    }
                )

    test_loss /= len(test_loader.dataset)
    test_accuracy=0

    return test_loss, test_accuracy


class CustomForceDataset(Dataset):
    def __init__(self, w_emg, w_strain, w_label):
        self.emg = w_emg
        self.strain = w_strain
        self.label = w_label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        e = torch.FloatTensor(self.emg[idx])
        s = torch.FloatTensor(self.strain[idx])
        y = torch.FloatTensor(self.label[idx])
        return e, s, y

    def collate_fn(self, data):
        batch_e, batch_s, batch_y = [], [], []
        for e, s, y in data:
            e = torch.FloatTensor(e)
            s = torch.FloatTensor(s)
            y = torch.FloatTensor(y)
            batch_e.append(e)
            batch_s.append(s)
            batch_y.append(y)
        batch_e = torch.stack(batch_e, dim=0).float()
        batch_s = torch.stack(batch_s, dim=0).float()
        batch_y = torch.stack(batch_y, dim=0).float()
        return batch_e, batch_s, batch_y


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss.to(torch.float32)


def testset_prediction(model, test_loader, criterion, device, wandb_set=False):

    model.eval()
    test_loss = 0.0

    tqdm_bar = tqdm(enumerate(test_loader))
    
    with torch.no_grad():
        for batch_idx, (emg, strain, label) in tqdm_bar:
            emg = emg.to(device)
            strain = strain.to(device)
            label = label.to(torch.float32)
            # label = torch.round(label, decimals=1)
            print("label")
            print(label)
            label = label.to(device)

            output = model(emg, strain)
            print("output")
            print(output)
            loss = criterion(output, label)
            test_loss += loss.item()
            tqdm_bar.set_description(
                "Test step: {} || Test loss: {:.6f}".format(
                    (batch_idx + 1)/len(test_loader), loss.item()))

    test_loss /= len(test_loader.dataset)
    test_accuracy=0
    if wandb_set == True:
        # wandb logging
        wandb.log(
            {
                'Test Loss':
                    round(test_loss, 6),
            }
        )

    return test_loss, test_accuracy


def ViT_DELSYS_case2_training(opt, wandb_set, model_dir, emg, strain, label,
                              patch_size, MODEL_DIM, HIDDEN_DIM, HIDDEN1_DIM,
                              HIDDEN2_DIM, N_OUTPUT, N_HEAD, N_LAYER, N_PATCH,
                              N_STRAIN_PATCH, DROPOUT_P, TEST_RATIO,
                              RANDOM_STATE, BATCH_SIZE, EPOCHS, DEVICE,
                              LR, N_WARMUP_STEPS, DECAY_RATE):

    # dataset
    dataset = CustomForceDataset(emg, strain, label)
    # dataset division
    angle_value = np.mean(np.array(dataset[:][:][2]), axis=1)
    angle_range = np.linspace(round(min(angle_value), 1),
                              round(max(angle_value), 1),
                              num=10, endpoint=False)
    angle_interval = round(angle_range[1] - angle_range[0], 1)
    RANDOM_STATE = 42

    # class
    angle_class = pd.cut(angle_value, bins = np.arange(
        angle_range[0] - angle_interval, round(max(angle_value), 2),
        angle_interval), labels = [str(s) for s in np.arange(0, len(angle_range))])
    df_angle_class = pd.get_dummies(angle_class)
    df_angle_class = df_angle_class.values.argmax(1)

    SSS1 = StratifiedShuffleSplit(n_splits=1, test_size=TEST_RATIO,
                                 random_state=RANDOM_STATE)
    for train_index, valid_test_index in SSS1.split(
            np.arange(len(dataset)), df_angle_class):
        pass

    SSS2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5,
                                 random_state=RANDOM_STATE)
    for valid_index, test_index in SSS2.split(
            valid_test_index, df_angle_class[valid_test_index]):
        pass

    train_dataset = Subset(dataset, train_index)
    valid_dataset = Subset(dataset, valid_index)
    test_dataset = Subset(dataset, test_index)
    #######################################
    # same number data for each label
    counter = Counter(df_angle_class)
    label_weights = [len(df_angle_class) / list(counter.values())[i]
                      for i in range(len(counter))]
    train_weights = [label_weights[int(df_angle_class[i])] for i in train_index]
    valid_weights = [label_weights[int(df_angle_class[i])] for i in valid_index]
    train_sampler = WeightedRandomSampler(torch.DoubleTensor(train_weights),
                                          len(train_index))
    valid_sampler = WeightedRandomSampler(torch.DoubleTensor(valid_weights),
                                          len(valid_index))
    # dataloader implementation
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  sampler = train_sampler, drop_last=True,
                                  collate_fn=dataset.collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                                 sampler = valid_sampler, drop_last=True,
                                 collate_fn=dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                 drop_last=True, collate_fn=dataset.collate_fn)

    DEVICE = torch.device('cuda') \
        if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(
        torch.__version__, DEVICE))

    model = ViT_Regression(p=int(patch_size), model_dim=MODEL_DIM,
                           hidden_dim=HIDDEN_DIM, hidden1_dim=HIDDEN1_DIM,
                           hidden2_dim=HIDDEN2_DIM, n_output=N_OUTPUT,
                           n_heads=N_HEAD, n_layers=N_LAYER, n_patches=N_PATCH,
                           n_strain_patches=N_STRAIN_PATCH,
                           dropout_p=DROPOUT_P, training_phase='p',
                           pool='mean', drop_hidden=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), LR)
    scheduler = ScheduleOptim(optimizer, N_WARMUP_STEPS, DECAY_RATE)
    criterion = RMSELoss()

    patience = 0
    best_loss = 1000
    history = {'train_loss':[], 'test_loss':[], 'lr':[]}

    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_dataloader, scheduler,
                           epoch, criterion, DEVICE, opt, wandb_set)
        test_loss, _ = evaluate(
            model, valid_dataloader, criterion, DEVICE, opt, wandb_set)
        lr = scheduler.get_lr()
        print("\n[EPOCH: {:2d}], \tModel: ViT, \tLR: {:8.6f}, ".format(
            epoch, lr) \
            + "\tTrain Loss: {:8.6f}, \tTest Loss: {:8.6f} \n".format(
                train_loss, test_loss))

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['lr'].append(lr)

        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break

    final_test_loss, _ = testset_prediction(model, test_dataloader,
                                            criterion, DEVICE, wandb_set)

    # ViT regression model save
    if model_dir is not None:
        torch.save(model.state_dict(), model_dir)

    return test_loss, history, final_test_loss


def plot_history(history, name):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("%s" % str(name))
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("Training and Validation Loss")
    ax1.plot(history['train_loss'], label="train_loss")
    ax1.plot(history['test_loss'], label="test_loss")
    # ax1.plot(history['test_accuracy'], label="test_accuracy")
    ax1.set_xlabel("iterations")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title("Learning Rate")
    ax2.plot(history['lr'], label="learning rate")
    ax2.set_xlabel("iterations")
    ax2.set_ylabel("LR")
    plt.show()
