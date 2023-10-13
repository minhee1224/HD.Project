# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 16:22:59 2023

@author: mleem
"""


import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb


from einops.layers.torch import Rearrange
from einops import rearrange
from collections import OrderedDict, Counter
from scipy import signal


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset,\
    WeightedRandomSampler
from torchvision import transforms


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score


from test_data import *


################################################################
################################################################
# Preprocessing
################################################################
################################################################


class Preprocess:
    @staticmethod
    def parameter_cal(
            data: pd.DataFrame,
            sampling_freq=2000,  # hz
            window_size=200,  # ms
            overlapping_ratio=0.75,
            time_advance=100,  # ms
            label_half_size=5  # ms
    ):
        total_sample = len(data)
        increment = float(1000/sampling_freq)  # ms per sample
        label_half_width_ratio = float(label_half_size/time_advance)
        window_sample = int(np.floor(float(
            window_size/increment
        )))
        advance_sample = int(np.floor(float(
            time_advance/increment
        )))
        moving_sample = int(np.floor(float(
            (1 - overlapping_ratio)*window_sample
        )))
        label_sample = int(np.floor(float(
            2*label_half_size/increment
        )))

        train_number = int(np.floor(
            (total_sample
             - window_sample
             - advance_sample * (1 + label_half_width_ratio)
             ) / moving_sample
        ))  # available training number - 1
        # available training samples
        train_sample = int(
            window_sample
            + advance_sample * (1 + label_half_width_ratio)
            + moving_sample*train_number
        )
        train_sample_per_label =\
            train_sample - moving_sample*train_number - window_sample
        train_number = train_number + 1
        return [
            train_number,
            train_sample,
            train_sample_per_label,
            window_sample,
            moving_sample,
            advance_sample,
            label_sample
        ]

    @staticmethod
    def data_row_wise_indexing(
            data: pd.Series,
            start_index: int,
            end_index: int):
        data = torch.from_numpy(np.array(data).astype(np.float))
        data = torch.index_select(data, 0, torch.from_numpy(
            np.arange(start_index, end_index)
        ))
        return data

    @staticmethod
    # total sample -> N * window
    def data_window_division(
            data: torch.Tensor,
            window_sample: int,
            moving_sample: int
    ):
        data = data.unfold(0, window_sample, moving_sample)  # N * window
        return data

    @staticmethod
    # N * window -> total sample
    def data_window_combination(
            data: torch.Tensor
    ):
        data = data.view([-1, 1])
        data = data.squeeze(1)
        return data

    @staticmethod
    # window, numpy array
    def fast_fourier_transform(
            emg_data: np.array,
            sampling_freq=2000
    ):
        data_length = len(emg_data)
        data_time = float(data_length/sampling_freq)  # sec

        freq_list = np.arange(data_length)
        freq_list = freq_list/data_time
        freq_list = freq_list[range(int(data_length/2))]

        freq_emg_list = np.fft.fft(list(emg_data))/data_length
        freq_emg_list = freq_emg_list[range(int(data_length/2))]
        return freq_list, abs(freq_emg_list)  # x_data, y_data

    @staticmethod
    def fft_plotting(
            x_data: np.array,
            y_data: np.array,
            title: str,
            save_path=None
    ):
        plt.figure(figsize=(12, 5))
        plt.plot(x_data, y_data)
        plt.title(title)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Intensity")
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        return 0

    @staticmethod
    # numpy array
    def remove_mean(
            data: np.array
    ):
        data -= np.mean(data)
        return data

    @staticmethod
    # numpy array
    def notch_filter(
            data: np.array,
            notch_freq=50,
            lowest_freq=30,
            highest_freq=250,
            sampling_freq=2000,
            normalized=False
    ):
        quality_factor = notch_freq/(highest_freq - lowest_freq)
        notch_freq = notch_freq/(sampling_freq/2) if normalized else notch_freq

        numerator, denominator =\
            signal.iirnotch(notch_freq, quality_factor, sampling_freq)
        filtered_data = signal.lfilter(numerator, denominator, data)
        return filtered_data

    @staticmethod
    # numpy array
    def bandpass_filter(
            data: np.array,
            low_f=30,
            high_f=250,
            order=4,
            sampling_freq=2000,
            normalized=False
    ):
        low_f = low_f/(sampling_freq/2) if normalized else low_f
        high_f = high_f/(sampling_freq/2) if normalized else high_f
        numerator, denominator =\
            signal.butter(order, [low_f, high_f], btype='bandpass')
        filtered_data = signal.filtfilt(numerator, denominator, data)
        return filtered_data

    @staticmethod
    # numpy array
    def rectification(
            data: np.array
    ):
        data = abs(data)
        return data

    @staticmethod
    # numpy array
    def moving_rms_smoothing(
            data: np.array,
            smoothing_window=10.0,  # ms
            sampling_freq=2000
    ):
        data = pd.DataFrame(data)
        smoothed_data = data.pow(2).rolling(
            window=round(
                (smoothing_window/1000)/(1/sampling_freq)
            ),
            min_periods=1
        ).mean().apply(np.sqrt, raw=True)
        return smoothed_data

    @staticmethod
    def standard_value_calculation(
            data: np.array,
            mvc_flag: bool,  # mvc or max
            mvc_value=100.0  # arbitrary value
    ):
        if mvc_flag:
            standard_value = mvc_value
        else:
            standard_value = np.max(data)
        return standard_value

    @staticmethod
    # numpy array
    def normalization(
            data: np.array,
            standard_value: float
    ):
        data = data * 100 / standard_value
        return data

    @staticmethod
    def window_wise_normalization(
            tensor_list: list,  # list of 2d tensors
            width_number: int,
            height_number: int
    ):
        for tensor_num, tensor in enumerate(tensor_list):
            tensor = tensor.unsqueeze(2)  # N * window * 1
            if tensor_num == 0:
                data = tensor
            else:
                # N * window * tensor num
                data = torch.cat((data, tensor), dim=2)
        data = data.unsqueeze(3)  # N * window * tensor num * 1
        data_number = data.shape[0]
        window_sample = data.shape[1]
        # N * window * height_number * width_number
        data = data.view(
            [data_number, window_sample, height_number, width_number]
        )
        # window * N * height_number * width_number
        data = data.permute(1, 0, 2, 3)
        # mean, std = N * height_number * width_number
        normalize_data = transforms.Normalize(
            mean=torch.mean(data, dim=0),
            std=torch.std(data, dim=0)
        )
        data = normalize_data(data)
        # N * window * height_number * width_number
        data = data.permute(1, 0, 2, 3)
        return data

    @staticmethod
    def strain_calibration(
            strain_data: np.array,
            angle_data: np.array,
            polynomial_order=2,
            random_state=42,
            data_division_num=10.0,
            test_size=0.35
    ):
        calib_data = pd.DataFrame(columns=["strain", "angle"])
        calib_data["strain"] = strain_data
        calib_data["angle"] = angle_data
        calib_data = calib_data.dropna(axis=0)
        calib_data.reset_index(drop=True, inplace=True)
        #######################################################
        # data division for uniform sampling
        min_angle = np.min(calib_data.angle)
        max_angle = np.max(calib_data.angle)
        extra_angle = (max_angle - min_angle) * 0.001
        delta_angle = (
            max_angle - min_angle + extra_angle
        ) / data_division_num
        calib_data["label"] = pd.cut(
            calib_data.angle,
            bins=np.arange(
                min_angle - extra_angle, max_angle + extra_angle, delta_angle
            ),
            labels=[str(s) for s in np.arange(
                min_angle + delta_angle, max_angle + delta_angle, delta_angle
            )]
        )
        label_min_number = calib_data["label"].value_counts().min()
        calib_data = calib_data.groupby("label").apply(
            lambda x: x.sample(
                label_min_number, random_state=random_state
            )
        )
        #######################################################
        # data: X(strain), y(angle)
        strain_train, strain_test, angle_train, angle_test =\
            train_test_split(
                calib_data.strain,
                calib_data.angle,
                test_size=test_size,
                shuffle=True,
                stratify=calib_data.label,
                random_state=random_state
            )
        polynomial = PolynomialFeatures(degree=polynomial_order)
        strain_train_polynomial = polynomial.fit_transform(
            strain_train.values.reshape(-1, 1)
        )
        #######################################################
        # model
        lin_reg_model = LinearRegression(fit_intercept=True)
        lin_reg_model.fit(strain_train_polynomial, angle_train)
        # test
        strain_test_polynomial = polynomial.fit_transform(
            strain_test.values.reshape(-1, 1)
        )
        angle_test_pred = lin_reg_model.predict(strain_test_polynomial)
        # calculation
        angle_nRMSE = 100 * math.sqrt(
            mean_squared_error(angle_test, angle_test_pred)
        ) / (np.max(calib_data.angle) - np.min(calib_data.angle))
        angle_R2 = r2_score(angle_test, angle_test_pred)
        print("angle nRMSE")
        print(angle_nRMSE)
        print("angle R2")  # 1에 가까울수록 good
        print(angle_R2)
        #######################################################
        # coefficient output
        polynomial_second_order_coeffi = lin_reg_model.coef_[2]
        polynomial_first_order_coeffi = lin_reg_model.coef_[1]
        polynomial_bias = lin_reg_model.intercept_
        return [
            polynomial_bias,
            polynomial_first_order_coeffi,
            polynomial_second_order_coeffi
        ]

    @staticmethod
    def emg_preprocess(
            data: pd.DataFrame,  # emg 1~32
            train_sample: int,
            train_sample_per_label: int,
            window_sample: int,
            moving_sample: int,
            height_number=4,
            width_number=8,
            sampling_freq=2000,  # hz
            notch_freq=50,
            lowest_freq=30,
            highest_freq=250,
            bandpass_order=4,
            smoothing_window=10.0  # ms
    ):

        emg_list = list()
        # emg 1~32
        for emg_index in np.arange(height_number*width_number):
            # N * window
            emg_data = Preprocess.data_window_division(
                data=Preprocess.data_row_wise_indexing(
                    data=data.iloc[:, emg_index],
                    start_index=0,
                    end_index=train_sample-train_sample_per_label
                ),
                window_sample=window_sample,
                moving_sample=moving_sample
            )

            emg_data_np = emg_data.numpy()
            for each_window_index in np.arange(len(emg_data_np)):
                # window
                each_window_data = emg_data_np[each_window_index, :]
                ####################################################
                #                    FFT                           #
                ####################################################
                # freq_x, freq_y = Preprocess.fast_fourier_transform(
                #     emg_data=each_window_index,
                #     sampling_freq=sampling_freq
                #     )
                # Preprocess.fft_plotting(
                #     x_data=freq_x,
                #     y_data=freq_y,
                #     title=""
                #     )

                each_window_data = Preprocess.remove_mean(
                    data=each_window_data
                )
                each_window_data = Preprocess.bandpass_filter(
                    data=each_window_data,
                    low_f=lowest_freq,
                    high_f=highest_freq,
                    order=bandpass_order,
                    sampling_freq=sampling_freq,
                    normalized=True
                )
                each_window_data = Preprocess.notch_filter(
                    data=each_window_data,
                    notch_freq=notch_freq,
                    lowest_freq=lowest_freq,
                    highest_freq=highest_freq,
                    sampling_freq=sampling_freq,
                    normalized=True
                )
                each_window_data = Preprocess.rectification(
                    data=each_window_data
                )
                each_window_data = Preprocess.moving_rms_smoothing(
                    data=each_window_data,
                    smoothing_window=smoothing_window,
                    sampling_freq=sampling_freq
                )

                standard_value = Preprocess.standard_value_calculation(
                    data=each_window_data,
                    mvc_flag=False
                )
                each_window_data = Preprocess.normalization(
                    data=each_window_data,
                    standard_value=standard_value
                )

                # update
                emg_data_np[each_window_index, :] =\
                    list(each_window_data.values)
            # N * window
            emg_data = torch.from_numpy(emg_data_np)
            # len(list) = 32
            emg_list.append(emg_data)

        # N * window * height_number * width_number
        final_emg_data = Preprocess.window_wise_normalization(
            tensor_list=emg_list,
            width_number=width_number,
            height_number=height_number
        )

        # (N * window * height_number * width_number)
        return final_emg_data

    @staticmethod
    def strain_preprocess(
            data: pd.Series,  # strain
            polynomial_coeffi: list,  # bias, first order, second order, ...
            train_sample: int,
            train_sample_per_label: int,
            window_sample: int,
            moving_sample: int,
            label_sample: int,
            height_number=1,
            width_number=1,
            sampling_freq=2000
    ):

        # strain data calibration
        calibrated_data = data * 0.0
        for num, coeffi in enumerate(polynomial_coeffi):
            result = data.pow(num)
            calibrated_data += coeffi * result

        ################################################################
        # Strain data for initialization (positional embedding input)
        ################################################################
        # N * window
        strain_data = Preprocess.data_window_division(
            data=Preprocess.data_row_wise_indexing(
                data=calibrated_data,
                start_index=0,
                end_index=train_sample-train_sample_per_label
            ),
            window_sample=window_sample,
            moving_sample=moving_sample
        )

        strain_list = list()
        strain_list.append(strain_data)
        # N * window * height_number * width_number
        final_strain_data = Preprocess.window_wise_normalization(
            tensor_list=strain_list,
            width_number=width_number,
            height_number=height_number
        )

        ################################################################
        # Strain data for label
        ################################################################
        # N * window
        strain_label = Preprocess.data_window_division(
            data=Preprocess.data_row_wise_indexing(
                data=calibrated_data,
                start_index=window_sample,
                end_index=train_sample
            ),
            window_sample=train_sample_per_label,
            moving_sample=moving_sample
        )
        # N * window * 1
        strain_label = strain_label.unsqueeze(2)
        # window * N * 1
        strain_label = strain_label.permute(1, 0, 2)

        # average labels (N * 1)
        final_strain_label = strain_label[-label_sample:, :, :]
        final_strain_label = torch.mean(final_strain_label, dim=0)

        # (N * window * height_number * width_number), (N * 1)
        return final_strain_data, final_strain_label

    @staticmethod
    def total_preprocess(
            emg_data: pd.DataFrame,  # emg 1~32
            strain_data: pd.Series,
            polynomial_coeffi: list,  # bias, first order, second order, ...
            emg_height_number=4,
            emg_width_number=8,
            strain_height_number=1,
            strain_width_number=1,
            sampling_freq=2000,  # hz
            window_size=200,  # ms
            overlapping_ratio=0.75,
            time_advance=100,  # ms
            label_half_size=5,  # ms
            notch_freq=50,
            lowest_freq=30,
            highest_freq=250,
            bandpass_order=4,
            smoothing_window=10.0  # ms
    ):

        [
            train_number, train_sample, train_sample_per_label,
            window_sample, moving_sample, advance_sample, label_sample
        ] = Preprocess.parameter_cal(
            data=emg_data,
            sampling_freq=sampling_freq,
            window_size=window_size,
            overlapping_ratio=overlapping_ratio,
            time_advance=time_advance,
            label_half_size=label_half_size
        )

        final_emg_data = Preprocess.emg_preprocess(
            data=emg_data,
            train_sample=train_sample,
            train_sample_per_label=train_sample_per_label,
            window_sample=window_sample,
            moving_sample=moving_sample,
            height_number=emg_height_number,
            width_number=emg_width_number,
            sampling_freq=sampling_freq,
            notch_freq=notch_freq,
            lowest_freq=lowest_freq,
            highest_freq=highest_freq,
            bandpass_order=bandpass_order,
            smoothing_window=smoothing_window
        )
        final_strain_data, final_strain_label = Preprocess.strain_preprocess(
            data=strain_data,
            polynomial_coeffi=polynomial_coeffi,
            train_sample=train_sample,
            train_sample_per_label=train_sample_per_label,
            window_sample=window_sample,
            moving_sample=moving_sample,
            label_sample=label_sample,
            height_number=strain_height_number,
            width_number=strain_width_number,
            sampling_freq=sampling_freq
        )

        # emg: (N * window * height_number * width_number),
        # strain: (N * window * height_number * width_number),
        # label: (N * 1)
        return window_sample,\
            final_emg_data, final_strain_data, final_strain_label

################################################################
################################################################
# Model
################################################################
################################################################


class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class Embeddings(nn.Module):
    def __init__(self,
                 p: int,  # min(2,4) = 2
                 input_dim: int,  # H * W = 2 * 4
                 model_dim: int,  # D = 32
                 n_patches: int,  # N(window sample): 400
                 dropout_p: float):
        # projected_dim = N * D
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.n_patches = n_patches
        self.dropout_p = dropout_p
        self.to_patch_embedding = mySequential(
            OrderedDict(
                {
                    "rearrange": Rearrange(
                        'b n (h1 p1) (w1 p2) -> b n (h1 w1 p1 p2)',
                        p1=p, p2=p
                    ),
                    "projection": nn.Linear(self.input_dim, self.model_dim)
                })
        )
        # initialize by strain data
        self.to_pos_embedding = mySequential(
            OrderedDict(
                {
                    "rearrange": Rearrange(
                        'b n (h1 p1) (w1 p2) -> b n (h1 w1 p1 p2)',
                        p1=1, p2=1
                    ),
                    "projection": nn.Linear(1, self.model_dim)
                })
        )
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, self.model_dim)
        )  # 1 * (1 * D)
        self.pos_token = nn.Parameter(
            torch.randn(1, 1, self.model_dim)
        )  # 1 * (1 * D)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, emg_input, strain_input):
        emg_projection = self.to_patch_embedding(emg_input)  # b * N * D
        strain_projection = self.to_pos_embedding(strain_input)  # b * N * D
        b, _, _ = emg_projection.shape
        cls_token = self.cls_token.repeat(b, 1, 1)  # b * 1 * D
        pos_token = self.pos_token.repeat(b, 1, 1)  # b * 1 * D
        pos_emb = nn.Parameter(
            torch.cat((pos_token, strain_projection), dim=1)
        )  # b * (N+1) * D
        patch_emb = torch.cat(
            (cls_token, emg_projection), dim=1
        )  # b * (N+1) * D

        return self.dropout(pos_emb + patch_emb)


class MultiHeadSelfAttentionLayer(nn.Module):
    def __init__(self,
                 model_dim: int,  # D
                 n_heads: int,
                 dropout_p: float,  # a probability of a dropout masking
                 drop_hidden: bool):
        super().__init__()

        self.model_dim = model_dim
        self.n_heads = n_heads
        self.dropout_p = dropout_p
        self.drop_hidden = drop_hidden
        self.scaling = (self.model_dim / self.n_heads) ** (1/2)

        self.norm = nn.LayerNorm(self.model_dim)
        self.linear_qkv = nn.Linear(
            self.model_dim, 3*self.model_dim, bias=False)
        self.projection = nn.Identity()\
            if self.drop_hidden else mySequential(
                    nn.Linear(self.model_dim, self.model_dim),
                    nn.Dropout(self.dropout_p)
                    )

    def forward(self, z):
        b, n, _ = z.shape
        qkv = self.linear_qkv(self.norm(z))
        qkv_destack = rearrange(
            qkv,
            "b n (h d qkv_n) -> (qkv_n) b h n d",
            h=self.n_heads,
            qkv_n=3
            )
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
    def __init__(self,
                 model_dim: int,
                 hidden_dim: int,
                 dropout_p: float):
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
    def __init__(self,
                 n_layers: int,
                 model_dim: int,
                 n_heads: int,
                 hidden_dim: int,
                 dropout_p: float,
                 drop_hidden: bool):
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
                    MultiHeadSelfAttentionLayer(
                        self.model_dim,
                        self.n_heads,
                        self.dropout_p,
                        self.drop_hidden
                        ),
                    PositionWiseFeedForwardLayer(
                        self.model_dim,
                        self.hidden_dim,
                        self.dropout_p
                        )
                )
            )
        self.encoder = mySequential(*layers)

    def forward(self, z):
        return self.encoder(z)


class RegressionHead(nn.Module):
    def __init__(self,
                 model_dim: int,
                 n_output: int,
                 hidden1_dim: int,
                 hidden2_dim: int,
                 training_phase: str,
                 pool: str):
        super().__init__()

        self.model_dim = model_dim
        self.n_output = n_output
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.training_phase = training_phase

        assert pool in {'cls', 'mean'},\
            'pool type must be either cls (cls token) or mean (mean pooling)'

        self.pool = pool

        self.norm = nn.LayerNorm(self.model_dim)
        self.hidden1 = nn.Linear(self.model_dim, self.hidden1_dim)
        self.hidden2 = nn.Linear(self.hidden1_dim, self.hidden2_dim)
        self.hidden = nn.Linear(self.hidden2_dim, self.n_output)
        self.block = mySequential(
            self.norm, self.hidden1, self.hidden2, self.hidden
            )

    def forward(self, encoder_output):
        y = encoder_output.mean(dim=1)\
            if self.pool == 'mean' else encoder_output[:, 0]

        return self.block(y)


class ViT_Regression(nn.Module):
    def __init__(self,
                 p: int,  # 2
                 model_dim: int,  # 32
                 hidden_dim: int,
                 hidden1_dim: int,
                 hidden2_dim: int,
                 n_output: int,
                 n_heads: int,
                 n_layers: int,
                 n_patches: int,  # 400
                 dropout_p=0.1,
                 training_phase='p',
                 pool='mean',
                 drop_hidden=True):
        super().__init__()
        input_dim = (p**2)*2

        self.vit_regress = mySequential(
            OrderedDict({
                "embedding": Embeddings(
                    p, input_dim, model_dim,
                    n_patches, dropout_p
                    ),
                "encoder": Encoder(
                    n_layers, model_dim, n_heads,
                    hidden_dim, dropout_p, drop_hidden
                    ),
                "r_head": RegressionHead(
                    model_dim, n_output, hidden1_dim,
                    hidden2_dim, training_phase, pool
                    )
            })
        )

    def forward(self, emg_input, strain_input):
        return self.vit_regress(emg_input, strain_input)


################################################################
################################################################
# Training setting
################################################################
################################################################


class ScheduleOptim():
    def __init__(self,
                 optimizer,
                 n_warmup_steps=10,
                 decay_rate=0.9):
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


def train(
        model, train_loader, scheduler, epoch, criterion, device,
        opt, wandb_set=False
        ):
    model.train()
    train_loss = 0.0

    tqdm_bar = tqdm(enumerate(train_loader))

    for batch_idx, (emg, strain, label) in tqdm_bar:
        emg = emg.to(device)
        strain = strain.to(device)

        label = label.to(torch.float32)
        label = label.to(device)

        scheduler.zero_grad()
        output = model(emg, strain)
        loss = criterion(output, label)
        train_loss += loss.item()

        loss.backward()
        scheduler.step()
        tqdm_bar.set_description(
            "Epoch {} batch {} - train loss: {:.6f}".format(
                epoch, (batch_idx), loss.item()
                )
            )
        if wandb_set == 1:
            wandb.log(
                {
                    "Learning Rate": scheduler.get_lr()
                }
            )
            if (opt.log_interval > 0) and\
                    ((batch_idx + 1) % opt.log_interval == 0):
                wandb.log(
                    {
                        "Training Loss": round(
                            train_loss / opt.log_interval, 6
                            )
                    }
                )

    scheduler.update()
    train_loss /= len(train_loader.dataset)

    return train_loss


def evaluate(
        model, test_loader, criterion, device,
        opt, wandb_set=False
        ):
    model.eval()
    test_loss = 0.0

    tqdm_bar = tqdm(enumerate(test_loader))

    with torch.no_grad():
        for batch_idx, (emg, strain, label) in tqdm_bar:
            emg = emg.to(device)
            strain = strain.to(device)
            label = label.to(torch.float32)

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
                    (batch_idx + 1) / len(test_loader), loss.item()
                    )
                )
            if wandb_set == 1:
                wandb.log(
                    {
                        'Validation Loss':
                            round(loss.item(), 6)
                    }
                )

    test_loss /= len(test_loader.dataset)

    return test_loss


class CustomDataset(Dataset):
    def __init__(self, w_emg, w_strain, w_label):
        self.emg = w_emg
        self.strain = w_strain
        self.label = w_label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        emg = torch.DoubleTensor(self.emg[idx])
        strain = torch.DoubleTensor(self.strain[idx])
        label = torch.DoubleTensor(self.label[idx])
        return emg, strain, label

    def collate_fn(self, data):
        batch_emg, batch_strain, batch_label = [], [], []
        for emg, strain, label in data:
            emg = torch.DoubleTensor(emg)
            strain = torch.DoubleTensor(strain)
            label = torch.DoubleTensor(label)
            batch_emg.append(emg)
            batch_strain.append(strain)
            batch_label.append(label)
        batch_emg = torch.stack(batch_emg, dim=0).float()
        batch_strain = torch.stack(batch_strain, dim=0).float()
        batch_label = torch.stack(batch_label, dim=0).float()
        return batch_emg, batch_strain, batch_label


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss.to(torch.float32)


def testset_prediction(
        model, test_loader, criterion, device,
        wandb_set=False
        ):
    model.eval()
    test_loss = 0.0

    tqdm_bar = tqdm(enumerate(test_loader))

    with torch.no_grad():
        for batch_idx, (emg, strain, label) in tqdm_bar:
            emg = emg.to(device)
            strain = strain.to(device)
            label = label.to(torch.float32)

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
                    (batch_idx + 1) / len(test_loader), loss.item()
                    )
                )

    test_loss /= len(test_loader.dataset)
    if wandb_set == 1:
        wandb.log(
            {
                'Test Loss':
                    round(test_loss, 6)
            }
        )

    return test_loss


def plot_history(history, name):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("%s" % str(name))
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("Training and Validation Loss")
    ax1.plot(history['train_loss'], label="train_loss")
    ax1.plot(history['test_loss'], label="test_loss")
    ax1.set_xlabel("iterations")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title("Learning Rate")
    ax2.plot(history['lr'], label="learning rate")
    ax2.set_xlabel("iterations")
    ax2.set_ylabel("LR")
    plt.show()


# argument number: 25
def ViT_training(
        opt, wandb_set, model_save_dir,
        emg, strain, label,
        patch_size: int,
        model_dim: int,
        hidden_dim: int,
        hidden1_dim: int,
        hidden2_dim: int,
        n_output: int,
        n_heads: int,
        n_layers: int,
        n_patches: int,
        dropout_p: float,
        test_size: float,
        random_state: int,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        n_warmup_steps: int,
        decay_rate: float,
        data_division_num: int,
        pool: str  # mean, cls
        ):
    # dataset
    dataset = CustomDataset(emg, strain, label)
    # dataset division
    angle_value = np.mean(np.array(dataset[:][:][2]), axis=1)
    angle_range = np.linspace(int(round(min(angle_value))),
                              int(round(max(angle_value))),
                              num=int(data_division_num),
                              endpoint=False)
    angle_interval = round(angle_range[1] - angle_range[0], 1)

    # class
    print(len(np.arange(
        angle_range[0] - angle_interval / 2,
        round(max(angle_value), 2),
        angle_interval
        )))
    print(len(np.arange(0, len(angle_range))))
    angle_class = pd.cut(angle_value, bins=np.arange(
        angle_range[0] - angle_interval / 2,
        round(max(angle_value), 2),
        angle_interval
        ), labels=[
            str(s) for s in np.arange(0, len(angle_range))
            ]
        )
    df_angle_class = pd.get_dummies(angle_class)
    df_angle_class = df_angle_class.values.argmax(1)

    SSS1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
        )
    for train_index, valid_test_index in SSS1.split(
            np.arange(len(dataset)), df_angle_class):
        pass

    SSS2 = StratifiedShuffleSplit(
        n_splits=1, test_size=0.5,
        random_state=random_state
        )
    for valid_index, test_index in SSS2.split(
            valid_test_index, df_angle_class[valid_test_index]):
        pass

    train_dataset = Subset(dataset, train_index)
    valid_dataset = Subset(dataset, valid_index)
    test_dataset = Subset(dataset, test_index)
    #######################################
    # same number data for each label
    counter = Counter(df_angle_class)
    label_weights = [
        len(df_angle_class) / list(counter.values())[i]
        for i in range(len(counter))
        ]
    train_weights = [
        label_weights[int(df_angle_class[i])] for i in train_index
        ]
    valid_weights = [
        label_weights[int(df_angle_class[i])] for i in valid_index
        ]
    train_sampler = WeightedRandomSampler(
        torch.DoubleTensor(train_weights), len(train_index)
        )
    valid_sampler = WeightedRandomSampler(
        torch.DoubleTensor(valid_weights), len(valid_index)
        )
    # dataloader implementation
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=train_sampler, drop_last=True,
        collate_fn=dataset.collate_fn
        )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size,
        sampler=valid_sampler, drop_last=True,
        collate_fn=dataset.collate_fn
        )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size,
        drop_last=True, collate_fn=dataset.collate_fn
        )

    device = torch.device('cuda') \
        if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(
        torch.__version__, device)
        )

    model = ViT_Regression(
        p=patch_size,
        model_dim=model_dim,
        hidden_dim=hidden_dim,
        hidden1_dim=hidden1_dim,
        hidden2_dim=hidden2_dim,
        n_output=n_output,
        n_heads=n_heads,
        n_layers=n_layers,
        n_patches=n_patches,
        dropout_p=dropout_p,
        training_phase='p',
        pool='mean',
        drop_hidden=True
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    scheduler = ScheduleOptim(optimizer, n_warmup_steps, decay_rate)
    criterion = RMSELoss()

    patience = 0
    best_loss = 1000
    history = {'train_loss': [], 'test_loss': [], 'lr': []}

    for epoch in range(1, epochs + 1):
        train_loss = train(
            model, train_dataloader, scheduler,
            epoch, criterion, device, opt, wandb_set
            )
        test_loss = evaluate(
            model, valid_dataloader, criterion,
            device, opt, wandb_set
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
            if patience >= 10:
                break

    final_test_loss = testset_prediction(
        model, test_dataloader, criterion,
        device, wandb_set)

    # ViT regression model save
    if model_save_dir is not None:
        torch.save(model.state_dict(), model_save_dir)

    return test_loss, history, final_test_loss


################################################################
################################################################
# Hyperparameter
################################################################
################################################################


# hyperparameter_defaults = {
#     # Training
#     'random_state': 42,
#     'test_size': 0.35,
#     'batch_size': 16,
#     'epochs': 300,
#     'learning_rate': 0.001,
#     # wandb & logging
#     'prj_name': "HD_ViT",
#     'log_interval': 5,
#     # Model
#     'model_dim': 32,
#     'hidden_dim': 64,
#     'hidden1_dim': 16,
#     'hidden2_dim': 4,
#     'n_output': 1,
#     'n_heads': 8,
#     'n_layers': 10,
#     'dropout_p': 0.2,
#     'model_save_dir': "./ViT_model/ViT_test.pt",
#     # Scheduler
#     'n_warmup_steps': 10,
#     'decay_rate': 0.99
#     }


################################################################
################################################################
# Main
################################################################
################################################################

def main(
        wandb_set: bool,
        hyperparameter_defaults: dict,
        model_save_dir: str,
        history_title: str,
        main_emg_data: pd.DataFrame,
        main_strain_data: pd.Series,
        calib_strain_data: np.array,
        calib_angle_data: np.array
        ):
    # calibration result
    # bias, first order, second order, ...
    polynomial_coeffi = Preprocess.strain_calibration(
        strain_data=calib_strain_data,
        angle_data=calib_angle_data,
        polynomial_order=hyperparameter_defaults["polynomial_order"],
        random_state=hyperparameter_defaults["random_state"],
        data_division_num=hyperparameter_defaults["data_division_num"],
        test_size=hyperparameter_defaults["test_size"]
    )
    #############################
    # Preprocessing
    window_sample, emg_data, strain_data, strain_label =\
        Preprocess.total_preprocess(
            emg_data=main_emg_data,
            strain_data=main_strain_data,
            polynomial_coeffi=polynomial_coeffi,
            emg_height_number=hyperparameter_defaults["emg_height_number"],
            emg_width_number=hyperparameter_defaults["emg_width_number"],
            strain_height_number=hyperparameter_defaults[
                "strain_height_number"
                ],
            strain_width_number=hyperparameter_defaults[
                "strain_width_number"
                ],
            sampling_freq=hyperparameter_defaults["sampling_freq"],
            window_size=hyperparameter_defaults["window_size"],
            overlapping_ratio=hyperparameter_defaults["overlapping_ratio"],
            time_advance=hyperparameter_defaults["time_advance"],
            label_half_size=hyperparameter_defaults["label_half_size"],
            notch_freq=hyperparameter_defaults["notch_freq"],
            lowest_freq=hyperparameter_defaults["lowest_freq"],
            highest_freq=hyperparameter_defaults["highest_freq"],
            bandpass_order=hyperparameter_defaults["bandpass_order"],
            smoothing_window=hyperparameter_defaults["smoothing_window"]
        )
    #############################
    # vit training
    df_loss = pd.DataFrame(columns=["learning_rate", "loss", "test_loss"])

    # Device setting
    device = torch.device('cuda')\
        if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(
        torch.__version__, device)
        )

    if wandb_set == 1:
        wandb.init(project=hyperparameter_defaults["prj_name"],
                   entity='minheelee',
                   config=hyperparameter_defaults
                   )
        w_config = wandb.config
        n_patches = window_sample  # FIXED

        test_loss, history, final_test_loss =\
            ViT_training(
                w_config, wandb_set, model_save_dir,
                emg=emg_data, strain=strain_data, label=strain_label,
                patch_size=w_config.patch_size,
                model_dim=w_config.model_dim,
                hidden_dim=w_config.hidden_dim,
                hidden1_dim=w_config.hidden1_dim,
                hidden2_dim=w_config.hidden2_dim,
                n_output=w_config.n_output,
                n_heads=w_config.n_heads,
                n_layers=w_config.n_layers,
                n_patches=n_patches,
                dropout_p=w_config.dropout_p,
                test_size=w_config.test_size,
                random_state=w_config.random_state,
                batch_size=w_config.batch_size,
                epochs=w_config.epochs,
                learning_rate=w_config.learning_rate,
                n_warmup_steps=w_config.n_warmup_steps,
                decay_rate=w_config.decay_rate,
                data_division_num=w_config.data_division_num,
                pool=w_config.pool
                )

        df_loss = pd.concat([
            df_loss,
            pd.DataFrame([{
                'learning_rate': w_config.learning_rate,
                'loss': test_loss,
                'test_loss': final_test_loss}]
                )
            ], ignore_index=True)
        print(df_loss)
        plot_history(history, history_title)
        wandb.run.finish()
    else:
        w_config = hyperparameter_defaults
        n_patches = window_sample  # FIXED

        test_loss, history, final_test_loss =\
            ViT_training(
                w_config, wandb_set, model_save_dir,
                emg=emg_data, strain=strain_data, label=strain_label,
                patch_size=w_config["patch_size"],
                model_dim=w_config["model_dim"],
                hidden_dim=w_config["hidden_dim"],
                hidden1_dim=w_config["hidden1_dim"],
                hidden2_dim=w_config["hidden2_dim"],
                n_output=w_config["n_output"],
                n_heads=w_config["n_heads"],
                n_layers=w_config["n_layers"],
                n_patches=n_patches,
                dropout_p=w_config["dropout_p"],
                test_size=w_config["test_size"],
                random_state=w_config["random_state"],
                batch_size=w_config["batch_size"],
                epochs=w_config["epochs"],
                learning_rate=w_config["learning_rate"],
                n_warmup_steps=w_config["n_warmup_steps"],
                decay_rate=w_config["decay_rate"],
                data_division_num=w_config["data_division_num"],
                pool=w_config["pool"]
                )

        df_loss = pd.concat([
            df_loss,
            pd.DataFrame([{
                'learning_rate': w_config["learning_rate"],
                'loss': test_loss,
                'test_loss': final_test_loss}]
                )
            ], ignore_index=True)
        print(df_loss)
        plot_history(history, history_title)


if __name__ == "__main__":

    hyperparameter_defaults = {
        # Training
        'random_state': 42,
        'data_division_num': 10.0,
        'pool': 'mean',  # or 'cls'
        'test_size': 0.35,
        'batch_size': 16,
        'epochs': 300,
        'learning_rate': 0.001,
        # wandb & logging
        'prj_name': "HD_ViT",
        'log_interval': 5,
        # Model
        'patch_size': 2,  # p
        'model_dim': 32,
        'hidden_dim': 64,
        'hidden1_dim': 16,
        'hidden2_dim': 4,
        'n_output': 1,
        'n_heads': 8,
        'n_layers': 10,
        'dropout_p': 0.2,
        'model_save_dir': "./ViT_model/2310012_test1_ViT.pt",
        # Scheduler
        'n_warmup_steps': 10,
        'decay_rate': 0.99,
        # Strain calibration
        'polynomial_order': 2,
        # preprocessing parameters
        "emg_height_number": 2,  # 4
        "emg_width_number": 4,  # 8
        "strain_height_number": 1,
        "strain_width_number": 1,
        "sampling_freq": 2000,  # hz
        "window_size": 200,  # ms
        "overlapping_ratio": 0.75,
        "time_advance": 100,  # ms
        "label_half_size": 5,  # ms
        "notch_freq": 50,
        "lowest_freq": 30,
        "highest_freq": 250,
        "bandpass_order": 4,
        "smoothing_window": 10.0  # ms
        }

    model_save_dir = "./ViT_model/231013_test1(231012)_ViT_LR%s.pt" %\
        str(hyperparameter_defaults["learning_rate"])
    history_title = "231013_test1(231012)_ViT_LR_%s" %\
        str(hyperparameter_defaults["learning_rate"])

    #############################
    # Data Reading
    #############################
    # data for emg + strain
    main_path = "./Test_data/1012.xlsx"
    main_sheet = "1012_HSJ_prior_test_sleeve"

    main_data_list = emg_data_txt_reading_231012(main_path, main_sheet)
    main_df = pd.DataFrame()
    for i in np.arange(len(main_data_list) - 1):
        main_df["emg" + str(i + 1)] = main_data_list[i]
    main_df["strain"] = main_data_list[-1]

    # data for strain & angle (mocap)
    # strain_calib_csv_path = ...
    # mocap_calib_csv_path = ...
    #############################
    # data split
    main_emg_data = main_df   # dataframe (32 columns)
    main_strain_data = main_df["strain"]  # Series
    calib_strain_data = np.array(main_df["strain"])  # np.array
    calib_angle_data = np.array(main_df["strain"])  # np.array
    #############################
    # main
    main(
        wandb_set=False,
        hyperparameter_defaults=hyperparameter_defaults,
        model_save_dir=model_save_dir,
        history_title=history_title,
        main_emg_data=main_emg_data,
        main_strain_data=main_strain_data,
        calib_strain_data=calib_strain_data,
        calib_angle_data=calib_angle_data
        )
