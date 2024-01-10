# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 16:22:59 2023

@author: mleem
"""


import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import time

# from pyts.image import GramianAngularField, MarkovTransitionField

from einops.layers.torch import Rearrange
from einops import rearrange
from collections import OrderedDict, Counter
from scipy import signal, interpolate


import torch
import torch.fx
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
# Utils
################################################################
################################################################

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def basic_interpolation(standard_time: pd.Series,
                        target_time: pd.Series,
                        target_data: pd.DataFrame):
    target_col = target_data.columns
    if len(target_data) != 1:
        target_synced = pd.DataFrame(columns=target_col)
        for t in target_col:
            temp_data = target_data[str(t)]
            tmp = np.interp(standard_time, target_time, temp_data)
            target_synced[str(t)] = tmp
    else:
        target_synced = pd.DataFrame(
            np.zeros(target_synced.shape[0], len(target_col)),
            columns=target_col)
    return target_synced


def mocap_data_csv_reading(path: str,
                           column_list: list,
                           angle_option: int):  # 1: case 1, 2: case 2
    data = pd.read_csv(path, header=5, sep=",")
    data.columns = column_list
    data = data.iloc[:, 1:]

    # shoulder - elbow - wrist
    if angle_option == 1:
        data["wrist_x"] =\
            0.5 * (data["wrist_medial_x"] + data["wrist_lateral_x"])
        data["wrist_y"] =\
            0.5 * (data["wrist_medial_y"] + data["wrist_lateral_y"])
        data["wrist_z"] =\
            0.5 * (data["wrist_medial_z"] + data["wrist_lateral_z"])
        length_a = np.sqrt(
            (data["shoulder_x"] - data["elbow_x"]).pow(2) +
            (data["shoulder_y"] - data["elbow_y"]).pow(2) +
            (data["shoulder_z"] - data["elbow_z"]).pow(2)
            )
        length_b = np.sqrt(
            (data["wrist_x"] - data["elbow_x"]).pow(2) +
            (data["wrist_y"] - data["elbow_y"]).pow(2) +
            (data["wrist_z"] - data["elbow_z"]).pow(2)
            )
        length_c = np.sqrt(
            (data["shoulder_x"] - data["wrist_x"]).pow(2) +
            (data["shoulder_y"] - data["wrist_y"]).pow(2) +
            (data["shoulder_z"] - data["wrist_z"]).pow(2)
            )
        data["angle"] = np.arccos(
            (length_a.pow(2) + length_b.pow(2) - length_c.pow(2)) /
            (2.0 * length_a * length_b))
        data["angle"] *= (180.0 / math.pi)
    # biceps - elbow - forearm
    elif angle_option == 2:
        length_a = np.sqrt(
            (data["biceps_x"] - data["elbow_x"]).pow(2) +
            (data["biceps_y"] - data["elbow_y"]).pow(2) +
            (data["biceps_z"] - data["elbow_z"]).pow(2)
            )
        length_b = np.sqrt(
            (data["forearm_x"] - data["elbow_x"]).pow(2) +
            (data["forearm_y"] - data["elbow_y"]).pow(2) +
            (data["forearm_z"] - data["elbow_z"]).pow(2)
            )
        length_c = np.sqrt(
            (data["biceps_x"] - data["forearm_x"]).pow(2) +
            (data["biceps_y"] - data["forearm_y"]).pow(2) +
            (data["biceps_z"] - data["forearm_z"]).pow(2)
            )
        data["angle"] = np.arccos(
            (length_a.pow(2) + length_b.pow(2) - length_c.pow(2)) /
            (2.0 * length_a * length_b))
        data["angle"] *= (180.0 / math.pi)

    return data  # dataframe


def strain_mocap_interpolation(strain_data: pd.Series,
                               mocap_data: pd.DataFrame,
                               strain_freq=2000.0
                               ):
    # time [ms]
    strain_time = np.linspace(
        0,
        (len(strain_data) - 1) * (1000.0 / strain_freq),
        len(strain_data)
        )
    mocap_data["time"] *= 1000
    f = interpolate.interp1d(strain_time, np.array(strain_data))
    strain_interp = f(mocap_data.time)

    return pd.Series(strain_interp)  # pd.Series


def strain_data_txt_reading(path: str,
                            sheet_name="Sheet1"
                            ):
    if path[-3:] != "txt":
        data = pd.read_excel(path,
                             sheet_name=sheet_name,
                             index_col=None,
                             header=None)
    else:
        data = pd.read_csv(path, sep=" ")
    data = data.iloc[1:, 1:-2]
    data = data.iloc[:, -1]  # pd.Series
    data.name = 'strain'

    return data  # pd.Series


def emg_data_txt_reading_231011(path: str,
                                sheet_name="Sheet1"
                                ):
    if path[-3:] != "txt":
        data = pd.read_excel(path,
                             sheet_name=sheet_name,
                             index_col=None,
                             header=None)
    else:
        data = pd.read_csv(path, sep=" ")
    data = data.iloc[5:, 1:-2]
    data.columns = ["emg1", "emg2", "emg3", "emg4", "emg5", "emg6",
                    "emg7", "emg8", "emg9", "emg10", "emg11", "emg12",
                    "emg13", "emg14", "emg15", "emg16", "emg17", "emg18",
                    "emg19", "emg20", "emg21", "emg22", "emg23", "emg24",
                    "emg25", "emg26", "emg27", "emg28", "emg29", "emg30",
                    "emg31", "emg32", "strain"]
    data = data.dropna(axis=0)
    data[list(data.columns)[:-1]] =\
        (data[list(data.columns)[:-1]] - 32768) * 0.195

    data_list = []
    data_list.append(data.emg10)
    data_list.append(data.emg9)
    data_list.append(data.emg32)
    data_list.append(data.emg13)
    data_list.append(data.emg22)
    data_list.append(data.emg20)
    data_list.append(data.emg25)
    data_list.append(data.emg23)
    data_list.append(data.strain)

    return data_list


def emg_data_txt_reading_231012(path: str,
                                sheet_name="Sheet1"
                                ):
    if path[-3:] != "txt":
        data = pd.read_excel(path,
                             sheet_name=sheet_name,
                             index_col=None,
                             header=None)
    else:
        data = pd.read_csv(path, sep=" ")
    data = data.iloc[5:, 1:-2]
    data.columns = ["emg1", "emg2", "emg3", "emg4", "emg5", "emg6",
                    "emg7", "emg8", "emg9", "emg10", "emg11", "emg12",
                    "emg13", "emg14", "emg15", "emg16", "emg17", "emg18",
                    "emg19", "emg20", "emg21", "emg22", "emg23", "emg24",
                    "emg25", "emg26", "emg27", "emg28", "emg29", "emg30",
                    "emg31", "emg32", "strain"]
    data = data.dropna(axis=0)
    data[list(data.columns)[:-1]] =\
        (data[list(data.columns)[:-1]] - 32768) * 0.195

# number 1: emg 8
# number 2: emg 11
# number 3: emg 5
# number 4: emg 17
# number 5: emg 14
# number 6: emg 19
# number 7: emg 2
# number 8: emg 29
    data_list = []
    data_list.append(data.emg8)
    data_list.append(data.emg11)
    data_list.append(data.emg5)
    data_list.append(data.emg17)
    data_list.append(data.emg14)
    data_list.append(data.emg19)
    data_list.append(data.emg2)
    data_list.append(data.emg29)
    data_list.append(data.strain)

    return data_list


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
            notch_limit=5.0,
            sampling_freq=2000
    ):
        data_length = emg_data.size
        data_time = float(data_length/sampling_freq)  # sec

        freq_list = np.arange(data_length)
        freq_list = freq_list/data_time
        freq_list = freq_list[range(int(data_length/2))]

        fft = np.fft.fft(emg_data.tolist())
        freq_emg_list = abs(fft)/data_length
        freq_emg_list = freq_emg_list[range(int(data_length/2))]

        # notch freq
        notch_index = np.where(abs(freq_emg_list) > notch_limit)[0]
        notch_freq_list = freq_list[notch_index]

        return freq_list, abs(freq_emg_list), notch_freq_list
        # x_data, y_data, notch_freq

    @staticmethod
    def fft_plotting(
            x_data: np.array,
            y_data: np.array,
            title: str,
            lowest_freq=30,
            highest_freq=250,
            save_path=None
    ):
        # plt.figure(figsize=(12, 5))
        plt.plot(x_data, y_data)
        plt.title(title)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Intensity")
        plt.xlim([lowest_freq, highest_freq])
        plt.ylim([0, 16])
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        return 0

    @staticmethod
    def fft_list_subplots(
            x_data: list,  # set * emg number * freq list
            y_data: list,
            title: str,  # '[FFT] EMG before preprocessing'
            save_path: str,  # './Test_data/graph/1013/number_'
            lowest_freq=30,
            highest_freq=250
    ):
        # For comparison between test sets
        for emg_ind in np.arange(1, len(x_data[0]) + 1):
            # FFT plot
            fig_fft, axs = plt.subplots(
                ((len(x_data) + 1) // 2), 2, sharex=True, figsize=(80, 100))
            fig_fft.suptitle(title + "_number_" + str(emg_ind), fontsize=80)
            for set_ind in np.arange(len(x_data)):
                j = set_ind // 2
                k = set_ind % 2
                axs[j, k].plot(
                    x_data[set_ind][emg_ind - 1],
                    y_data[set_ind][emg_ind - 1],
                    label="test set" + str(set_ind + 1)
                    )
                axs[j, k].set_ylabel("Intensity", fontsize=50)
                axs[j, k].set_xlabel("Frequency [Hz]", fontsize=50)
                axs[j, k].legend(loc="best", fontsize=50)
                axs[j, k].set_xlim(
                    [lowest_freq, highest_freq]
                    )
                axs[j, k].set_ylim([0, 20])
                axs[j, k].tick_params(axis="x", labelsize=50)
                axs[j, k].tick_params(axis="y", labelsize=50)
            fig_fft.tight_layout()
            fig_fft.savefig(
                save_path + "EMG_" + str(emg_ind) + "_" + title + '.png'
                )
            plt.close()

        # For comparison between test sets
        for set_ind in np.arange(len(x_data)):
            # FFT plot
            fig_fft, axs = plt.subplots(
                (len(x_data[0]) // 2), 2, sharex=True, figsize=(80, 100))
            fig_fft.suptitle(title + "_set_" + str(set_ind + 1), fontsize=80)
            for emg_ind in np.arange(len(x_data[0])):
                j = emg_ind // 2
                k = emg_ind % 2
                axs[j, k].plot(
                    x_data[set_ind][emg_ind],
                    y_data[set_ind][emg_ind],
                    label="EMG number" + str(emg_ind + 1)
                    )
                axs[j, k].set_ylabel("Intensity", fontsize=50)
                axs[j, k].set_xlabel("Frequency [Hz]", fontsize=50)
                axs[j, k].legend(loc="best", fontsize=50)
                axs[j, k].set_xlim(
                    [lowest_freq, highest_freq]
                    )
                axs[j, k].set_ylim([0, 20])
                axs[j, k].tick_params(axis="x", labelsize=50)
                axs[j, k].tick_params(axis="y", labelsize=50)
            fig_fft.tight_layout()
            fig_fft.savefig(
                save_path + "SET_" + str(set_ind + 1) + "_" + title + '.png'
                )
            plt.close()

    @staticmethod
    # numpy array
    def remove_mean(
            data: np.array
    ):
        data -= np.mean(data, axis=0)
        return data

    @staticmethod
    # numpy array
    def notch_filter(
            data: np.array,
            quality_value: float,  # 0.001 or 0.0015
            notch_freq=50,
            lowest_freq=30,
            highest_freq=250,
            sampling_freq=2000,
            normalized=False
    ):
        if (
                ((notch_freq % 60) > 0) &
                ((notch_freq % 60) <= 0.05)
                ) |\
            (
                ((60.0 - (notch_freq % 60)) > 0) &
                ((60.0 - (notch_freq % 60)) <= 0.05)
                ):
            print(notch_freq)
            quality_factor =\
                notch_freq / ((highest_freq - lowest_freq) * quality_value)
            print(quality_factor)
            notch_freq = notch_freq / (sampling_freq / 2) if\
                normalized else notch_freq

            numerator, denominator =\
                signal.iirnotch(notch_freq, quality_factor, sampling_freq)
            filtered_data = signal.lfilter(numerator, denominator, data)
        else:
            filtered_data = data
        return filtered_data

    # @staticmethod
    # numpy array, bandstop filter
    # def notch_filter(
    #         data: np.array,
    #         notch_freq=50,
    #         sampling_freq=2000,
    #         order=4
    #         ):

    #     highcut = round(notch_freq, 1)
    #     lowcut = highcut - 0.1

    #     nyq = 0.5 * sampling_freq
    #     low = lowcut / nyq
    #     high = highcut / nyq

    #     if high < 1.0:
    #         numerator, denominator =\
    #             signal.butter(order, [low, high], btype='bandstop')
    #         filtered_data = signal.lfilter(numerator, denominator, data)
    #     else:
    #         filtered_data = data

    #     return filtered_data

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
        # plt.plot(calib_data.angle)
        min_angle = np.min(calib_data.angle)
        max_angle = np.max(calib_data.angle)
        extra_angle = (max_angle - min_angle) * 0.001
        delta_angle = (
            max_angle - min_angle + extra_angle
        ) / data_division_num
        print(min_angle)
        print(extra_angle)
        print(delta_angle)
        print(max_angle)
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
    def each_mvc_calculation(
            data: pd.DataFrame,
            mvc_limit: float,
            mvc_quality_value: float,
            height_number=4,
            width_number=8,
            sampling_freq=2000,  # hz
            lowest_freq=30,
            highest_freq=250,
            bandpass_order=4,
            smoothing_window=10.0  # ms
    ):
        mvc_list = list()
        # emg 1~32
        for emg_index in np.arange(height_number*width_number):
            each_data = np.array(data.iloc[:, emg_index].values)
            # FFT
            freq_x, freq_y, notch_f = Preprocess.fast_fourier_transform(
                emg_data=each_data,
                notch_limit=5.0,
                sampling_freq=sampling_freq
                )
            # Remove mean
            each_data = Preprocess.remove_mean(
                data=each_data
                )
            # Bandpass filter
            each_data = Preprocess.bandpass_filter(
                data=each_data,
                low_f=lowest_freq,
                high_f=highest_freq,
                order=bandpass_order,
                sampling_freq=sampling_freq,
                normalized=True
                )
            # Notch filter
            for each_notch_f in notch_f:
                each_data = Preprocess.notch_filter(
                    data=each_data,
                    quality_value=mvc_quality_value,
                    notch_freq=each_notch_f,
                    lowest_freq=lowest_freq,
                    highest_freq=highest_freq,
                    sampling_freq=sampling_freq
                    )
            # Rectification
            each_data = Preprocess.rectification(
                data=each_data
                )
            # Moving rms smoothing
            each_data = Preprocess.moving_rms_smoothing(
                data=each_data,
                smoothing_window=smoothing_window,
                sampling_freq=sampling_freq
                )
            # MVC calculation
            each_data = each_data[each_data > mvc_limit]
            each_mvc = np.mean(each_data)
            mvc_list.append(each_mvc)
        return np.array(mvc_list)

    @staticmethod
    def mvc_calculation(
            data_list: list,
            mvc_limit: float,
            mvc_quality_value: float,
            height_number=4,
            width_number=8,
            sampling_freq=2000,  # hz
            lowest_freq=30,
            highest_freq=250,
            bandpass_order=4,
            smoothing_window=10.0  # ms
            ):
        print("mvc calculation")
        print(len(data_list))
        for mvc_ind in np.arange(len(data_list)):
            mvc_value = Preprocess.each_mvc_calculation(
                data_list[mvc_ind],
                mvc_limit,
                mvc_quality_value,
                height_number=height_number,
                width_number=width_number,
                sampling_freq=sampling_freq,  # hz
                lowest_freq=lowest_freq,
                highest_freq=highest_freq,
                bandpass_order=bandpass_order,
                smoothing_window=smoothing_window  # ms
                )
            if mvc_ind == 0:
                mvc_value_list = mvc_value
            else:
                mvc_value_list += mvc_value
        mvc_value_list /= len(data_list)
        return mvc_value_list  # np.array

    @staticmethod
    def emg_preprocess(
            data: pd.DataFrame,  # emg 1~32
            mvc_data_list: list,  # list of dataframes
            train_sample: int,
            train_sample_per_label: int,
            window_sample: int,
            moving_sample: int,
            height_number=4,
            width_number=8,
            sampling_freq=2000,  # hz
            lowest_freq=30,
            highest_freq=250,
            bandpass_order=4,
            smoothing_window=10.0,  # ms
            mvc_limit=25.0,  # uV
            mvc_quality_value=0.001,
            quality_value=0.0015
    ):

        # MVC calculation
        mvc_value_list = Preprocess.mvc_calculation(
            data_list=mvc_data_list,
            mvc_limit=mvc_limit,
            mvc_quality_value=mvc_quality_value,
            height_number=height_number,
            width_number=width_number,
            sampling_freq=sampling_freq,  # hz
            lowest_freq=lowest_freq,
            highest_freq=highest_freq,
            bandpass_order=bandpass_order,
            smoothing_window=smoothing_window  # ms
            )
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
                freq_x, freq_y, notch_f = Preprocess.fast_fourier_transform(
                    emg_data=each_window_data,
                    notch_limit=2.5,
                    sampling_freq=sampling_freq
                    )
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
                for notch_freq in notch_f:
                    each_window_data = Preprocess.notch_filter(
                        data=each_window_data,
                        quality_value=quality_value,
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
                    mvc_flag=True,
                    mvc_value=mvc_value_list[emg_index]
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
            mvc_data_list: list,  # list of dataframes
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
            lowest_freq=30,
            highest_freq=250,
            bandpass_order=4,
            smoothing_window=10.0,  # ms
            mvc_limit=25.0,  # uV
            mvc_quality_value=0.001,
            quality_value=0.0015
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
            mvc_data_list=mvc_data_list,
            train_sample=train_sample,
            train_sample_per_label=train_sample_per_label,
            window_sample=window_sample,
            moving_sample=moving_sample,
            height_number=emg_height_number,
            width_number=emg_width_number,
            sampling_freq=sampling_freq,
            lowest_freq=lowest_freq,
            highest_freq=highest_freq,
            bandpass_order=bandpass_order,
            smoothing_window=smoothing_window,
            mvc_limit=mvc_limit,
            mvc_quality_value=mvc_quality_value,
            quality_value=quality_value
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


@torch.fx.wrap
def torch_randn(shape):
    return torch.randn(shape)


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
                 p: int,  # 10
                 input_dim: int,  # 20*20
                 model_dim: int,  # D = 200 
                 n_patches: int,  # N = 8
                 dropout_p: float):
        # projected_dim = N * D
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.n_patches = n_patches
        self.dropout_p = dropout_p
        # self.to_image_embedding = nn.Sequential(
        #     # p1=20, p2=10, N1=4, N2=4
        #     # in_channel: 3, emb_size: 3*p1*p2=600
        #     # kernel size=(p1, p2)=(20, 10)
        #     # stride=(p1, p2)=(20, 10)
        #     nn.Conv2d(3, 600, kernel_size=(20, 10), stride=(20, 10)),
        #     Rearrange('b e (h) (w) -> b (h w) e'),  # B * 16 * 600
        #     )
        self.to_patch_embedding = nn.Sequential(
            OrderedDict(
                {
                    # B(=16) * (8+1) * 20 * 20
                    # B * 1 * 20 * 20
                    "rearrange": Rearrange(
                        'b n (h1 p1) (w1 p2) -> b n (h1 w1 p1 p2)',
                        p1=p, p2=p
                    ),
                    "projection": nn.Linear(self.input_dim, self.model_dim)
                })
        )
        # initialize by strain data
        # self.to_pos_embedding = mySequential(
        #     OrderedDict(
        #         {
        #             "rearrange": Rearrange(
        #                 'b n (h1 p1) (w1 p2) -> b n (h1 w1 p1 p2)',
        #                 p1=1, p2=1
        #             ),
        #             "projection": nn.Linear(
        #                 8, self.model_dim)
        #         })
        # )
        self.cls_token = nn.Parameter(
            torch.randn((1, 1, self.model_dim))
            )  # 1 * (1 * D)
        self.pos_emb = nn.Parameter(
            torch.randn(1, self.n_patches + 1, self.model_dim)
            )  # 1 * (N+1) * D
        # self.pos_token = nn.Parameter(
        #     torch_randn((1, 1, self.model_dim))
        # )  # 1 * (1 * D)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, sensor_input):  # emg_input, strain_input
        # input_projection = self.to_image_embedding(sensor_input)  # b * N * D
        input_projection = self.to_patch_embedding(sensor_input)  # b * N * D
        # strain_projection = self.to_pos_embedding(strain_input)  # b * N * D
        b, _, _ = input_projection.shape
        cls_token = self.cls_token.repeat(b, 1, 1)  # b * 1 * D
        # pos_token = self.pos_token.repeat(b, 1, 1)  # b * 1 * D
        # pos_emb = nn.Parameter(
        #     torch.cat((pos_token, strain_projection), dim=1)
        # )  # b * (N+1) * D
        patch_emb = torch.cat(
            (cls_token, input_projection), dim=1
        )  # b * (N+1) * D

        return self.dropout(self.pos_emb + patch_emb)


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
            if self.drop_hidden else nn.Sequential(
                    nn.Linear(self.model_dim, self.model_dim),
                    nn.Dropout(self.dropout_p)
                    )
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.softmax = nn.Softmax(dim=-1)

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
        # att = F.softmax(qk_att / self.scaling, dim=-1)
        scaled = qk_att/self.scaling
        att = self.softmax(scaled)

        if self.drop_hidden:
            # att = F.dropout(att, p=self.dropout_p)
            att = self.dropout(att)

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

        self.block = nn.Sequential(
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
                nn.Sequential(
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
        self.encoder = nn.Sequential(*layers)

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

        # self.input_dim = input_dim
        self.model_dim = model_dim
        self.n_output = n_output
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.training_phase = training_phase

        assert pool in {'cls', 'mean'},\
            'pool type must be either cls (cls token) or mean (mean pooling)'

        self.pool = pool

        # self.norm = nn.LayerNorm(self.model_dim)
        self.hidden1 = nn.Linear(self.model_dim, self.hidden1_dim)
        self.hidden2 = nn.Linear(self.hidden1_dim, self.hidden2_dim)
        self.hidden = nn.Linear(self.hidden2_dim, self.n_output)
        self.block = nn.Sequential(
            self.hidden1, self.hidden2, self.hidden
            )

        # # self.strain_norm = nn.LayerNorm(self.input_dim)
        # self.strain_embedding = nn.Linear(self.input_dim, self.model_dim)
        # self.strain_hidden = nn.Linear(self.model_dim, self.hidden2_dim)
        # self.strain_block = nn.Sequential(
        #     self.strain_embedding, self.strain_hidden
        #     )

        # # self.output_norm = nn.LayerNorm(2*self.hidden2_dim)
        # self.hidden = nn.Linear(2*self.hidden2_dim, self.n_output)
        # # self.output_block = nn.Sequential(
        # #     self.output_norm, self.hidden
        # #     )

    def forward(self, encoder_output):  # encoder_output, strain_input
        y = encoder_output.mean(dim=1)\
            if self.pool == 'mean' else encoder_output[:, 0]
        # emg_out = self.block(y)

        # strain_out = self.strain_block(strain_input)

        # return self.hidden(torch.cat([emg_out, strain_out], dim=1))
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
        input_dim = (p**2)*1

        self.vit = nn.Sequential(
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
        # self.regression = RegressionHead(
        #     input_dim,
        #     model_dim, n_output, hidden1_dim,
        #     hidden2_dim, training_phase, pool
        #     )

    def forward(self, sensor_input):
        # emg_input = inputs[0]
        # strain_input = inputs[1]
        # return self.regression(self.vit(emg_input), strain_input)
        return self.vit(sensor_input)


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
        model, train_loader, optimizer, scheduler, epoch, criterion, device,
        opt, wandb_set=False
        ):
    model.train()
    train_loss = 0.0
    print('Training start.')
    total_time = 0

    tqdm_bar = tqdm(enumerate(train_loader))

    for batch_idx, (sensor_input, label) in tqdm_bar:
        start = time.time()
        sensor_input = sensor_input.to(device)

        label = label.to(torch.float32)
        label = label.to(device)

        optimizer.zero_grad()
        # inputs = [emg, strain]
        output = model(sensor_input)
        loss = criterion(output, label)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
        # scheduler.step()
        tqdm_bar.set_description(
            "Epoch {} batch {} - train loss: {:.6f}".format(
                epoch, (batch_idx), loss.item()
                )
            )
        total_time += time.time() - start
        if wandb_set == 1:
            wandb.watch(model, criterion, 'all')
            wandb.log({"Inference time": total_time})
            # wandb.log(
            #     {
            #         "Learning Rate": scheduler.get_lr()
            #     }
            # )
            # if (opt['log_interval'] > 0) and\
            #         ((batch_idx + 1) % opt['log_interval'] == 0):
            #     wandb.log(
            #         {
            #             "Training Loss": round(
            #                 train_loss / opt['log_interval'], 6
            #                 )
            #         }
            #     )

    scheduler.step(train_loss)
    # scheduler.update()
    train_loss /= len(train_loader)

    return train_loss


def evaluate(
        model, test_loader, criterion, device,
        opt, wandb_set=False
        ):
    model.eval()
    test_loss = 0.0

    tqdm_bar = tqdm(enumerate(test_loader))

    with torch.no_grad():
        for batch_idx, (sensor_input, label) in tqdm_bar:
            sensor_input = sensor_input.to(device)
            label = label.to(torch.float32)

            # print("label")
            # print(label)
            label = label.to(device)

            # inputs = [emg, strain]
            # print("input")
            # print(emg.shape)
            # print(strain.shape)
            output = model(sensor_input)
            # print("output")
            # print(output)

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


def sweep_evaluate(
        model, validation_loader, random_loader,
        criterion, device, opt, wandb_set=False
        ):
    model.eval()
    test_loss = 0.0
    validation_loss = 0.0
    random_loss = 0.0

    tqdm_validation_bar = tqdm(enumerate(validation_loader))
    tqdm_random_bar = tqdm(enumerate(random_loader))

    with torch.no_grad():
        for batch_idx, (sensor_input, label) in tqdm_validation_bar:
            sensor_input = sensor_input.to(device)
            label = label.to(torch.float32)
            label = label.to(device)

            output = model(sensor_input)
            
            loss = criterion(output, label)
            test_loss += loss.item()
            validation_loss += loss.item()

            tqdm_validation_bar.set_description(
                "Validation step: {} || Validation loss: {:.6f}".format(
                    (batch_idx + 1) / len(validation_loader), loss.item()
                    )
                )
        validation_loss /= len(validation_loader.dataset)
    with torch.no_grad():
        for batch_idx, (sensor_input, label) in tqdm_random_bar:
            sensor_input = sensor_input.to(device)
            label = label.to(torch.float32)
            label = label.to(device)

            output = model(sensor_input)
            
            loss = criterion(output, label)
            test_loss += loss.item()
            random_loss += loss.item()

            tqdm_random_bar.set_description(
                "Random step: {} || Random loss: {:.6f}".format(
                    (batch_idx + 1) / len(random_loader), loss.item()
                    )
                )
        random_loss /= len(random_loader.dataset)
        test_loss /= (len(validation_loader.dataset)+len(random_loader.dataset)) 
        if wandb_set == 1:
            wandb.log(
                {
                    'Test Loss':
                        round(test_loss, 6)
                }
            )

    return test_loss, validation_loss, random_loss


def print_evaluate(
        model, validation_loader, random_loader,
        criterion, device, opt, wandb_set=False
        ):
    model.eval()
    test_loss = 0.0
    validation_loss = 0.0
    random_loss = 0.0
    inference_time = 0.0
    total_num = 0
    history = {
        'validation_RMSE': [], 'random_RMSE': [],
        'validation_true': [], 'validation_prediction': [],
        'random_true': [], 'random_prediction': []}

    tqdm_validation_bar = tqdm(enumerate(validation_loader))
    tqdm_random_bar = tqdm(enumerate(random_loader))

    with torch.no_grad():
        for batch_idx, (sensor_input, label) in tqdm_validation_bar:
            sensor_input = sensor_input.to(device)
            label = label.to(torch.float32)
            label = label.to(device)

            val_start = time.time()
            output = model(sensor_input)
            val_end = time.time()
            inference_time += (val_end - val_start)
            total_num += 1
            
            history['validation_true'].append(label.item())
            history['validation_prediction'].append(output.item())
            
            loss = criterion(output, label)
            
            history['validation_RMSE'].append(loss.item())
            
            test_loss += loss.item()
            validation_loss += loss.item()

            tqdm_validation_bar.set_description(
                "Validation step: {} || Validation loss: {:.6f}".format(
                    (batch_idx + 1) / len(validation_loader), loss.item()
                    )
                )
        validation_loss /= len(validation_loader.dataset)
    with torch.no_grad():
        for batch_idx, (sensor_input, label) in tqdm_random_bar:
            sensor_input = sensor_input.to(device)
            label = label.to(torch.float32)
            label = label.to(device)

            random_start = time.time()
            output = model(sensor_input)
            random_end = time.time()
            inference_time += (random_end - random_start)
            total_num += 1
            
            history['random_true'].append(label.item())
            history['random_prediction'].append(output.item())
            
            loss = criterion(output, label)
            
            history['random_RMSE'].append(loss.item())
            
            test_loss += loss.item()
            random_loss += loss.item()

            tqdm_random_bar.set_description(
                "Random step: {} || Random loss: {:.6f}".format(
                    (batch_idx + 1) / len(random_loader), loss.item()
                    )
                )
        random_loss /= len(random_loader.dataset)
        test_loss /= (len(validation_loader.dataset)+len(random_loader.dataset))

    return test_loss, validation_loss, random_loss, history, inference_time, total_num


class CustomDataset(Dataset):
    def __init__(self, w_emg, w_strain, w_label):
        self.emg = w_emg
        self.strain = w_strain
        self.label = w_label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        emg = torch.Tensor(self.emg[idx])
        strain = torch.Tensor(self.strain[idx])
        label = torch.Tensor(self.label[idx])
        return emg, strain, label

    def collate_fn(self, data):
        batch_emg, batch_strain, batch_label = [], [], []
        for emg, strain, label in data:
            emg = torch.Tensor(emg)
            strain = torch.Tensor(strain)
            label = torch.Tensor(label)
            batch_emg.append(emg)
            batch_strain.append(strain)
            batch_label.append(label)
        batch_emg = torch.stack(batch_emg, dim=0).float()
        batch_strain = torch.stack(batch_strain, dim=0).float()
        batch_label = torch.stack(batch_label, dim=0).float()
        return batch_emg, batch_strain, batch_label


class CustomDataset_case1(Dataset):
    def __init__(self, w_input, w_label):
        self.input = w_input
        self.label = w_label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        sensor_input = torch.Tensor(self.input[idx])
        label = torch.Tensor(self.label[idx])
        return sensor_input, label

    def collate_fn(self, data):
        batch_input, batch_label = [], []
        for sensor_input, label in data:
            sensor_input = torch.Tensor(sensor_input)
            label = torch.Tensor(label)
            batch_input.append(sensor_input)
            batch_label.append(label)
        batch_input = torch.stack(batch_input, dim=0).float()
        batch_label = torch.stack(batch_label, dim=0).float()
        return batch_input, batch_label


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss.to(torch.float32)


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


def testset_prediction(
        model, test_loader, criterion, device,
        wandb_set=False
        ):
    model.eval()
    test_loss = 0.0
    test_loss_list = []

    tqdm_bar = tqdm(enumerate(test_loader))

    with torch.no_grad():
        for batch_idx, (emg, strain, label) in tqdm_bar:
            emg = emg.to(device)
            strain = strain.to(device)
            label = label.to(torch.float32)

            print("label")
            print(label)
            label = label.to(device)

            # inputs = [emg, strain]
            output = model(emg, strain)
            print("output")
            print(output)

            loss = criterion(output, label)
            test_loss += loss.item()
            test_loss_list.append(loss.item())
            tqdm_bar.set_description(
                "Test step: {} || Test loss: {:.6f}".format(
                    (batch_idx + 1) / len(test_loader), loss.item()
                    )
                )

    test_loss /= len(test_loader.dataset)
    print("number of test dataset")
    print(len(test_loader.dataset))
    if wandb_set == 1:
        wandb.log(
            {
                'Test Loss':
                    round(test_loss, 6)
            }
        )

    return test_loss, test_loss_list


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
        test_dataset, batch_size=1,
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

    final_test_loss, test_loss_list = testset_prediction(
        model, test_dataloader, criterion,
        device, wandb_set)

    # ViT regression model save
    if model_save_dir is not None:
        torch.save(model.state_dict(), model_save_dir)

    return test_loss, history, final_test_loss, test_loss_list


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
        mvc_data_list: list,  # list of dataframes
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
            mvc_data_list=mvc_data_list,
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
            lowest_freq=hyperparameter_defaults["lowest_freq"],
            highest_freq=hyperparameter_defaults["highest_freq"],
            bandpass_order=hyperparameter_defaults["bandpass_order"],
            smoothing_window=hyperparameter_defaults["smoothing_window"],
            mvc_limit=hyperparameter_defaults["mvc_limit"],
            mvc_quality_value=hyperparameter_defaults["mvc_quality_value"],
            quality_value=hyperparameter_defaults["quality_value"]
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

        test_loss, history, final_test_loss, test_loss_list =\
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

        test_loss, history, final_test_loss, test_loss_list =\
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

    # test loss for logging
    # predictor = DataPredictor(
    #     test_dataset,
    #     model_save_dir,
    #     hyperparameter_defaults,
    #     device)
    # test_loss_list = predictor.prediction()
    plt.plot(test_loss_list)
    print("Test loss one by one")
    print("Maximum loss")
    print(np.max(np.abs(test_loss_list)))
    print("Mean loss")
    print(np.mean(np.abs(test_loss_list)))

################################################################
################################################################
# Main
################################################################
################################################################

# hyperparameter_defaults = {
#     # Training
#     'random_state': 42,
#     'data_division_num': 10.0,
#     'pool': 'mean',  # or 'cls'
#     'test_size': 0.35,
#     'batch_size': 16,
#     'epochs': 300,
#     'learning_rate': 0.001,
#     # wandb & logging
#     'prj_name': "HD_ViT",
#     'log_interval': 5,
#     # Model
#     'patch_size': 2,  # p
#     'model_dim': 32,
#     'hidden_dim': 64,
#     'hidden1_dim': 16,
#     'hidden2_dim': 4,
#     'n_output': 1,
#     'n_heads': 8,
#     'n_layers': 10,
#     'dropout_p': 0.2,
#     'model_save_dir': "./ViT_model/2310012_test1_ViT.pt",
#     # Scheduler
#     'n_warmup_steps': 10,
#     'decay_rate': 0.99,
#     # Strain calibration
#     'polynomial_order': 2,
#     # preprocessing parameters
#     "emg_height_number": 2,  # 4
#     "emg_width_number": 4,  # 8
#     "strain_height_number": 1,
#     "strain_width_number": 1,
#     "sampling_freq": 2000,  # hz
#     "window_size": 200,  # ms
#     "overlapping_ratio": 0.75,
#     "time_advance": 100,  # ms
#     "label_half_size": 5,  # ms
#     "lowest_freq": 30,
#     "highest_freq": 250,
#     "bandpass_order": 4,
#     "smoothing_window": 10.0,  # ms
#     "data_list": [1, 2, 3, 4, 5],
#     "data_valley_index_dict": {
#         1: [[0, -7], [-5, -1]],
#         2: [[1, -2]],
#         3: [[1, -1]],
#         4: [[0, -2]],
#         5: [[0, -1]]
#         },
#     "mvc_data_list": [1, 2, 3],
#     "mvc_limit": 50.0,  # uV
#     "mvc_quality_value": 0.001,
#     "quality_value": 0.0015
#     }

# if __name__ == "__main__":

#     model_save_dir = "./ViT_model/231021_trial1_2(231013)_ViT_LR%s.pt" %\
#         str(hyperparameter_defaults["learning_rate"])
#     history_title = "231021_trial1_2(231013)_ViT_LR_%s" %\
#         str(hyperparameter_defaults["learning_rate"])

#     #############################
#     # Path Reading
#     #############################
#     mvc_path = './Test_data/1013/1013_HSJ_312_test_MVC_'
#     mvc_sheet = ""

#     main_path = "./Test_data/1013/1013_HSJ_312_test_set"
#     main_sheet = ""

#     strain_calib_txt_path =\
#         './Test_data/1013/1013_HSJ_314_strain_calibration.txt'
#     mocap_calib_csv_path = './Test_data/1013/HSJ_Mocap_after_EMG.csv'
#     mocap_column_list = [
#         "frame", "time",
#         "biceps_x", "biceps_y", "biceps_z",
#         "elbow_x", "elbow_y", "elbow_z",
#         "forearm_x", "forearm_y", "forearm_z",
#         "shoulder_x", "shoulder_y", "shoulder_z",
#         "wrist_lateral_x", "wrist_lateral_y", "wrist_lateral_z",
#         "wrist_medial_x", "wrist_medial_y", "wrist_medial_z"]
#     mocap_index_list = [
#         [400, 18500],
#         [20000, 23500],
#         [27500, 41200],
#         [53000, 61800],
#         [71800, 81200],
#         [87800, 90200],
#         [91100, 98400],
#         [104200, 109800]
#         ]
#     #############################
#     # Data Reading
#     #############################
#     # data list for emg mvc
#     mvc_list = list()
#     for mvc_ind in hyperparameter_defaults["mvc_data_list"]:
#         each_mvc_path = mvc_path + str(mvc_ind) + '.txt'
#         mvc_data_list = emg_data_txt_reading_231011(
#              each_mvc_path, mvc_sheet
#              )
#         mvc_df = pd.DataFrame()
#         for i in np.arange(len(mvc_data_list) - 1):
#             mvc_df["emg" + str(i + 1)] = mvc_data_list[i]
#         mvc_df.reset_index(drop=True, inplace=True)
#         mvc_list.append(mvc_df)  # list(dataframe)
#     #############################
#     # data for emg + strain
#     main_df = pd.DataFrame()
#     for main_ind in hyperparameter_defaults["data_list"]:
#         each_main_path = main_path + str(main_ind) + '.txt'
#         main_data_list = emg_data_txt_reading_231011(
#             each_main_path, main_sheet
#             )

#         # data indexing (peak)
#         peaks, _ = signal.find_peaks(
#             main_data_list[-1], prominence=300.0
#             )
#         new_peak = []
#         for peak_ind, peak in enumerate(peaks):
#             if peak_ind == 0:
#                 new_peak.append(peak)
#                 continue
#             if peak - new_peak[-1] <= 2000:
#                 continue
#             new_peak.append(peak)
#         # data indexing (valley)
#         valleys, _ = signal.find_peaks(
#             (-1) * main_data_list[-1], prominence=100.0
#             )
#         new_valley = []
#         for valley_ind, valley in enumerate(valleys):
#             if valley_ind == 0:
#                 new_valley.append(valley)
#                 continue
#             if valley - new_valley[-1] <= 2000:
#                 continue
#             new_valley.append(valley)

#         # Append main_df
#         for append_num in np.arange(
#                 len(hyperparameter_defaults["data_valley_index_dict"]
#                     [main_ind])
#                 ):
#             start_index = hyperparameter_defaults["data_valley_index_dict"]\
#                 [main_ind][append_num][0]
#             end_index = hyperparameter_defaults["data_valley_index_dict"]\
#                 [main_ind][append_num][1]

#             if len(main_df) == 0:
#                 for i in np.arange(len(main_data_list) - 1):
#                     main_df["emg" + str(i + 1)] =\
#                         main_data_list[i][start_index:end_index + 1]
#                 main_df["strain"] =\
#                     main_data_list[-1][start_index:end_index + 1]
#             else:
#                 for i in np.arange(len(main_data_list) - 1):
#                     main_df["emg" + str(i + 1)] =\
#                         main_df["emg" + str(i + 1)].append(
#                             pd.Series(
#                                 main_data_list[i][start_index:end_index + 1]
#                                 ),
#                             ignore_index=True
#                             )
#                 main_df["strain"] = main_df["strain"].append(
#                     pd.Series(
#                         main_data_list[-1][start_index:end_index + 1]
#                         ),
#                     ignore_index=True
#                     )
#     #############################
#     # data for strain & angle (mocap) -- strain
#     calib_df = pd.DataFrame()
#     strain_calib = strain_data_txt_reading(
#         path=strain_calib_txt_path,
#         sheet_name="")
#     # time [ms]
#     strain_calib_time = np.linspace(
#         0,
#         (len(strain_calib) - 1) *
#         (1000.0 / hyperparameter_defaults["sampling_freq"]),
#         len(strain_calib)
#         )
#     #############################
#     # By index
#     # part 1: 7000:310000
#     # part 2: 458000:688000
#     # part 3: 884000:1030000
#     # part 4: 1196000:1354000
#     # part 5: 1465000:1504000
#     # part 6: 1518000:1641000
#     # part 7: 1736500:1825000

#     #############################
#     # data for strain & angle (mocap) -- mocap
#     mocap_calib = mocap_data_csv_reading(
#         mocap_calib_csv_path,
#         mocap_column_list,
#         angle_option=2  # biceps-elbow-forearm
#         )
#     mocap_calib = mocap_calib[
#         mocap_calib.time <= (strain_calib_time[-1] * 0.001)
#         ]

#     strain_calib_interp = strain_mocap_interpolation(
#         strain_data=strain_calib,
#         mocap_data=mocap_calib,
#         strain_freq=hyperparameter_defaults["sampling_freq"]
#         )
#     #############################
#     # By index
#     # part 1: 400:18500
#     # part 2: 20000:23500
#     # part 3: 27500:41200
#     # part 4: 53000:61800
#     # part 5: 71800:81200
#     # part 6: 87800:90200
#     # part 7: 91100:98400
#     # part 8: 104200:109800

#     # Final indexing
#     final_strain_data = []
#     final_angle_data = []
#     for final_ind in np.arange(len(mocap_index_list)):
#         start_ind = mocap_index_list[final_ind][0]
#         end_ind = mocap_index_list[final_ind][1]

#         final_strain_data.extend(strain_calib_interp
#                                  [start_ind:end_ind + 1].values)

#         final_angle_data.extend(
#             np.array(mocap_calib.angle)[start_ind:end_ind + 1]
#             )
#         # plt.plot(mocap_calib_interp[start_ind:end_ind + 1])
#     calib_df["strain"] = final_strain_data
#     calib_df["angle"] = final_angle_data
#     #############################
#     # data split
#     main_emg_data = main_df   # dataframe (32 columns)
#     main_strain_data = main_df["strain"]  # Series
#     calib_strain_data = np.array(calib_df["strain"])  # np.array
#     calib_angle_data = np.array(calib_df["angle"])  # np.array
#     # plt.plot(calib_angle_data)
#     #############################
#     # main
#     main(
#         wandb_set=False,
#         hyperparameter_defaults=hyperparameter_defaults,
#         model_save_dir=model_save_dir,
#         history_title=history_title,
#         main_emg_data=main_emg_data,
#         mvc_data_list=mvc_list,
#         main_strain_data=main_strain_data,
#         calib_strain_data=calib_strain_data,
#         calib_angle_data=calib_angle_data
#         )
