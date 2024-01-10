# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 20:20:36 2023

@author: mleem
"""

# import scipy.io as sio
# import mat73
# import h5py
# import mat4py
import os
import math
from collections import Counter
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
# from sklearn.model_selection import StratifiedShuffleSplit

# import torch
# from torchvision import transforms
# from torch.utils.data import Subset, Dataset, DataLoader, WeightedRandomSampler

from DELSYS_model import *
from DELSYS_model_case2 import *
from DELSYS_config import *


def draw_stft(f, t, Zxx):
    plt.figure(figsize=(12,5))
    # plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=1, shading='gouraud')
    # plt.title('STFT Magnitude')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


# Short Time Fourier Transform
def calc_stft(data, Fs, nperseg):
    f, t, Zxx = signal.stft(data, Fs, nperseg=nperseg)
    draw_stft(f, t, Zxx)


def notch_filter(data, f, f_interval, Fs, normalized=False):
    f = f/(Fs/2) if normalized else f
    b, a = signal.iirnotch(f, f/f_interval, Fs)
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data


def bandpass_filter(data, low_f, high_f, order, Fs, normalized=False):
    low_f = low_f/(Fs/2) if normalized else low_f
    high_f = high_f/(Fs/2) if normalized else high_f
    b, a = signal.butter(order, [low_f, high_f], btype='bandpass')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


def moving_RMS(data, window, Fs):
    smoothed_data = data.pow(2).rolling(window=round(
        window/(1/Fs)), min_periods=1).mean().apply(np.sqrt, raw=True)
    return smoothed_data


def basic_interpolation(standard_data, target_data, target_columns):
    time_synced = standard_data["time"]
    time_target = target_data["time"]
    target = target_data.loc[:, target_columns]
    target_col = target.columns
    if len(target_data) != 1:
        target_synced = pd.DataFrame(columns = target_col)
        for t in target_col:
            temp_data = target[str(t)]
            tmp = np.interp(time_synced,
                            time_target, temp_data)
            target_synced[str(t)] = tmp
    else:
        target_synced = pd.DataFrame(
            np.zeros(target_synced.shape[0], len(target_col)),
            columns = target_col)
    return target_synced


def HSJ_preprocessing():
    EMG_path = "./DELSYS/230627/EMG_HSJ.csv"
    mocap_path = "./DELSYS/230627/mocap_HSJ-180.csv"
    strain_path = "./DELSYS/230627/Strain_HSJ.txt"
    # read EMG data
    df_EMG = pd.read_csv(EMG_path, delimiter=",", header=0)
    df_EMG.columns = ["time", "EMG"]
    # read mocap data
    df_mocap = pd.read_csv(mocap_path, delimiter=",", header=0)
    df_mocap.columns = ["frame", "time", "M1_X", "M1_Y", "M1_Z", "M2_X",
                        "M2_Y", "M2_Z", "M3_X", "M3_Y", "M3_Z", "angle_raw",
                        "angle"]
    # read strain data
    df_strain = pd.read_table(strain_path, sep=" ", header=None)
    df_strain.drop([0], axis=1, inplace=True)
    df_strain.columns = ["strain"]
    strain_index = list(signal.find_peaks((-1)*df_strain.strain,
                                              prominence=400.0)[0][:])
    del strain_index[10]    
    strain_start_index = strain_index[2]
    strain_end_index = strain_index[-1]

    # EMG start sync
    EMG_start_index = signal.find_peaks(
        df_EMG.loc[:10000, "EMG"], height=1.5*np.power(0.1, 5))[0][0]
    EMG_start_time = df_EMG.loc[EMG_start_index, "time"]
    df_EMG.time -= EMG_start_time
    df_EMG = df_EMG[df_EMG.time >= 0.0]
    df_EMG.reset_index(drop=True, inplace=True)
    
    # mocap angle end sync
    mocap_end_index = signal.find_peaks((-1)*df_mocap.loc[7800:, "angle"],
                                        prominence=0.75)[0][0]
    mocap_end_index += 7800
    mocap_end_time = df_mocap.loc[mocap_end_index, "time"]

    # sync
    df_EMG = df_EMG[df_EMG.time <= mocap_end_time]
    df_EMG.reset_index(drop=True, inplace=True)
    df_mocap = df_mocap[df_mocap.time <= mocap_end_time]
    df_mocap.reset_index(drop=True, inplace=True)
    df_strain = df_strain.loc[:int(strain_end_index), :]
    # strain data time update
    strain_start_time = 0.0
    strain_end_time = df_mocap.at[len(df_mocap) - 1, "time"]
    df_strain["time"] = np.linspace(strain_start_time, strain_end_time,
                                    len(df_strain))
    df_strain = df_strain[["time", "strain"]]

    # plotting
    fig = plt.figure(figsize=(12,5))
    ax = fig.gca()
    bx = ax.twinx()
    ax.plot(df_EMG.time, df_EMG.EMG, label="EMG", c="blue")
    bx.plot(df_mocap.time, df_mocap.angle, label="angle", c="darkorange")
    ax.set_ylabel("Voltage [mV]")
    bx.set_ylabel("Angle [deg]")
    ax.set_xlabel("Time [sec]")
    ax.legend()
    bx.legend()
    ax.set_title("Synced data_EMG & angle")

    # plotting
    fig = plt.figure(figsize=(12,5))
    ax = fig.gca()
    bx = ax.twinx()
    ax.plot(df_strain.time, df_strain.strain, label="strain", c="lightgreen")
    bx.plot(df_mocap.time, df_mocap.angle, label="angle", c="darkorange")
    ax.set_ylabel("Sensor value")
    bx.set_ylabel("Angle [deg]")
    ax.set_xlabel("Time [sec]")
    ax.legend()
    bx.legend()
    ax.set_title("Synced data_strain sensor & angle")

    # df for MVC
    MVC_end_index = signal.find_peaks(
        df_EMG.loc[60000:65000, "EMG"], height=3*np.power(0.1, 5))[0][0]
    MVC_end_index += 60000
    df_MVC = df_EMG.iloc[:MVC_end_index, :]

    # angle start update
    angle_start_index = signal.find_peaks((-1)*df_mocap.loc[3500:3800, "angle"],
                                        prominence=0.1)[0][0]
    angle_start_index += 3500
    angle_start_time = df_mocap.loc[angle_start_index, "time"]
    # df EMG start update
    df_EMG = df_EMG[df_EMG.time >= angle_start_time]
    df_EMG.time -= df_EMG.iloc[0, 0]
    df_EMG.reset_index(drop=True, inplace=True)
    # df mocap start update
    df_mocap = df_mocap[df_mocap.time >= angle_start_time]
    df_mocap.time -= angle_start_time
    df_mocap.reset_index(drop=True, inplace=True)
    # df strain start update
    df_strain = df_strain[df_strain.time >= angle_start_time]
    df_strain.time -= df_strain.iloc[0, 0]
    df_strain.reset_index(drop=True, inplace=True)

    # FFT
    ## sampling rate 2148 Hz
    Fs = 2148
    n = len(df_EMG) # total data length
    k = np.arange(n) 
    T = n/Fs
    freq = k/T
    freq = freq[range(int(n/2))]

    freq_EMG = np.fft.fft(list(df_EMG["EMG"]))/n
    freq_EMG = freq_EMG[range(int(n/2))]
    
    # freq plotting
    plt.figure(figsize=(12,5))
    plt.plot(freq, abs(freq_EMG))
    plt.title("EMG_HSJ_FFT")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Intensity")
    plt.xlim([0, 300])
    plt.show()

    # short time fourier transform
    # calc_stft(list(df_EMG["EMG"]), Fs, 100)
    
    # notch filter
    notch_f = 50

    # Bandpass filter
    high_f = 250
    low_f = 30

    # Remove mean
    EMG_mean_ori = np.mean(df_EMG.EMG)
    df_EMG.EMG -= EMG_mean_ori
    # for MVC
    EMG_mean_MVC = np.mean(df_MVC.EMG)
    df_MVC.EMG -= EMG_mean_MVC

    # plotting
    plt.figure(figsize=(12,5))
    plt.plot(df_EMG.time, df_EMG.EMG)
    plt.title("EMG_HSJ_remove mean")
    plt.xlabel("time [sec]")
    plt.ylabel("Voltage [mV]")
    plt.show()

    # Notch filter
    print("notch")
    print(df_EMG.EMG.isnull().sum())
    df_EMG.EMG = notch_filter(list(df_EMG.EMG),
                              notch_f, f_interval=20,
                              Fs=Fs,
                              normalized=True)
    print(df_EMG.EMG.isnull().sum())
    df_MVC.EMG = notch_filter(list(df_MVC.EMG), notch_f, f_interval=20, Fs=Fs,
                              normalized=True)

    # Bandpass filter
    print("bandpass")
    print(df_EMG.EMG.isnull().sum())
    df_EMG.EMG = bandpass_filter(list(df_EMG.EMG),
                                 low_f, high_f, order=4,
                                  Fs=Fs, normalized=True)
    print(df_EMG.EMG.isnull().sum())
    df_MVC.EMG = bandpass_filter(list(df_MVC.EMG), low_f, high_f, order=4,
                                  Fs=Fs, normalized=True)

    # plotting
    plt.figure(figsize=(12,5))
    plt.plot(df_EMG.time, df_EMG.EMG)
    plt.title("EMG_HSJ_bandpass_filter")
    plt.xlabel("time [sec]")
    plt.ylabel("Voltage [mV]")
    plt.show()

    # Absolute value
    df_EMG.EMG = abs(df_EMG.EMG)
    df_MVC.EMG = abs(df_MVC.EMG)
    # plotting
    plt.figure(figsize=(12,5))
    plt.plot(df_EMG.time, df_EMG.EMG)
    plt.title("EMG_HSJ_Rectified")
    plt.xlabel("time [sec]")
    plt.ylabel("Voltage [mV]")
    plt.show()

    # Moving RMS smoothing
    # window size: 50 ms
    RMS_window = 0.05
    print("RMS")
    print(df_EMG.EMG.isnull().sum())
    df_EMG.EMG = moving_RMS(df_EMG["EMG"], RMS_window, Fs=Fs)
    print(df_EMG.EMG.isnull().sum())
    df_MVC.EMG = moving_RMS(df_MVC["EMG"], RMS_window, Fs=Fs)
    # plotting
    plt.figure(figsize=(12,5))
    plt.plot(df_EMG.time, df_EMG.EMG)
    plt.title("EMG_HSJ_Smoothing")
    plt.xlabel("time [sec]")
    plt.ylabel("Voltage [mV]")
    plt.show()

    # MVC calculation
    # plotting
    plt.figure(figsize=(12,5))
    plt.plot(df_MVC.time, df_MVC.EMG)
    plt.title("EMG_MVC_HSJ_Smoothing")
    plt.xlabel("time [sec]")
    plt.ylabel("Voltage [mV]")
    plt.show()
    # Average 3 MVC
    # MVC1
    # df_MVC.loc[663:1687, "EMG"]
    MVC1 = np.mean(df_MVC.loc[663:1687, "EMG"])
    # plotting
    plt.figure(figsize=(12,5))
    plt.plot(df_MVC.loc[663:1687, "time"], df_MVC.loc[663:1687, "EMG"])
    plt.title("EMG_MVC1_HSJ_Smoothing")
    plt.xlabel("time [sec]")
    plt.ylabel("Voltage [mV]")
    plt.show()
    # MVC2
    # df_MVC.loc[21150:21957, "EMG"]
    MVC2 = np.mean(df_MVC.loc[21400:21600, "EMG"])
    # plotting
    plt.figure(figsize=(12,5))
    plt.plot(df_MVC.loc[21400:21600, "time"], df_MVC.loc[21400:21600, "EMG"])
    plt.title("EMG_MVC2_HSJ_Smoothing")
    plt.xlabel("time [sec]")
    plt.ylabel("Voltage [mV]")
    plt.show()
    # MVC3
    # df_MVC.loc[42050:43780,"EMG"]
    MVC3 = np.mean(df_MVC.loc[42050:42995,"EMG"])
    # plotting
    plt.figure(figsize=(12,5))
    plt.plot(df_MVC.loc[42050:42995, "time"], df_MVC.loc[42050:42995, "EMG"])
    plt.title("EMG_MVC3_HSJ_Smoothing")
    plt.xlabel("time [sec]")
    plt.ylabel("Voltage [mV]")
    plt.show()
    # total MVC
    MVC_total = np.mean([MVC1, MVC2, MVC3])
    print(MVC_total)

    # Normalization
    df_EMG.EMG = df_EMG.EMG * 100 / MVC_total
    # plotting
    plt.figure(figsize=(12,5))
    plt.plot(df_EMG.time, df_EMG.EMG)
    plt.title("EMG_HSJ_Normalization")
    plt.xlabel("time [sec]")
    plt.ylabel("%MVC [%]")
    plt.show()

    # interpolation
    df_HSJ = df_EMG
    df_HSJ["angle"] = basic_interpolation(df_EMG, df_mocap, ["angle"])
    df_HSJ["strain"] = basic_interpolation(df_EMG, df_strain, ["strain"])
    # df save
    df_HSJ.to_csv("./DELSYS/230627/processed_HSJ.csv", sep=",",
                  header=True, index=False)


def DELSYS_emg_preprocessing():

    path = "./DELSYS/230627/processed_HSJ.csv"
    # load data
    data_1 = pd.read_csv(path, delimiter=",", header=0)    
    #######################################################################
    ####### DATA DESCRIPTIONS #############################################
    #######################################################################
    # The number of data: 77638 (HSJ 기준)
    # Do not reflect outlier processing!!
    #######################################################################
    # Angle label
    label_1 = torch.Tensor(data_1['angle'])
    emg_1 = torch.Tensor(data_1['EMG'])
    ########################################################################
    # Sliding Window Normalization
    # Window size: 200 ms, 429 samples
    # Overlapping size: 150 ms, 214 samples
    # Time advance: 100ms, 107 samples
    ######################################
    ######################################
    # PARAMETERS
    sampling_freq = 2148
    window_size = 200 # ms
    overlapping_ratio = float(0.75)
    time_advance = 100 # ms
    total_sample = len(label_1)
    label_size = 2.5 # ms (200 Hz 기준)
    label_width_ratio = float(int(label_size)/int(time_advance))
    ######################################
    ######################################
    # implementation
    increment = float(1000/sampling_freq) # ms
    avg_sample = int(np.floor(float((2*label_size)/increment)))
    window_sample = int(np.floor(float(window_size/increment)))
    predict_sample = int(np.floor(float(time_advance/increment)))
    overlapping_sample = int(np.floor(float((1 - overlapping_ratio)*window_sample)))
    predict_num = int(np.floor(
        (total_sample - window_sample - predict_sample *
         (1 + label_width_ratio)) / overlapping_sample))
    total_length = int(window_sample + predict_sample *
                       (1 + label_width_ratio) +
                       overlapping_sample*predict_num)
    predict_angle_sample = total_length - overlapping_sample * predict_num -\
        window_sample
    #####################################
    # sliding window division
    # angle_label 
    label_1 = torch.index_select(label_1, 0, torch.from_numpy(
        np.arange(window_sample, total_length)))
    emg_1 = torch.index_select(emg_1, 0, torch.from_numpy(
        np.arange(0, total_length - predict_angle_sample)))
    # for loop - window division
    emg_1 = emg_1.unfold(0, window_sample, overlapping_sample)
    label_1 = label_1.unfold(0, predict_angle_sample, overlapping_sample)
    # unsqueeze - for dimension matching
    label_1 = label_1.unsqueeze(2)
    emg_1 = emg_1.unsqueeze(2)
    emg_1 = emg_1.unsqueeze(3)
    # permute for dimension matching
    emg_1 = emg_1.permute(1, 0, 2, 3)
    label_1 = label_1.permute(1, 0, 2)
    ######################################
    # window-wise, electrode-wise normalization (mean, std)
    # mean, std calculation 
    # normalize mean 0, std 1
    normalize_emg = transforms.Normalize(mean = torch.mean(
        emg_1, dim=0), std = torch.std(emg_1, dim=0))
    emg_1 = normalize_emg(emg_1)
    #######################################
    # Re-labelling (angle average)
    avg_label = label_1[-avg_sample:, :, :]
    avg_label = torch.mean(avg_label, dim=0)
    #######################################
    # Dataloader (train, test)
    emg_1 = emg_1.permute(1, 0, 2, 3)

    return window_sample, emg_1, avg_label


def DELSYS_emg_strain_preprocessing():

    path = "./DELSYS/230627/processed_HSJ.csv"
    # load data
    data_1 = pd.read_csv(path, delimiter=",", header=0)    
    #######################################################################
    ####### DATA DESCRIPTIONS #############################################
    #######################################################################
    # The number of data: 77638 (HSJ 기준)
    # Do not reflect outlier processing!!
    #######################################################################
    # Angle label
    label_1 = torch.from_numpy(data_1['angle'].values)
    emg_1 = torch.from_numpy(data_1['EMG'].values)
    strain_1 = torch.from_numpy(data_1["strain"].values)
    ########################################################################
    # Sliding Window Normalization
    # Window size: 200 ms, 429 samples
    # Overlapping size: 150 ms, 214 samples
    # Time advance: 100ms, 107 samples
    ######################################
    ######################################
    # PARAMETERS
    sampling_freq = 2148
    window_size = 200 # ms
    strain_window_size = 200 # ms
    overlapping_ratio = float(0.75)
    time_advance = 100 # ms
    strain_time_advance = 150 # ms
    total_sample = len(label_1)
    label_size = 2.5 # ms (200 Hz 기준)
    label_width_ratio = float(int(label_size)/int(time_advance))
    ######################################
    ######################################
    # implementation
    increment = float(1000/sampling_freq) # ms
    avg_sample = int(np.floor(float((2*label_size)/increment)))
    window_sample = int(np.floor(float(window_size/increment)))
    strain_window_sample = int(np.floor(float(strain_window_size/increment)))
    predict_sample = int(np.floor(float(time_advance/increment)))
    strain_advance_sample = int(np.floor(float(strain_time_advance/increment)))
    overlapping_sample = int(np.floor(float((1 - overlapping_ratio)*window_sample)))
    predict_num = int(np.floor(
        (total_sample - window_sample - predict_sample *
         (1 + label_width_ratio)) / overlapping_sample))
    total_length = int(window_sample + predict_sample *
                       (1 + label_width_ratio) +
                       overlapping_sample*predict_num)
    predict_angle_sample = total_length - overlapping_sample * predict_num -\
        window_sample
    #####################################
    # sliding window division
    # angle_label 
    label_1 = torch.index_select(label_1, 0, torch.from_numpy(
        np.arange(window_sample, total_length)))
    strain_1 = torch.index_select(strain_1, 0, torch.from_numpy(
        np.arange(strain_advance_sample, total_length - predict_angle_sample)))
    emg_1 = torch.index_select(emg_1, 0, torch.from_numpy(
        np.arange(0, total_length - predict_angle_sample)))
    # for loop - window division
    emg_1 = emg_1.unfold(0, window_sample, overlapping_sample)
    label_1 = label_1.unfold(0, predict_angle_sample, overlapping_sample)
    strain_1 = strain_1.unfold(0, strain_window_sample, overlapping_sample)
    # unsqueeze - for dimension matching # 여기부터
    label_1 = label_1.unsqueeze(2)
    emg_1 = emg_1.unsqueeze(2)
    emg_1 = emg_1.unsqueeze(3)
    strain_1 = strain_1.unsqueeze(2)
    strain_1 = strain_1.unsqueeze(3)
    # permute for dimension matching
    emg_1 = emg_1.permute(1, 0, 2, 3)
    strain_1 = strain_1.permute(1, 0, 2, 3)
    label_1 = label_1.permute(1, 0, 2)
    ######################################
    # window-wise, electrode-wise normalization (mean, std)
    # mean, std calculation 
    # normalize mean 0, std 1
    normalize_emg = transforms.Normalize(mean = torch.mean(
        emg_1, dim=0), std = torch.std(emg_1, dim=0))
    #### Strain normalize 여부 고민!!!!
    normalize_strain = transforms.Normalize(mean = torch.mean(
        strain_1, dim=0), std = torch.std(strain_1, dim=0))
    emg_1 = normalize_emg(emg_1)
    strain_1 = normalize_strain(strain_1)
    #######################################
    # Re-labelling (angle average)
    avg_label = label_1[-avg_sample:, :, :]
    avg_label = torch.mean(avg_label, dim=0)
    #######################################
    # Dataloader (train, test)
    emg_1 = emg_1.permute(1, 0, 2, 3)
    strain_1 = strain_1.permute(1, 0, 2, 3)

    return window_sample, strain_window_sample, emg_1, strain_1, avg_label


# if __name__ == "__main__":
#     HSJ_preprocessing()
    # window_sample, emg, angle = DELSYS_emg_preprocessing()
    # window_sample, strain_window_sample, emg, strain, angle =\
    #     DELSYS_emg_strain_preprocessing()
