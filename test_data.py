# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 23:36:10 2023

@author: mleem
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from final_function import *


def emg_main(txt_path: str,
             graph_path: str,
             test_label: str,
             frequency: float,
             low_f: int,
             high_f: int,
             notch_f: list,
             bandpass_order: int,
             smoothing_window: float,
             sheet_name='Sheet1'
             ):
    if test_label[:4] == '1011':
        data_list = emg_data_txt_reading_231011(
            txt_path, sheet_name=sheet_name
            )
    else:  # 1012 data
        data_list = emg_data_txt_reading_231012(
            txt_path, sheet_name=sheet_name
            )

    # Beforce preprocessing
    plot_raw_emg(
        data_list,
        frequency,
        "Before preprocessing",
        graph_path,
        test_label
        )

    # Preprocessing
    data_processed_list = emg_preprocess(
        emg_list=data_list,
        frequency=frequency,
        low_f=low_f,
        high_f=high_f,
        notch_f=notch_f,
        bandpass_order=bandpass_order,
        smoothing_window=smoothing_window,
        base_save_path=graph_path
        )
    print(np.max(data_processed_list[-1]))
    print(np.min(data_processed_list[-1]))

    # After preprocessing
    plot_raw_emg(
        data_processed_list,
        frequency,
        "After preprocessing",
        graph_path,
        test_label
        )


def emg_preprocess(emg_list: list,
                   frequency: float,
                   low_f: int,
                   high_f: int,
                   notch_f: list,
                   bandpass_order: int,
                   smoothing_window: float,  # 10.0 ms
                   base_save_path: str  # ./Test_data/graph/
                   ):
    # Basic procedure
    ############
    # FFT -> plot -> remove mean ->
    # notch filter -> bandpass filter -> rectification
    # moving rms smoothing -> normalization
    ############
    processed_emg_list = []
    for data_index in np.arange(len(emg_list) - 1):
        each_data = np.array(emg_list[data_index])
        # FFT
        freq_x, freq_y = Preprocess.fast_fourier_transform(
            emg_data=each_data,
            sampling_freq=frequency
            )
        # FFT plotting
        Preprocess.fft_plotting(
            x_data=freq_x,
            y_data=freq_y,
            title="EMG_" + str(data_index + 1) + "_before_preprocessing",
            save_path=\
                base_save_path + "EMG_" +\
                    str(data_index + 1) + "_before_preprocessing.png"
            )

        # Remove mean
        each_data = Preprocess.remove_mean(
            data=each_data
            )
        # Bandpass filter
        each_data = Preprocess.bandpass_filter(
            data=each_data,
            low_f=low_f,
            high_f=high_f,
            order=bandpass_order,
            sampling_freq=frequency,
            normalized=True
            )
        # Notch filter
        for each_notch_f in notch_f:
            each_data = Preprocess.notch_filter(
                data=each_data,
                notch_freq=each_notch_f,
                lowest_freq=low_f,
                highest_freq=high_f,
                sampling_freq=frequency,
                normalized=True
                )

        # FFT
        freq_x, freq_y = Preprocess.fast_fourier_transform(
            emg_data=np.array(each_data),
            sampling_freq=frequency
            )
        # FFT plotting
        Preprocess.fft_plotting(
            x_data=freq_x,
            y_data=freq_y,
            title="EMG_" + str(data_index + 1) + "_after_filtering",
            save_path=\
                base_save_path + "EMG_" +\
                    str(data_index + 1) + "_after_filtering.png"
            )

        # Rectification
        each_data = Preprocess.rectification(
            data=each_data
            )
        # Moving rms smoothing
        each_data = Preprocess.moving_rms_smoothing(
            data=each_data,
            smoothing_window=smoothing_window,
            sampling_freq=frequency
            )
        # Standard value calculation
        standard_value = Preprocess.standard_value_calculation(
            data=each_data,
            mvc_flag=False
        )
        # Normalization
        each_data = Preprocess.normalization(
            data=each_data,
            standard_value=standard_value
        )
        each_data = pd.DataFrame(each_data)
        processed_emg_list.append(each_data)
    processed_emg_list.append(emg_list[-1])

    return processed_emg_list


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


def plot_raw_emg(emg_list: list,
                 frequency: float,
                 title: str,
                 base_save_path: str,  # ./Test_data/
                 test_label: str):  # 1011_test1
    # time list [ms]
    data_time = np.linspace(
        0, (len(emg_list[0]) - 1) * (1000.0 / frequency), len(emg_list[0])
        )

    # EMG plot
    fig_emg, axs = plt.subplots(
        ((len(emg_list) - 1) // 2), 2, sharex=True, figsize=(20, 30)
        )
    fig_emg.suptitle('[' + test_label + ']' + title, fontsize=50)
    for i in np.arange(len(emg_list) - 1):
        j = i // 2
        axs[j, (i % 2)].plot(
            data_time, np.array(emg_list[i]), label="emg" + str(i + 1)
            )
        axs[j, (i % 2)].set_ylabel("EMG [uV]", fontsize=25)
        axs[j, (i % 2)].set_xlabel("Time [ms]", fontsize=25)
        axs[j, (i % 2)].legend(loc="best", fontsize=20)
        axs[j, (i % 2)].tick_params(axis="x", labelsize=15)
        axs[j, (i % 2)].tick_params(axis="y", labelsize=15)
    fig_emg.show()
    fig_emg.savefig(
        base_save_path + 'emg_' + title + '.png'
        )

    # Strain plot
    fig_strain, bx = plt.subplots(1, figsize=(10, 8))
    fig_strain.suptitle(title, fontsize=20)
    bx.plot(
        data_time, np.array(emg_list[len(emg_list) - 1]),
        label="strain")
    bx.set_ylabel("Strain", fontsize=20)
    bx.set_xlabel("Time [ms]", fontsize=20)
    bx.legend(loc="best", fontsize=20)
    bx.tick_params(axis="x", labelsize=15)
    bx.tick_params(axis="y", labelsize=15)
    fig_strain.tight_layout()
    fig_strain.show()
    fig_strain.savefig(
        base_save_path + 'strain_' + title + '.png'
        )


# Data 1: 1011 test 1 sleeve
# Data 2: 1012 column 1
# Data 3: 1012 column 2
# Data 4: 1012 column 3

# EMG mapping (relabeling for code) <- only 231011 standard
############
# Biceps
# 1 2
# 3 4
# 5 6
# 7 8
# Elbow
############
# number 1: emg 10
# number 2: emg 9
# number 3: emg 32
# number 4: emg 13
# number 5: emg 22
# number 6: emg 20
# number 7: emg 25
# number 8: emg 23
############
# <- only 231012 standard
# Biceps
# 1 2
# 3 4
# 5 6
# 7 8
# Elbow
############
# number 1: emg 8
# number 2: emg 11
# number 3: emg 5
# number 4: emg 17
# number 5: emg 14
# number 6: emg 19
# number 7: emg 2
# number 8: emg 29

# EMG (uV)
# (Data - 32768) * 0.195 uV
# Frequency: 2000 Hz
# Time (ms)
# interval: 0.5 ms

if __name__ == "__main__":

    frequency = 2000.0
    low_f = 30
    high_f = 250
    notch_f = [50]
    bandpass_order = 4
    smoothing_window = 10.0

    #######################################################
    # 1011 test 1
    # path_1 = "./Test_data/1011_HSJ_prior_test_1_sleeve.txt"
    # graph_base_path_1 = './Test_data/graph/1011_test1/'
    # test_1_label = '1011_test1'

    # emg_main(txt_path=path_1,
    #          graph_path=graph_base_path_1,
    #          test_label=test_1_label,
    #          frequency=frequency,
    #          low_f=low_f,
    #          high_f=high_f,
    #          notch_f=notch_f,
    #          bandpass_order=bandpass_order,
    #          smoothing_window=smoothing_window
    #          )

    #######################################################
    # 1012 test 1: 1012_HSJ_prior_test_sleeve
    path_2 = "./Test_data/1012.xlsx"
    sheet_name_2 = '1012_HSJ_prior_test_sleeve'
    graph_base_path_2 = './Test_data/graph/1012_test1/'
    test_2_label = '1012_test1'

    emg_main(txt_path=path_2,
             graph_path=graph_base_path_2,
             test_label=test_2_label,
             frequency=frequency,
             low_f=low_f,
             high_f=high_f,
             notch_f=notch_f,
             bandpass_order=bandpass_order,
             smoothing_window=smoothing_window,
             sheet_name=sheet_name_2
             )

    #######################################################
    # 1012 test 2: 1012_HSJ_prior_test_sleeve_1 (2
    # path_3 = "./Test_data/1012.xlsx"
    # sheet_name_3 = '1012_HSJ_prior_test_sleeve_1 (2'
    # graph_base_path_3 = './Test_data/graph/1012_test2/'
    # test_3_label = '1012_test2'

    # emg_main(txt_path=path_3,
    #          graph_path=graph_base_path_3,
    #          test_label=test_3_label,
    #          frequency=frequency,
    #          low_f=low_f,
    #          high_f=high_f,
    #          notch_f=notch_f,
    #          bandpass_order=bandpass_order,
    #          smoothing_window=smoothing_window,
    #          sheet_name=sheet_name_3
    #          )

    #######################################################
    # 1012 test 3: 1012_HSJ_prior_test_sleeve_1
    # path_4 = "./Test_data/1012.xlsx"
    # sheet_name_4 = '1012_HSJ_prior_test_sleeve_1'
    # graph_base_path_4 = './Test_data/graph/1012_test3/'
    # test_4_label = '1012_test3'

    # emg_main(txt_path=path_4,
    #          graph_path=graph_base_path_4,
    #          test_label=test_4_label,
    #          frequency=frequency,
    #          low_f=low_f,
    #          high_f=high_f,
    #          notch_f=notch_f,
    #          bandpass_order=bandpass_order,
    #          smoothing_window=smoothing_window,
    #          sheet_name=sheet_name_4
    #          )
