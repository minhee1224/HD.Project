# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 23:36:10 2023

@author: mleem
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


from final_function import *


def emg_main(base_path: str,
             mvc_num: int,
             txt_path: str,
             graph_path: str,
             test_label: str,
             mvc_test_label: str,
             frequency: float,
             low_f: int,
             high_f: int,
             bandpass_order: int,
             smoothing_window: float,
             mvc_limit: float,
             mvc_quality_value: list,
             quality_value: list,
             sheet_name='Sheet1',
             mvc_sheet_name='Sheet1',
             plot_flag=False,
             mvc_plot_flag=False,
             mvc_flag=False
             ):

    graph_path = graph_path + test_label + '/'
    create_folder(graph_path)

    if test_label[:4] == '1012':
        data_list = emg_data_txt_reading_231012(
            txt_path, sheet_name=sheet_name
            )
    else:  # 1011, 1013 data
        data_list = emg_data_txt_reading_231011(
            txt_path, sheet_name=sheet_name
            )

    if plot_flag:
        # Beforce preprocessing
        plot_raw_emg(
            data_list,
            frequency,
            "Before preprocessing",
            graph_path,
            test_label
            )

    # MVC calculation
    if mvc_flag:
        mvc_list = np.arange(1, mvc_num + 1)
        for mvc_ind in mvc_list:
            mvc_ind = str(mvc_ind)
            mvc_path_ = base_path + mvc_sheet_name + mvc_ind + '.txt'
            test_label_ = mvc_test_label + mvc_ind
            print("mvc")
            mvc_value_list_ = emg_mvc_main(txt_path=mvc_path_,
                                           graph_path=graph_path,
                                           test_label=test_label_,
                                           frequency=frequency,
                                           low_f=low_f,
                                           high_f=high_f,
                                           bandpass_order=bandpass_order,
                                           smoothing_window=smoothing_window,
                                           mvc_limit=mvc_limit,
                                           mvc_quality_value=mvc_quality_value,
                                           sheet_name=mvc_sheet_name,
                                           plot_flag=mvc_plot_flag
                                           )
            if mvc_ind == '1':
                mvc_value_list = mvc_value_list_
            else:
                mvc_value_list += mvc_value_list_
        mvc_value_list /= mvc_num
        # print("mvc")
        # print(mvc_value_list)
    else:
        mvc_value_list = [0, 0, 0, 0, 0, 0, 0, 0]

    # Preprocessing
    print("emg main")
    data_processed_list,\
        freq_x_before_inner_list, freq_y_before_inner_list,\
            freq_x_after_inner_list, freq_y_after_inner_list = emg_preprocess(
                emg_list=data_list,
                frequency=frequency,
                low_f=low_f,
                high_f=high_f,
                bandpass_order=bandpass_order,
                smoothing_window=smoothing_window,
                base_save_path=graph_path,
                mvc_flag=mvc_flag,
                mvc_list=mvc_value_list,
                quality_value=quality_value,
                plot_flag=plot_flag
                )

    if plot_flag:
        # After preprocessing
        plot_raw_emg(
            data_processed_list,
            frequency,
            "After preprocessing",
            graph_path,
            test_label
            )

    # strain data division
    # time list [ms]
    data_time = np.linspace(
        0,
        (len(data_processed_list[-1]) - 1) * (1000.0 / frequency),
        len(data_processed_list[-1])
        )
    peaks, _ = signal.find_peaks(
        data_processed_list[-1], prominence=300.0
        )
    new_peak = []
    for peak_ind, peak in enumerate(peaks):
        if peak_ind == 0:
            new_peak.append(peak)
            continue
        if peak - new_peak[-1] <= 2000:
            continue
        new_peak.append(peak)
    valleys, _ = signal.find_peaks(
        (-1) * data_processed_list[-1], prominence=100.0
        )
    new_valley = []
    for valley_ind, valley in enumerate(valleys):
        if valley_ind == 0:
            new_valley.append(valley)
            continue
        if valley - new_valley[-1] <= 2000:
            continue
        new_valley.append(valley)
    # print(data_time[peaks])
    # print(data_time[new_peak])
    # print(data_time[valleys])
    # print(data_time[new_valley])
    # plt.plot(data_time, data_processed_list[-1])
    # plt.plot(data_time[new_peak], data_processed_list[-1][new_peak], "x")
    # plt.plot(data_time[new_valley], data_processed_list[-1][new_valley], "o")

    # emg number * freq list
    return freq_x_before_inner_list, freq_y_before_inner_list,\
        freq_x_after_inner_list, freq_y_after_inner_list


def emg_mvc_main(txt_path: str,
                 graph_path: str,
                 test_label: str,
                 frequency: float,
                 low_f: int,
                 high_f: int,
                 bandpass_order: int,
                 smoothing_window: float,
                 mvc_limit: float,
                 mvc_quality_value: list,
                 sheet_name='Sheet1',
                 plot_flag=False
                 ):

    graph_path = graph_path + test_label + '/'
    create_folder(graph_path)

    if test_label[:4] == '1012':
        data_list = emg_data_txt_reading_231012(
            txt_path, sheet_name=sheet_name
            )
    else:  # 1011, 1013 data
        data_list = emg_data_txt_reading_231011(
            txt_path, sheet_name=sheet_name
            )

    if plot_flag:
        # Beforce preprocessing
        plot_raw_emg(
            data_list,
            frequency,
            "Before preprocessing",
            graph_path,
            test_label
            )

    # Preprocessing
    data_processed_list, mvc_list = emg_mvc_preprocess(
        emg_list=data_list,
        frequency=frequency,
        low_f=low_f,
        high_f=high_f,
        bandpass_order=bandpass_order,
        smoothing_window=smoothing_window,
        mvc_limit=mvc_limit,
        mvc_quality_value=mvc_quality_value,
        base_save_path=graph_path,
        plot_flag=plot_flag
        )

    if plot_flag:
        # After preprocessing
        plot_raw_emg(
            data_processed_list,
            frequency,
            "After preprocessing",
            graph_path,
            test_label
            )

    return mvc_list


def emg_preprocess(emg_list: list,
                   frequency: float,
                   low_f: int,
                   high_f: int,
                   bandpass_order: int,
                   smoothing_window: float,  # 10.0 ms
                   base_save_path: str,  # ./Test_data/graph/,
                   mvc_list: np.array,
                   quality_value: list,
                   mvc_flag=False,
                   plot_flag=False
                   ):
    # Basic procedure
    ############
    # FFT -> plot -> remove mean ->
    # notch filter -> bandpass filter -> rectification
    # moving rms smoothing -> normalization
    ############
    processed_emg_list = []
    freq_x_before_inner_list = []
    freq_y_before_inner_list = []
    freq_x_after_inner_list = []
    freq_y_after_inner_list = []

    # emg number
    for data_index in np.arange(len(emg_list) - 1):
        inner_quality_value = quality_value[data_index]
        each_data = np.array(emg_list[data_index])
        # FFT
        freq_x_before, freq_y_before, notch_f =\
            Preprocess.fast_fourier_transform(
                emg_data=each_data,
                notch_limit=2.5,
                sampling_freq=frequency
                )
        freq_x_before_inner_list.append(list(freq_x_before))
        freq_y_before_inner_list.append(list(freq_y_before))

        if plot_flag:
            # FFT plotting
            Preprocess.fft_plotting(
                x_data=freq_x_before,
                y_data=freq_y_before,
                title="EMG_" + str(data_index + 1) + "_before_preprocessing",
                lowest_freq=low_f,
                highest_freq=high_f,
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
                quality_value=inner_quality_value,
                notch_freq=each_notch_f,
                lowest_freq=low_f,
                highest_freq=high_f,
                sampling_freq=frequency
                )

        # FFT
        freq_x_after, freq_y_after, notch_f =\
            Preprocess.fast_fourier_transform(
                emg_data=np.array(each_data),
                notch_limit=2.5,
                sampling_freq=frequency
                )
        freq_x_after_inner_list.append(list(freq_x_after))
        freq_y_after_inner_list.append(list(freq_y_after))

        if plot_flag:
            # FFT plotting
            Preprocess.fft_plotting(
                x_data=freq_x_after,
                y_data=freq_y_after,
                title="EMG_" + str(data_index + 1) + "_after_filtering",
                lowest_freq=low_f,
                highest_freq=high_f,
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
            mvc_flag=mvc_flag,
            mvc_value=mvc_list[data_index]
        )
        # Normalization
        each_data = Preprocess.normalization(
            data=each_data,
            standard_value=standard_value
        )
        each_data = pd.DataFrame(each_data)
        processed_emg_list.append(each_data)
    processed_emg_list.append(emg_list[-1])

    # emg number * freq list
    return processed_emg_list,\
        freq_x_before_inner_list, freq_y_before_inner_list,\
            freq_x_after_inner_list, freq_y_after_inner_list


def emg_mvc_preprocess(emg_list: list,
                       frequency: float,
                       low_f: int,
                       high_f: int,
                       bandpass_order: int,
                       smoothing_window: float,  # 10.0 ms
                       mvc_limit: float,  # 25.0
                       mvc_quality_value: list,
                       base_save_path: str,  # ./Test_data/graph/
                       plot_flag=False
                       ):
    # Basic procedure
    ############
    # FFT -> plot -> remove mean ->
    # notch filter -> bandpass filter -> rectification
    # moving rms smoothing -> normalization
    ############
    processed_emg_list = []
    processed_mvc_list = []
    for data_index in np.arange(len(emg_list) - 1):
        quality_value = mvc_quality_value[data_index]
        each_data = np.array(emg_list[data_index])
        # FFT
        freq_x, freq_y, notch_f = Preprocess.fast_fourier_transform(
            emg_data=each_data,
            notch_limit=5.0,
            sampling_freq=frequency
            )
        if plot_flag:
            # FFT plotting
            Preprocess.fft_plotting(
                x_data=freq_x,
                y_data=freq_y,
                title="EMG_" + str(data_index + 1) + "_before_preprocessing",
                lowest_freq=low_f,
                highest_freq=high_f,
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
                quality_value=quality_value,
                notch_freq=each_notch_f,
                lowest_freq=low_f,
                highest_freq=high_f,
                sampling_freq=frequency
                )

        # FFT
        freq_x, freq_y, notch_f = Preprocess.fast_fourier_transform(
            emg_data=np.array(each_data),
            notch_limit=4.0,
            sampling_freq=frequency
            )
        if plot_flag:
            # FFT plotting
            Preprocess.fft_plotting(
                x_data=freq_x,
                y_data=freq_y,
                title="EMG_" + str(data_index + 1) + "_after_filtering",
                lowest_freq=low_f,
                highest_freq=high_f,
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
        # MVC calculation
        each_data = each_data[each_data > mvc_limit]
        each_mvc = np.mean(each_data, axis=0)
        # print(each_mvc)
        processed_mvc_list.append(each_mvc)

        each_data = pd.DataFrame(each_data)
        processed_emg_list.append(each_data)
    processed_emg_list.append(emg_list[-1])

    return processed_emg_list, np.array(processed_mvc_list)


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
    fig_emg.savefig(
        base_save_path + 'emg_' + title + '.png'
        )

    # Strain plot (+ EMG 8)
    fig_strain, bx = plt.subplots(1, figsize=(10, 8))
    fig_strain.suptitle(title, fontsize=20)
    bx.plot(
        data_time, np.array(emg_list[len(emg_list) - 1]),
        label="strain", color="orange", linewidth=5.0)
    # 비교용 EMG
    bx2 = bx.twinx()
    bx2.plot(
        data_time, np.array(emg_list[len(emg_list) - 2]),
        label="emg8", color="blue")
    bx2.legend(loc="best", fontsize=20)
    ######################################
    bx.set_ylabel("Strain", fontsize=20)
    bx.set_xlabel("Time [ms]", fontsize=20)
    bx.legend(loc="best", fontsize=20)
    bx.tick_params(axis="x", labelsize=15)
    bx.tick_params(axis="y", labelsize=15)
    fig_strain.tight_layout()
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
    bandpass_order = 4
    smoothing_window = 10.0
    mvc_limit = 50.0
    mvc_quality_value = [0.001, 0.001, 0.001, 0.001,
                         0.001, 0.001, 0.001, 0.001]
    quality_value = [0.0015, 0.0015, 0.0015, 0.0015,
                     0.0015, 0.0015, 0.0015, 0.0015]

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
    # path_2 = "./Test_data/1012.xlsx"
    # sheet_name_2 = '1012_HSJ_prior_test_sleeve'
    # graph_base_path_2 = './Test_data/graph/1012_test1/'
    # test_2_label = '1012_test1'

    # emg_main(txt_path=path_2,
    #          graph_path=graph_base_path_2,
    #          test_label=test_2_label,
    #          frequency=frequency,
    #          low_f=low_f,
    #          high_f=high_f,
    #          notch_f=notch_f,
    #          bandpass_order=bandpass_order,
    #          smoothing_window=smoothing_window,
    #          sheet_name=sheet_name_2
    #          )

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

    #######################################################
    # 1013 test:
    # test_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # test_list = [1, 2]

    # freq list
    # freq_x_before_list_total = []  # 1
    # freq_y_before_list_total = []  # 1
    # freq_x_after_list_total = []  # 1
    # freq_y_after_list_total = []  # 1

    # path_ = "./Test_data/1013/"
    # sheet_name_ = '1013_HSJ_312_test_set'
    # mvc_sheet_name_ = '1013_HSJ_312_test_MVC_'
    # mvc_test_label = '1013_mvc'
    # graph_base_path = './Test_data/graph/1013/'
    # for emg_ind in test_list:
    #     emg_ind = str(emg_ind)
    #     path_1 = path_ + sheet_name_ + emg_ind + '.txt'
    #     test_label_1 = '1013_test' + emg_ind

    #     # emg number * freq list
    #     freq_x_before_list, freq_y_before_list,\
    #         freq_x_after_list, freq_y_after_list =\
    #             emg_main(
    #                 base_path=path_,
    #                 mvc_num=3,
    #                 txt_path=path_1,
    #                 graph_path=graph_base_path,
    #                 test_label=test_label_1,
    #                 mvc_test_label=mvc_test_label,
    #                 frequency=frequency,
    #                 low_f=low_f,
    #                 high_f=high_f,
    #                 bandpass_order=bandpass_order,
    #                 smoothing_window=smoothing_window,
    #                 mvc_limit=mvc_limit,
    #                 mvc_quality_value=mvc_quality_value,
    #                 quality_value=quality_value,
    #                 sheet_name=sheet_name_,
    #                 mvc_sheet_name=mvc_sheet_name_,
    #                 plot_flag=False,
    #                 mvc_plot_flag=False
    #                 )
    #     # set * emg number * freq list
    #     freq_x_before_list_total.append(freq_x_before_list)  # set
    #     freq_y_before_list_total.append(freq_y_before_list)  # set
    #     freq_x_after_list_total.append(freq_x_after_list)  # set
    #     freq_y_after_list_total.append(freq_y_after_list)  # set

    # fft subplots (before)
    # Preprocess.fft_list_subplots(
    #     x_data=freq_x_before_list_total,
    #     y_data=freq_y_before_list_total,
    #     title='[FFT] EMG before preprocessing',
    #     save_path=graph_base_path + "number_",
    #     lowest_freq=low_f,
    #     highest_freq=high_f
    #     )

    # # fft subplots (after)
    # Preprocess.fft_list_subplots(
    #     x_data=freq_x_after_list_total,
    #     y_data=freq_y_after_list_total,
    #     title='[FFT] EMG after preprocessing',
    #     save_path=graph_base_path + "number_",
    #     lowest_freq=low_f,
    #     highest_freq=high_f
    #     )

    #######################################################
    # 1016 test:
    # test_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    test_list = [1]

    # freq list
    freq_x_before_list_total = []  # 1
    freq_y_before_list_total = []  # 1
    freq_x_after_list_total = []  # 1
    freq_y_after_list_total = []  # 1

    path_ = "./Test_data/1016/"
    sheet_name_ = '1016_KJN_312_test_'
    mvc_sheet_name_ = ''
    mvc_test_label = ''
    graph_base_path = './Test_data/graph/1016/'
    for emg_ind in test_list:
        emg_ind = str(emg_ind)
        path_1 = path_ + sheet_name_ + emg_ind + '.txt'
        test_label_1 = '1016_test' + emg_ind

        # emg number * freq list
        freq_x_before_list, freq_y_before_list,\
            freq_x_after_list, freq_y_after_list =\
                emg_main(
                    base_path=path_,
                    mvc_num=3,
                    txt_path=path_1,
                    graph_path=graph_base_path,
                    test_label=test_label_1,
                    mvc_test_label=mvc_test_label,
                    frequency=frequency,
                    low_f=low_f,
                    high_f=high_f,
                    bandpass_order=bandpass_order,
                    smoothing_window=smoothing_window,
                    mvc_limit=mvc_limit,
                    mvc_quality_value=mvc_quality_value,
                    quality_value=quality_value,
                    sheet_name=sheet_name_,
                    mvc_sheet_name=mvc_sheet_name_,
                    mvc_flag=False,
                    plot_flag=True,
                    mvc_plot_flag=False
                    )
        # set * emg number * freq list
        freq_x_before_list_total.append(freq_x_before_list)  # set
        freq_y_before_list_total.append(freq_y_before_list)  # set
        freq_x_after_list_total.append(freq_x_after_list)  # set
        freq_y_after_list_total.append(freq_y_after_list)  # set

    # fft subplots (before)
    Preprocess.fft_list_subplots(
        x_data=freq_x_before_list_total,
        y_data=freq_y_before_list_total,
        title='[FFT] EMG before preprocessing',
        save_path=graph_base_path + "number_",
        lowest_freq=low_f,
        highest_freq=high_f
        )

    # # fft subplots (after)
    Preprocess.fft_list_subplots(
        x_data=freq_x_after_list_total,
        y_data=freq_y_after_list_total,
        title='[FFT] EMG after preprocessing',
        save_path=graph_base_path + "number_",
        lowest_freq=low_f,
        highest_freq=high_f
        )

