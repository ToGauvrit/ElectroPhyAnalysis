"""
Théo Gauvrit
Mai 2020

Compute the detection of Upstate and Downstates from two datasets(WT,KO).
Each Up/Downstates parameters from every files will be returned in a dataframe.
A dataframe (Metrics) can be returned to have visual of the detection in the signal in a plot.
"""
import numpy as np
import math as math
from scipy.stats import variation
import pandas as pd
import time
import os
from progress.bar import IncrementalBar
import pyabf
from Utils import browse_directory


# Todo rework this code again to make it clearer and faster, implement downsampling


def get_states(signal, sampling_frequency, filename, downsampling, p_duration=None):
    """Return the signal with the corresponding detection decision ready to plot,
    the dataframe containing the list of states and the correspondings measures and
    a dictionary with the average measure for the whole signal """

    def get_ti(timestamp, time_series):
        return len(time_series) - len(time_series[time_series > timestamp])

    def shift_corection(data, index, coeff):
        return data + (-coeff * index)

    vec_shift = np.vectorize(shift_corection)

    signal = signal[::downsampling]
    chunk1 = np.median(signal[:100000])
    chunk2 = np.median(signal[-100000:])
    coeff = (chunk2 - chunk1) / len(signal)
    indices = range(len(signal))
    signal = vec_shift(signal, indices, coeff)
    n_points = len(signal)  # number of points
    # time values
    time_step = (5e-5) * downsampling
    if p_duration == None:
        duration = n_points / sampling_frequency
    else:
        duration = p_duration
    time_set = np.arange(0, duration, time_step)
    rang = np.arange(1, len(time_set), 1)
    signal = np.array(signal, dtype=object)
    # automatic detection of up and down states
    min_value = min(signal)
    max_value = max(signal)
    uads = []
    min_state_duration = 0.1  # s
    intervalle = get_ti(min_state_duration, time_set) - get_ti(0, time_set)
    h_int = math.floor(intervalle / 2)
    h_int_t = int(math.floor((8 / time_step) / 2))
    loc_range = np.arange((1 + h_int), (-1 + len(rang) - h_int), 1)
    loc_signal = []
    loc_time = []
    loc_threshold = []
    val = min_value
    threshold = np.median(signal[(0 + h_int_t)])
    for i in loc_range:
        loc_series = signal[(i - h_int):(i + h_int)]
        temp_median = np.median(loc_series)

        if i % 20000 == 0:
            threshold = np.median(signal[(i - h_int_t):(i + h_int_t)])
        to_test = temp_median
        if not np.isnan(to_test):

            if to_test > threshold:
                val = max_value

            else:
                val = min_value
        uads.append(val)
        loc_time.append(time_set[i])
        loc_signal.append(signal[i])
        loc_threshold.append(threshold)

    metrics = pd.DataFrame({"Time": loc_time, "Signal": loc_signal, "Threshold": loc_threshold, "uads": uads})
    metrics = metrics[5 * 20000:]

    res = {}
    mylen = np.vectorize(len)
    states_splitted = np.split(metrics["uads"], np.argwhere(np.diff(metrics["uads"]) != 0)[:, 0] + 1)
    index_states = np.argwhere(np.diff(metrics["uads"]) != 0)[:, 0] + 1
    states_list = pd.DataFrame()
    states_splitted.pop(0)
    for i in range(len(states_splitted) - 1):
        if states_splitted[i].iloc[0] == min_value:
            state = "Down"
        else:
            state = "Up"
        onset = index_states[i] / 20000
        state_item = pd.DataFrame({"Filename": filename,
                                   "State": state,
                                   "Start": onset + 5.05,
                                   "End": (index_states[i] + len(states_splitted[i])) / 20000 + 5.05,
                                   "Duration": len(states_splitted[i]) / 20000,
                                   "Mean Value": np.round(np.mean(
                                       metrics["Signal"][index_states[i]:index_states[i] + len(states_splitted[i])]),
                                                          3),
                                   "Variation PM": np.round(variation(
                                       metrics["Signal"][index_states[i]:index_states[i] + len(states_splitted[i])]), 3)
                                   }, index=[i])
        states_list = states_list.append(state_item)
    # frequency of Upstate and Downstate
    frequency_up = len(metrics["uads"][metrics["uads"] == max_value])
    frequency_down = len(metrics["uads"][metrics["uads"] == min_value])
    res["up_frequency"] = (frequency_up / len(metrics["uads"]))
    res["down_frequency"] = (frequency_down / len(metrics["uads"]))
    # Average of duration of Up/Downstate
    res["down_duration"] = (np.mean(mylen(states_splitted[::2])) * time_step)
    res["up_duration"] = (np.mean(mylen(np.delete(states_splitted, 0)[::2])) * time_step)
    # Average value of Up/Downstate
    res["down_value"] = np.mean(metrics["Signal"][metrics["uads"] == min_value])
    res["up_value"] = np.mean(metrics["Signal"][metrics["uads"] == max_value])
    res["filename"] = filename
    return metrics, states_list, res


def udsd(group_name, directory_path, sf, downsampling_coeff):
    print("Up/Down state computaion for " + str(group_name) + " files")
    output_dataframe = pd.DataFrame()
    all_states_df = pd.DataFrame()
    files = browse_directory(directory_path, ".abf")
    bar = IncrementalBar('Files anlyzed', max=len(files))
    for filename in files:
        bar.next()
        abf_signal = pyabf.ABF(os.path.join(directory_path, filename))
        signal = np.array(abf_signal.sweepY, dtype=object)
        metrics, states_list, udsd_dic = get_states(signal, sf, filename, downsampling_coeff)
        udsd_dic["group"] = group_name
        all_states_df = all_states_df.append(states_list, ignore_index=True)
        output_dataframe = output_dataframe.append(udsd_dic, ignore_index=True)
    bar.finish()
    return all_states_df, output_dataframe


def two_groups_states_computation(group1_name, group2_name, group1_path, group2_path, sf, downsampling_coeff,
                                  filename_output=None, filename_states=None):
    """Return two dataframes:
            -every_states_dataframe: contains details mesure on every states detected
            -output_data_frame: classic dataframe containing Up/Down states measures
            for every cells for both groups
    """
    output_dataframe = pd.DataFrame()
    states_dataframe = pd.DataFrame()
    grp1_all_states, grp1_output = udsd(group1_name, group1_path, sf, downsampling_coeff)
    grp2_all_states, grp2_output = udsd(group2_name, group2_path, sf, downsampling_coeff)
    output_dataframe = output_dataframe.append(grp1_output, ignore_index=True)
    output_dataframe = output_dataframe.append(grp2_output, ignore_index=True)
    states_dataframe = states_dataframe.append(grp1_all_states, ignore_index=True)
    states_dataframe = states_dataframe.append(grp2_all_states, ignore_index=True)
    output_dataframe = output_dataframe[
        ["filename", "group", "up_duration", 'up_frequency', 'up_value', "down_duration", "down_frequency",
         "down_value"]]  # reorder the columns
    if filename_output:
        output_dataframe.to_excel(os.path.join('UDSD' + group1_name + group2_name + ".xlsx"))
    else:
        output_dataframe.to_excel(os.path.join('UDSD' + group1_name + group2_name + ".xlsx"))

    if filename_states:
        states_dataframe.to_excel(os.path.join(filename_states))
    else:
        states_dataframe.to_excel(os.path.join('states' + group1_name + group2_name + ".xlsx"))
    return states_dataframe, output_dataframe


if __name__ == '__main__':
    start_time = time.time()
    # Parameters to modify
    group1_name = "ForTheo"
    group2_name = "WT DMSO"
    group1_path = ""
    group2_path = ""
    sampling_frequency = 20000  # Hz
    ###########################
    """!!!!!!!!!!!!Don't change downsampling_coeff!!!!!!!!!!!!!!!!"""
    """The downsamping coeff is to make the computation faster by reducing the number of points taken in account 
    for the computation. It's only useful when the sampling rate is very high(ex: >3khz).But need improvement.
    """
    downsampling_coeff = 1
    ###########################
    folders = {group1_name: group1_path, group2_name: group2_path}
    every_states_df, output_df = two_groups_states_computation(group1_name, group2_name, group1_path, group2_path,
                                                               sampling_frequency, downsampling_coeff)
    print("--- %s seconds ---" % (time.time() - start_time))