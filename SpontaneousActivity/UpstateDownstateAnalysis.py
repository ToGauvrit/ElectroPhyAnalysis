"""
Théo Gauvrit

September 2021

Rework of the upstate/downstate detection scripts to make them clearer and easier to use
The process is still long.
"""
from UDSD_09 import get_states

import matplotlib.pyplot as plt
import pandas as pd
import os
import pyabf
import numpy as np
from Utils import browse_directory


def states_computation(group_name, directory_path, sf, downsampling_coeff):
    print("Up/Down state computaion for " + str(group_name) + " files")
    output_dataframe = pd.DataFrame()
    all_states_df = pd.DataFrame()
    files = browse_directory(directory_path, ".abf")
    for filename in files:
        print(filename)
        abf_signal = pyabf.ABF(os.path.join(directory_path, filename))
        signal = np.array(abf_signal.sweepY)
        metrics, states_list, udsd_dic = get_states(signal, sf, filename, downsampling_coeff)
        udsd_dic["group"] = group_name
        all_states_df = all_states_df.append(states_list, ignore_index=True)
        output_dataframe = output_dataframe.append(udsd_dic, ignore_index=True)
    return all_states_df, output_dataframe


def two_groups_states_computation(group1_name, group2_name, group1_path, group2_path, sf, downsampling_coeff, filename_output=None,filename_states=None):
    """Return two dataframes:
            -every_states_dataframe: contains details mesure on every states detected
            -output_data_frame: classic dataframe containing Up/Down states measures
            for every cells for both groups
    """
    output_dataframe=pd.DataFrame()
    states_dataframe = pd.DataFrame()
    grp1_all_states, grp1_output = states_computation(group1_name, group1_path, sf, downsampling_coeff)
    grp2_all_states, grp2_output = states_computation(group2_name, group2_path, sf, downsampling_coeff)
    output_dataframe = output_dataframe.append(grp1_output,ignore_index=True)
    output_dataframe = output_dataframe.append(grp2_output, ignore_index=True)
    states_dataframe = states_dataframe.append(grp1_all_states,ignore_index=True)
    states_dataframe = states_dataframe.append(grp2_all_states, ignore_index=True)
    output_dataframe = output_dataframe[["filename", "group", "up_duration", 'up_frequency', 'up_value', "down_duration", "down_frequency", "down_value"]]  # reorder the columns
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
    # Parameters to modify
    group1_name = "WT"
    group2_name = "KO"
    group1_path = "temp_test_WT"
    group2_path = "temp_test_ko"
    sampling_frequency = 20000  # Hz
    #(Optional) name of the filename that will be save containing info by cell
    filename_output="Test1.xlsx"
    #(Optional) name of the filename that will be save containing info by states
    filename_states="Test2.xlsx"
    ###########################
    """!!!!!!!!!!!!Don't change downsampling_coeff!!!!!!!!!!!!!!!!"""
    """The downsamping coeff is to make the computation faster by reducing the number of points taken in account 
    for the computation. It's only useful when the sampling rate is very high(ex: >3khz).But need improvement.
    """
    downsampling_coeff = 1#Serioulsy don't
    ###########################
    folders = {group1_name: group1_path, group2_name: group2_path}
    every_states_df, output_df = two_groups_states_computation(group1_name, group2_name, group1_path, group2_path,
                                                               sampling_frequency, downsampling_coeff)
