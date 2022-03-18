"""
Théo Gauvrit
Mars 2022
Centralize all the functions of EPA.
Compute PSD analysis, first method upstate/downstate detection, and second method (alenda et al) upstate/downstate method
"""
import pandas as pd
import os
from SpontaneousActivity.PSDAnalysis import plot_periodogram, computation_psd_bands, psd_computation, psd
from SpontaneousActivity.UpstateDownstateAlendaMethod import group_UDSD_computation
from SpontaneousActivity.UpstateDownstateAnalysis import two_groups_states_computation, udsd
from EvokedActivity.MAE import mae
from Utils import merge
if __name__ == '__main__':
    # # Parameters to modify
    # group1_name = "WT"
    # group2_name = "KO"
    # group1_path = "WT/subthreshold responses/Spontaneous activity"
    # group2_path = "KO/subthreshold responses/Spontaneous activity"
    # sampling_frequency = 20000
    # bands = {"Delta": [0.5, 4], "Theta": [4, 7], "Alpha": [8, 12], "Beta": [13, 30],
    #          "Gamma": [30, 100]}
    # ##########################
    #
    # folders = {group1_name: group1_path, group2_name: group2_path}
    #
    # # PSD
    # psd_grp1, freqs = psd_computation(group1_name, group1_path, sampling_frequency)
    # psd_grp2, freqs = psd_computation(group2_name, group2_path, sampling_frequency)
    # plot_periodogram(group1_name, group2_name, psd_grp1, psd_grp2, freqs, lim_band=[0, 100], median=True)
    # output_result = computation_psd_bands(psd_grp1, psd_grp2, freqs, group1_name, group2_name, bands)
    #
    # # Upstate Downstate first method
    # downsampling_coeff = 1  # don't modify
    # every_states_df, output_df = two_groups_states_computation(group1_name, group2_name, group1_path, group2_path,
    #                                                            sampling_frequency, downsampling_coeff)
    #
    # # Upstate Downstate second method (Alenda method)
    # output = group_UDSD_computation(group1_name, group2_name, group1_path, group2_path)

    # Test
    sampling_frequency = 20000
    bands = {"Delta": [0.5, 4], "Theta": [4, 7], "Alpha": [8, 12], "Beta": [13, 30],
             "Gamma": [30, 100]}
    server_path = "/run/user/1004/gvfs/afp-volume:host=engram.local,user=Theo%20Gauvrit,volume=Data/Yukti/" \
                  "In Vivo Patch Clamp Recordings/"
    groups_path = {
        "KO BMS191011": {
            "spontaneous": server_path + "Spontaneous Activity_FmKO/For Theo"
            , "evoked":  server_path + "Stimulus Evoked Responses_FmKO/KO BMS191011"
        }
        , "KO DMSO": {
            "spontaneous":  server_path + "Spontaneous Activity_FmKO/KO DMSO"
            , "evoked": server_path + "Stimulus Evoked Responses_FmKO/KO DMSO"
        }
    }
    output_table_evoked = pd.DataFrame()
    output_table_spontaneuous = pd.DataFrame()
    for group_name in groups_path.keys():
        # Spontaneous
        # output_psd = psd(groups_path[group_name]["spontaneous"], group_name, sampling_frequency, bands)
        # all_states_df, output_udsd = udsd(group_name, groups_path[group_name]["spontaneous"], sampling_frequency, 1)
        # output_merged = merge([output_psd, output_udsd])
        # output_table_spontaneuous = output_table_spontaneuous.append(output_merged, ignore_index=True)
        # Evoked
        output_mae = mae(group_name, groups_path[group_name]["evoked"], sampling_frequency)
        output_table_evoked = output_table_evoked.append(output_mae, ignore_index=True)

    # output_table_spontaneuous .to_excel("analysed_spontaneuous_data.xlsx")
    output_table_evoked.to_excel("analysed_evoked_data.xlsx")
