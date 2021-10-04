"""
Théo Gauvrit
Spetembre 2021
Centralize all the functions for the spontaneous recordings analysis.
Compute PSD analysis, first method upstate/downstate detection, and second method (alenda et al) upstate/downstate method
"""

from PSDAnalysis import plot_periodogram, computation_psd_bands, psd_computation
from UpstateDownstateAnalysis import two_groups_states_computation
from UpstateDownstateAlendaMethod import group_UDSD_computation




if __name__ == '__main__':
    # Parameters to modify
    group1_name = "WT"
    group2_name = "KO"
    group1_path = "WT/subthreshold responses/Spontaneous activity"
    group2_path = "KO/subthreshold responses/Spontaneous activity"
    sampling_frequency = 20000
    bands = {"Delta": [0.5, 4], "Theta": [4, 7], "Alpha": [8, 12], "Beta": [13, 30],
             "Gamma": [30, 100]}
    ##########################


    folders = {group1_name: group1_path, group2_name: group2_path}

    ###PSD###
    psd_grp1, freqs = psd_computation(group1_name, group1_path, sampling_frequency)
    psd_grp2, freqs = psd_computation(group2_name, group2_path, sampling_frequency)
    plot_periodogram(group1_name,group2_name,psd_grp1,psd_grp2,freqs,lim_band=[0, 100],median=True)
    output_result=computation_psd_bands(psd_grp1,psd_grp2,freqs,group1_name, group2_name,bands)

    ###Upstate Downstate first method###
    downsampling_coeff = 1  # don't modify
    every_states_df, output_df = two_groups_states_computation(group1_name, group2_name, group1_path, group2_path,
                                                               sampling_frequency, downsampling_coeff)

    ###Upstate Downstate second method (Alenda method)###
    output=group_UDSD_computation(group1_name, group2_name, group1_path, group2_path)
