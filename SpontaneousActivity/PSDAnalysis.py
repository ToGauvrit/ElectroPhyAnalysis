"""
Théo Gauvrit
Spetembre 2021
General script to compute PSD for different frequency bands from resting menbrane potential recordings.
Comparaison of two groups(ex: WT vs KO)
Compute the PSD with bands, plot the wlech periodogram.
"""
import pyabf
import numpy as np
import matplotlib
import pandas as pd
import os
import time
import seaborn as sns
from scipy.signal import welch
import statsmodels as stats
from scipy.integrate import simps
import scipy.stats as sc
import matplotlib.pyplot as plt
from Utils import browse_directory


# Params to have improve plotting
plt.rcParams['font.size'] = 30
plt.rcParams['axes.linewidth'] = 3

start_time = time.time()
med_vec = np.vectorize(np.median)





def welch_psd(signal,sampling_frequency):
    signal = np.array(signal)
    win = 4 * sampling_frequency
    freqs, psd = welch(signal, sampling_frequency, nperseg=win, average='mean')
    return freqs, psd


def psd_computation(group_name,directory_path, sampling_frequency):
    print("PSD computaion for " + str(group_name) + " files")
    output_dataframe=pd.DataFrame()
    files = browse_directory(directory_path, ".abf")
    psd_list = []
    for filename in files:
        print(filename)
        abf_signal = pyabf.ABF(os.path.join(directory_path, filename))
        signal = np.array(abf_signal.sweepY)
        freqs, psd = welch_psd(signal, sampling_frequency)
        psd_list.append(psd)
    output_dataframe["PSD"] = psd_list
    output_dataframe["Filename"] = files
    output_dataframe["Group"] = [group_name]*len(output_dataframe["PSD"])
    return output_dataframe, freqs


def plot_periodogram(group1_name, group2_name, psd_dataframe_group1,psd_dataframe_group2, freqs, lim_band,output_filename=None,median=False,):
    """Plot the periodogram with two groups"""
    fig, ax = plt.subplots(1, 1, figsize=(13, 8))
    if median is True:
        plt.plot(freqs, np.median(list(psd_dataframe_group1["PSD"]), axis=0), color="black", lw=3, label=group1_name)
        plt.plot(freqs, np.median(list(psd_dataframe_group2["PSD"]), axis=0), color="red", lw=3, label=group2_name)
    else:
        plt.plot(freqs, np.mean(psd_dataframe_group1["PSD"], axis=0), color="black", lw=3, label=group1_name)
        plt.plot(freqs, np.mean(psd_dataframe_group2["PSD"], axis=0), color="red", lw=3, label=group2_name)
    plt.xlim(lim_band)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power spectral density(mV²/Hz)")
    plt.yscale("log")
    if output_filename:
        plt.savefig(output_filename)
    else:
        plt.savefig("perio" + group1_name + "Vs" + group2_name + ".pdf")
    plt.legend(fontsize=18, loc=3)
    plt.tight_layout()
    plt.show()


def get_power_per_bands(psd, bands, freqs):
    psd_per_bands = {"Delta": np.nan, "Theta": np.nan, "Alpha": np.nan, "Beta": np.nan, "Gamma": np.nan}
    for band in bands.keys():
        idx_delta = np.logical_and(freqs >= bands[band][0], freqs <= bands[band][1])
        # Frequency resolution
        freq_res = freqs[1] - freqs[0]
        # Compute the absolute power by approximating the area under the curve
        power = simps(psd[idx_delta], dx=freq_res)
        # relative power here but not used
        dp_relative = power / simps(psd, dx=freq_res)
        psd_per_bands[band] = power
    return psd_per_bands


def computation_psd_bands(psd_dataframe_group1,psd_dataframe_group2,freqs,group1_name, group2_name,bands, output_filename=None):
    """"Return and write the CSV containing the values for the two groups PSD"""
    output_result = pd.DataFrame()
    for index, row in psd_dataframe_group1.iterrows():
        psd_per_bands = get_power_per_bands(row["PSD"],bands,freqs)
        psd_per_bands["Group"]=row["Group"]
        psd_per_bands["Filename"] = row["Filename"]
        output_result=output_result.append(psd_per_bands,ignore_index=True)

    for index, row in psd_dataframe_group2.iterrows():
        psd_per_bands = get_power_per_bands(row["PSD"],bands,freqs)
        psd_per_bands["Group"]=row["Group"]
        psd_per_bands["Filename"] = row["Filename"]
        output_result=output_result.append(psd_per_bands,ignore_index=True)
    output_result = output_result[["Filename", "Group", "Delta", 'Theta', 'Alpha', "Beta", "Gamma"]]  # reorder the columns
    if output_filename:
        output_result.to_csv(output_filename)
    else:
        output_result.to_csv("PSDBands" + group1_name + group2_name + ".csv")

    return output_result

if __name__ == '__main__':
    # Parameters to modify
    group1_name = "WT"
    group2_name = "KO"
    group1_path = "WT/subthreshold responses/Spontaneous activity"
    group2_path = "KO/subthreshold responses/Spontaneous activity"
    sampling_frequency = 20000
    bands = {"Delta": [0.5, 4], "Theta": [4, 7], "Alpha": [8, 12], "Beta": [13, 30],
             "Gamma": [30, 100]}


    folders = {group1_name: group1_path, group2_name: group2_path}
    psd_grp1, freqs = psd_computation(group1_name, group1_path, sampling_frequency)
    psd_grp2, freqs = psd_computation(group2_name, group2_path, sampling_frequency)
    plot_periodogram(group1_name,group2_name,psd_grp1,psd_grp2,freqs,lim_band=[0, 100],median=True)
    output_result=computation_psd_bands(psd_grp1,psd_grp2,freqs,group1_name, group2_name,bands)