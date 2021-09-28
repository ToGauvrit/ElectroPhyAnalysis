"""
Théo Gauvrit
24/08/2021
Try to implement the Alenda et al 2010 method absed on histograms of the Membrane potential Vm
to discriminate Upstate and Downstates from resting menbrane potential recordings.
The computing of the trough is changed to the direct mean of the Vm
"""

import numpy as np
import pyabf
import scipy.stats as sc
import pandas as pd
import os
from Utils import browse_directory
import matplotlib.pyplot as plt
import time
from outliers import smirnov_grubbs as grubbs

def compute_states_discrimination(signal):

    sampling_freq=20000#Hz
    up_minimum_duration=2000#=100ms
    down_minimum_duration=1000#=50ms
    output= {"Up_duration": np.nan, "Up_Vm": np.nan, "Up_frequency": np.nan, "Down_duration": np.nan, "Down_Vm": np.nan,"Down_frequency":np.nan}

    Down_thresh, Up_thresh = compute_thresholds(signal)

    points_preupstates_index = np.where(signal > Down_thresh)[0]
    points_predownstates_index = np.where(signal < Up_thresh)[0]
    points_unclass_index = np.where( (signal < Up_thresh)  & (signal > Down_thresh))[0]
    #regroup the points above/below threshold into states by computing if there are continuous
    pre_up_states_splitted = np.split(points_preupstates_index, np.argwhere(np.diff(points_preupstates_index) != 1)[:, 0] + 1)
    pre_down_states_splitted = np.split(points_predownstates_index, np.argwhere(np.diff(points_predownstates_index) != 1)[:, 0] + 1)
    # unclass_splitted = np.split(points_unclass_index, np.argwhere(np.diff(points_unclass_index) != 1)[:, 0] + 1)
    upstates=[]
    for i in range(len(pre_up_states_splitted )):
        state=pre_up_states_splitted[i]
        state=state[np.where(signal[state] > Up_thresh)[0]]
        if state.size==0:
            continue
        state=np.arange(state[0],state[len(state)-1])
        if state.size>up_minimum_duration:
            upstates.append(state)
            up_start = state[0] / sampling_freq
            up_end = state[-1] / sampling_freq
            up_vm = np.mean(signal[state])
            up_duration = len(state) / sampling_freq
            up_vm_std = np.std(signal[state])

    downstates = []
    for i in range(len(pre_down_states_splitted)):
        state=pre_down_states_splitted[i]
        state=state[np.where(signal[state] < Down_thresh)[0]]
        if state.size == 0:
            continue
        state=np.arange(state[0], state[len(state)-1])
        if state.size>down_minimum_duration:
            downstates.append(state)
            down_start=state[0]/sampling_freq
            down_end = state[-1]/sampling_freq
            down_vm = np.mean(signal[state])
            down_duration = len(state)/sampling_freq
            down_vm_std = np.std(signal[state])
    veclen=np.vectorize(len)
    output["Up_duration"]=np.mean(veclen(upstates))/sampling_freq
    output["Up_Vm"] = np.mean(signal[np.concatenate(np.array(upstates)).ravel()])
    output["Up_frequency"] = len(upstates)/(len(signal)/sampling_freq)
    output["Down_duration"] = np.mean(veclen(downstates))/sampling_freq
    output["Down_Vm"] = np.mean(signal[np.concatenate(np.array(downstates)).ravel()])
    output["Down_frequency"] = len(downstates) / (len(signal) / sampling_freq)
    #temp ploting
    # fig, axs = plt.subplots(1, 1, figsize=(13, 20))
    # plt.plot(signal)
    # plt.plot(points_unclass_index , signal[points_unclass_index ])
    # plt.plot(upstates[0], signal[upstates[0]])
    # plt.plot(upstates[1], signal[upstates[1]])
    # plt.plot(upstates[2], signal[upstates[2]])
    # plt.plot(downstates[2], signal[downstates[2]])
    # plt.plot()

    return output

def compute_thresholds(signal, plotting=False):
    """Compute the thresholds for the Upstate and Downstate
    Slowest part of the code due to the KDE fitting"""
    print("Calcul of the thresholds!")


    mean_Vm = np.mean(signal)
    sample = np.hstack(signal)
    min_signal = np.min(signal)
    max_signal = np.max(signal)
    iqr=sc.iqr(signal)
    if max_signal-min_signal>3*iqr:
        max_signal= np.median(signal) + 2*iqr
        min_signal = np.median(signal) - 2*iqr
    # Freedman-Diaconis formula to calcul the bin width
    h=2*iqr/ np.power(len(signal), (np.divide(1,3)))
    x = np.arange(min_signal, max_signal, h)
    """the bw method permit to control the quality of the fitting, a bw too low will fit too much the 
    small flucuations. A bw method too large will make the fit not fitting the data. Here with 0.25 the fit is
    smooth and no little fluctuations will interfere.
    """
    density = sc.kde.gaussian_kde(sample,bw_method=0.25)
    fit = density(x)
    idx = (np.abs(x - mean_Vm)).argmin()
    derivative = np.gradient(fit)
    derivative = np.absolute(derivative)
    peak1_fit = np.argmax(fit[:idx])
    peak2_fit = np.argmax(fit[idx:]) + idx
    idx = fit[peak1_fit:peak2_fit].argmin()+peak1_fit
    if plotting:
        fig, axs = plt.subplots(3, 1, figsize=(13, 20))
        hist = axs[0].hist(signal, bins=round((max_signal-min_signal)/h))
        axs[0].set_xlim([min_signal, max_signal])
        # axs[1].plot(mean_Vm,fit[int(np.round(mean_Vm))],"o")
        axs[1].plot(x, fit, lw=3)
        axs[1].plot(x[idx], fit[idx], "bo")
        axs[2].plot(derivative, lw=3)
        axs[2].plot(peak1_fit, derivative[peak1_fit], "yo")
        axs[2].plot(peak2_fit, derivative[peak2_fit], "yo")
        axs[1].set_xlim([min_signal, max_signal])
        axs[2].set_xlim([0, len(derivative)])
        axs[2].plot(idx, derivative[idx], "bo")
        plt.plot()

    max_derivative_idx1 = np.argmax(derivative[peak1_fit-1:idx]) + peak1_fit
    threshold = 0.1 * derivative[max_derivative_idx1]
    if np.where(derivative[idx:peak1_fit] < threshold)[0].size==0:
        Vm_thres_idx1=idx-1
    else:
        Vm_thres_idx1 = np.where(derivative[max_derivative_idx1:idx] < threshold)[0][0] + max_derivative_idx1
    max_derivative_idx2 = np.argmax(derivative[idx:peak2_fit+1]) + idx
    threshold2 = 0.1* derivative[max_derivative_idx2]
    if np.where(derivative[idx:peak2_fit] < threshold2)[0].size==0:
        Vm_thres_idx2=idx+1
    else:
        Vm_thres_idx2 = np.where(derivative[idx:peak2_fit] < threshold2)[0][0] + idx
    if Vm_thres_idx2< Vm_thres_idx1:
        print("Threshold for upstate lower that threshold for downstates")
        raise ValueError
    if plotting:

        axs[1].plot(x[Vm_thres_idx1], fit[Vm_thres_idx1], "ro")
        axs[1].plot(x[Vm_thres_idx2], fit[Vm_thres_idx2], "ro")
        axs[2].plot(Vm_thres_idx1, derivative[Vm_thres_idx1], "ro")
        axs[2].plot(Vm_thres_idx2, derivative[Vm_thres_idx2], "ro")
        plt.plot()

    return x[Vm_thres_idx1], x[Vm_thres_idx2]
def group_UDSD_computation(group1_name,group2_name,group1_path,group2_path,output_filename=None):
    print("states computaion for " + str(group1_name) + " files")
    output_dataframe = pd.DataFrame()
    files = browse_directory(group1_path, ".abf")
    psd_list = []
    for filename in files:
        print(filename)
        abf_signal = pyabf.ABF(os.path.join(group1_path, filename))
        signal = np.array(abf_signal.sweepY)
        output1=compute_states_discrimination(signal)
        output1["Filename"]=filename
        output1["Group"]=group1_name
        output_dataframe=output_dataframe.append(output1,ignore_index=True)
    print("states computaion for " + str(group2_name) + " files")
    files = browse_directory(group2_path, ".abf")
    psd_list = []
    for filename in files:
        print(filename)
        abf_signal = pyabf.ABF(os.path.join(group2_path, filename))
        signal = np.array(abf_signal.sweepY)
        output2=compute_states_discrimination(signal)
        output2["Filename"]=filename
        output2["Group"]=group2_name
        output_dataframe = output_dataframe.append(output2, ignore_index=True)
    output_dataframe = output_dataframe[["Filename", "Group", "Up_duration", 'Up_frequency', 'Up_Vm', "Down_duration", "Down_frequency", "Down_Vm"]]  # reorder the columns

    if output_filename:
        output_dataframe.to_excel(output_filename)
    else:
        output_dataframe.to_excel("UDSDalenda" + group1_name + group2_name + ".xlsx")
    return output_dataframe

if __name__ == '__main__':
    # Parameters to modify
    group1_name = "WT"
    group2_name = "KO"
    group1_path = "WT/subthreshold responses/Spontaneous activity"
    group2_path = "KO/subthreshold responses/Spontaneous activity"
    sampling_frequency = 20000
    output_filename = "AlendaMethodUDSD.xlsx"#only excel file

    ##############To see the thresholds on histogram##############
    filename_test="18821022.abf"
    path_test="WT/subthreshold responses/Spontaneous activity"
    abf_signal = pyabf.ABF(os.path.join(path_test,filename_test ))
    signal = np.array(abf_signal.sweepY)
    # Down_thresh, Up_thresh = compute_thresholds(signal, plotting=True)

    start_time = time.time()
    output=group_UDSD_computation(group1_name, group2_name, group1_path, group2_path, output_filename=output_filename)
    print("--- %s seconds ---" % (time.time() - start_time))
    print('In minutes')
    print("--- %s minutes ---" % (time.time() - start_time/60))


    #Stats temp
    #An example of stats to read the excel file
    # df=pd.read_excel("AlendaMethodUDSD.xlsx")
    # param="Down_duration"
    # WT=df[param][df["Group"]=='WT']
    # KO_BMS=df[param][df["Group"]=='KO']
    # WTOut = grubbs.test(np.array(WT),alpha=0.05)
    # KO_BMSOut = grubbs.test(np.array(KO_BMS), alpha=0.05)
    # print("Nb outlier WT: "+str(len(WT)-len(WTOut)))
    # print("Nb outlier KO-BMS: " + str(len(KO_BMS) - len(KO_BMSOut)))
    # print(np.median(WTOut))
    # print(np.median(KO_BMSOut))
    # print(sc.shapiro(WTOut))
    # print(sc.shapiro(KO_BMSOut))
    # print(sc.ttest_ind(WTOut,KO_BMSOut))
    # print(sc.mannwhitneyu(WTOut,KO_BMSOut))