"""
Théo Gauvrit
Mars 2022
compute Mean Absolute Error on the evoked response(EPSP) of two groups of recordings.
"""
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabf
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
from Utils import browse_directory

def mae(abf):
    list_sweep = []
    for sweep in abf.sweepList:
        abf.setSweep(sweep, channel=1)
        data = np.array(abf.sweepY)
        list_sweep.append(data)
    average_sweep = np.mean(list_sweep, axis=0)[int(1.04 * 20000):int(1.160 * 20000)]
    resRMSD = []
    resMAE = []
    for sweep in abf.sweepList:
        abf.setSweep(sweep, channel=1)
        trial = np.array(abf.sweepY)[int(1.04 * 20000):int(1.160 * 20000)]
        mae = mean_absolute_error(average_sweep,
                                  trial,
                                  multioutput='uniform_average')
        resMAE.append(mae)
    return resMAE
    CellRes = {"Genotype": folder, "Filename": filename, "MAE": np.mean(resMAE)}
    ResDF = ResDF.append(CellRes, ignore_index=True)

def MAE_analysis():
    for folder in folders:
        for filename in os.listdir("/home/theogauvrit/Desktop/Theo2020/pyProject/StimulusResponse/" + path[folder]):
            if filename.endswith(".abf"):  # and filename in Newlist[folder]:
                abf = pyabf.ABF(path[folder] + "/" + filename)  # 14n18008.abf 18811037.abf 18821000.abf
                # if len(abf.sweepList) == 40:
                listSweep = []
                for sweep in abf.sweepList:
                    abf.setSweep(sweep, channel=1)
                    data = np.array(abf.sweepY)
                    listSweep.append(data)
                averageSweep = np.mean(listSweep, axis=0)
                # RMSD

