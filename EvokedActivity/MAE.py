"""
Théo Gauvrit
Mars 2022
compute Mean Absolute Error on the evoked response(EPSP) of two groups of recordings.
"""
import numpy as np
import pandas as pd
import pyabf
from sklearn.metrics import mean_absolute_error

from Utils import browse_directory


def mae(group_name, group_path, sampling_freq, stim_timing=1.04, window_duration=0.12):
    print("MAE computaion for " + str(group_name) + " files")
    output_dataframe = pd.DataFrame()
    files = browse_directory(group_path, ".abf")
    for filename in files:
        print(filename)
        abf = pyabf.ABF(group_path + "/" + filename)
        list_sweep = []
        for sweep in abf.sweepList:
            abf.setSweep(sweep, channel=1)
            data = np.array(abf.sweepY)
            list_sweep.append(data)
        average_sweep = np.mean(list_sweep, axis=0)[
                        int(stim_timing * sampling_freq):int((stim_timing + window_duration) * sampling_freq)]
        res_mae = []
        for sweep in abf.sweepList:
            abf.setSweep(sweep, channel=1)
            trial = np.array(abf.sweepY)[
                    int(stim_timing * sampling_freq):int((stim_timing + window_duration) * sampling_freq)]
            mae_value = mean_absolute_error(average_sweep,
                                            trial,
                                            multioutput='uniform_average')
            res_mae.append(mae_value)
        cell_res = {"Group": group_name, "Filename": filename, "MAE": np.mean(res_mae)}
        output_dataframe = output_dataframe.append(cell_res, ignore_index=True)
    return output_dataframe


if __name__ == '__main__':
    group1_path = "/run/user/1004/gvfs/afp-volume:host=engram.local,user=Theo%20Gauvrit,volume=Data/Yukti/In Vivo Patch Clamp Recordings/Evoked Responses_FmKO"
    output_df = mae(group_name="KO DMSO", group_path=group1_path, sampling_freq=20000)
    output_df.to_excel("MAE.xlsx")
