"""
Théo Gauvrit
This script contain the tools/utils that are used in other analysis scripts.
Like iterating through a directory ...
"""
import pandas as pd
import os

def browse_directory(directory_path,extension):
    """Browse a directory and read every file with the corresponding extension"""
    files = [f for f in os.listdir(os.path.join(directory_path))if os.path.isfile(os.path.join(directory_path, f))
             and f.endswith(extension)]
    return files


def merge(output_list):
    merged_df = output_list[0]
    for df in output_list[1:]:
        merged_df = pd.merge(merged_df, df, on=["Group", "Filename"])
    return merged_df


if __name__ == '__main__':
    print(browse_directory("WT_old/", ".abf"))
