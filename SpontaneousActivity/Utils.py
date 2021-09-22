"""
Théo Gauvrit
Spetembre 2021
This script contain the tools/utils that are used in other analysi scripts.
Like going through a directory ...
"""
import pyabf
import numpy as np
import pandas as pd
import os
import time
import seaborn as sns
import scipy.stats as sc

def browse_directory(directory_path,extension):
    """Browse a directory and read every file with the corresponding extension"""
    files=[f for f in os.listdir(os.path.join(directory_path))if os.path.isfile(os.path.join(directory_path, f))
           and f.endswith(extension)]
    return files


if __name__ == '__main__':
    print(browse_directory("WT_old/",".abf"))