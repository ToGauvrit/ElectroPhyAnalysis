"""
Théo Gauvrit
Mars 2022
Function to create from a excel/csv data file the correlation matrices and correlation nodes and heat map plots
"""
import pandas as pd
import networkx as nx
import numpy as np
from outliers import smirnov_grubbs as grubbs
import scipy.stats as sc
from pyvis.network import Network
import matplotlib.pyplot as plt
import seaborn as sns
from netgraph import Graph

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 40
plt.rcParams['axes.linewidth'] = 3


def heatmap(x, y, size, color, filename):
    n_colors = 256  # Use 256 colors for the diverging color palette
    palette = sns.diverging_palette(20, 220, n=n_colors)
    palette.reverse()  # Create the palette
    color_min, color_max = [-1,
                            1]  # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        val_position = float((val - color_min)) / (
                color_max - color_min)  # position of value in the input range, relative to the length of the input range
        ind = int(val_position * (n_colors - 1))  # target index in the color palette
        return palette[ind]

    plot_grid = plt.GridSpec(1, 24, hspace=0.2, wspace=0.1)  # Setup a 1x15 grid
    fig = plt.figure(figsize=(30, 25))
    ax = plt.subplot(plot_grid[:, :-1])
    # Mapping from column names to integer coordinates
    x_labels = [v for v in list(x.unique())]
    y_labels = [v for v in list(y.unique())]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}
    size_scale = 900
    ax.scatter(
        x=x.map(x_to_num),  # Use mapping for x
        y=y.map(y_to_num),  # Use mapping for y
        s=(size * size_scale).astype(float),  # Vector of square sizes, proportional to size parameter
        c=color.apply(value_to_color),
        marker='s'  # Use square as scatterplot marker
    )
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.tick_params(which='both', width=3)
    ax.tick_params(which='major', length=10)
    ax = plt.subplot(plot_grid[:, -1])  # Use the rightmost column of the plot
    col_x = [1.5] * len(palette)  # Fixed x coordinate for the bars
    bar_y = np.linspace(color_min, color_max, n_colors)  # y coordinates for each of the n_colors bars

    bar_height = bar_y[1] - bar_y[0]
    ax.barh(
        y=bar_y,
        width=[2] * len(palette),  # Make bars 5 units wide
        left=col_x,  # Make bars start at 0
        height=bar_height,
        color=palette,
        linewidth=0
    )
    ax.set_xlim(1, 3)  # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
    ax.grid(False)  # Hide grid
    ax.set_facecolor('white')  # Make background white
    ax.set_xticks([])  # Remove horizontal ticks
    ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))  # Show vertical ticks for min, middle and max
    ax.yaxis.tick_right()  # Show vertical ticks on the right
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(which='both', width=3)
    ax.tick_params(which='major', length=10)
    fig.subplots_adjust(bottom=0.30, left=0.30, right=0.95, top=0.99)
    fig.savefig(filename)


def correlation_matrix(all_parameters_data, parameters_to_exclude):
    print(all_parameters_data["GENOTYPE"])
    col_names1 = list(all_parameters_data.columns)
    for col in parameters_to_exclude:
        print(col)
        col_names1.remove(col)
        all_parameters_data = all_parameters_data.drop(col, axis=1)
    pvalue_matrix = pd.DataFrame(index=col_names1, columns=col_names1)
    coeff_matrix = pd.DataFrame(index=col_names1, columns=col_names1)
    duo = []
    for (column_name, column_data) in all_parameters_data.iteritems():
        for (column_name1, column_data1) in all_parameters_data.iteritems():
            if column_name1 == column_name:
                p_ = 1.
                r_ = 1
            else:
                print(column_name)
                column_data = column_data.astype(np.float)
                print(column_name1)
                column_data1 = column_data1.astype(np.float)
                # data = grubbs.test(column_data.values, alpha=0.05)
                # data1 = grubbs.test(column_data1.values, alpha=0.05)
                data = column_data.values
                data1 = column_data1.values
                col1_nan = set(np.argwhere(np.isnan(data)).flat)
                col2_nan = set(np.argwhere(np.isnan(data1)).flat)
                indices_nan = list(col1_nan) + list(col2_nan - col1_nan)
                col1 = np.delete(data, indices_nan)
                col2 = np.delete(data1, indices_nan)
                print(column_name, column_name1)
                print(sc.shapiro(col1))
                print(sc.shapiro(col2))
                print(col1)
                print(col2)
                r_, p_ = sc.pearsonr(col1, col2)
                print(sc.pearsonr(col1, col2))
                duo.append(column_name1 + "/" + column_name)
                if p_ > 0.05:
                    p_ = 0.3
                else:
                    p_ = 1.
            pvalue_matrix[column_name][column_name1] = p_
            coeff_matrix[column_name][column_name1] = r_
    return pvalue_matrix, coeff_matrix


def correlation_nodes_graph():
    pass


if __name__ == '__main__':
    # parameters_data = pd.read_csv("ArjunDataGlobal4.csv")
    # data_wt = parameters_data[parameters_data["GENOTYPE"] == "WT"]
    # data_wt = data_wt.replace('', np.nan)
    # to_exclude_wt = ["GENOTYPE", "RMSD", "Unnamed: 0", "CELL NUMBER", "CELL ID", "CELL ID spontaneous", "5th AP",
    #                  'AP halfwidth ratio (5/1)', "AP halfwidth ratio (3/1)", "Alpha1", "Alpha2", "EPSP SD response",
    #                  "Peak latency from Onset", 'Old SNR', "EPSP response var", "1st AP", "ADP amplitude", "MAE",
    #                  "spont.firing"]
    # pvalue_wt, coeff_wt = correlation_matrix(data_wt, to_exclude_wt)
    # corr = pd.melt(pvalue_wt.reset_index(), id_vars='index') # Unpivot the dataframe, to get pair of arrays for x and y
    # coeff = pd.melt(coeff_wt.reset_index(), id_vars='index')
    # corr.columns = ['x', 'y', 'value']
    # heatmap(
    #     x=corr['x'],
    #     y=corr['y'],
    #     size=corr['value'].abs(),
    #     color=coeff['value'],
    #     filename="WTcorrelationPlotReduced.pdf"
    # )
    parameters_data = pd.read_excel("2022.04.19 - Combined Data Individual Cells.xlsx",sheet_name="FmKO Individual Cells")
    data_kobms = parameters_data[parameters_data["GENOTYPE"] =="WT-BMS"]
    data_kobms = data_kobms.replace('', np.nan)
    to_exclude_kobms = ["GENOTYPE", "Unnamed: 0", "CELL NUMBER", "CELL ID", "CELL ID spontaneous"
                        , "AP halfwidth ratio (3/1)", "Alpha1", "Alpha2",
                     "Peak latency from Onset", "1st AP"]
    pvalue_kobms, coeff_kobms = correlation_matrix(data_kobms, to_exclude_kobms)
    corr = pd.melt(pvalue_kobms.reset_index(), id_vars='index') # Unpivot the dataframe, to get pair of arrays for x and y
    coeff = pd.melt(coeff_kobms.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']
    heatmap(
        x=corr['x'],
        y=corr['y'],
        size=corr['value'].abs(),
        color=coeff['value'],
        filename="WTBMS_corrmatrix.pdf"
    )
