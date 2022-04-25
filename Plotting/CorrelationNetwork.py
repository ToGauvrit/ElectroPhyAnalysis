



"""Théo Gauvrit
12 Octobre 2021
Make a graph with circular layout for correlation matrices"""

import pandas as pd
import networkx as nx
import numpy as np
import scipy.stats as sc
from pyvis.network import Network
import matplotlib.pyplot as plt
import seaborn as sns
from netgraph import Graph

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 40
plt.rcParams['axes.linewidth'] = 3

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


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
                column_data = column_data.astype(np.float)
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
                print(column_name+ " "+str(len(col1)))
                print(column_name1 + " " + str(len(col2)))
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


def heatmap(x, y, size, color, filename):
    n_colors = 256  # Use 256 colors for the diverging color palette
    palette = sns.diverging_palette(20, 220, n=n_colors)
    palette.reverse()  # Create the palette
    color_min, color_max = [-1,1]  # Range of values for the palette, i.e. min and max possible correlation

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


def correlation_graph(corrDF, communities, filename, radius):
    """
    Make the circular graph from the correlation and coef dataframe
    Arguments:
    ----------
    corrDF: dataframe. Adjency matrix + coeff correlation and a value to know if it's significant or no
    communities: list of list. Every list inside is a category of parameters to color
        and display together in the graph
    """
    n_colors = 256  # Use 256 colors for the diverging color palette
    palette = sns.diverging_palette(20, 220, n=n_colors)
    palette.reverse()  # Create the palette
    color_min, color_max = [-1, 1]

    def value_to_color(val):
        val_position = float((val - color_min)) / (
                color_max - color_min)  # position of value in the input range, relative to the length of the input range
        ind = int(val_position * (n_colors - 1))  # target index in the color palette
        return palette[ind]

    size_min = 0
    size_max = 18
    n_size = 10
    list_size = np.linspace(3, 15, n_size)

    def value_to_size(val):
        val_position = float((val - size_min)) / (size_max - size_min)
        ind = int(val_position * (n_size - 1))
        return list_size[ind]

    width_min = 0.20
    width_max = 1
    n_width = 10
    list_width = np.linspace(0.001, 4, n_width)

    def value_to_widths(val):
        val = np.absolute(val)
        val_position = float((val - width_min)) / (width_max - width_min)
        ind = int(val_position * (n_width - 1))
        return list_width[ind]

    # color of edges
    G = nx.from_pandas_edgelist(corrDF, source="x", target="y", edge_attr='coeff')
    color_dic = {}
    alpha_dic = {}
    edges_width = {}
    for row in corrDF.iterrows():
        print(row[1]["value"])
        if row[1]["value"] < 1:
            color_edge = "#0000ffff"  # make transparent every correlation not significant
            alpha_edge = 0.0
        else:
            color_edge = value_to_color(row[1]["coeff"])
            alpha_edge = 0.6
        color_dic[(row[1]["x"], row[1]["y"])] = color_edge
        alpha_dic[(row[1]["x"], row[1]["y"])] = alpha_edge
        edges_width[(row[1]["x"], row[1]["y"])] = value_to_widths(row[1]["coeff"])

    # node size
    size_dic = {}
    corrDFnode = corrDF.drop(corrDF[corrDF["value"] < 1].index)
    corrDFnode = corrDFnode.drop(corrDFnode[corrDFnode["x"] == corrDFnode["y"]].index)
    Gnode = nx.from_pandas_edgelist(corrDFnode, source="x", target="y", edge_attr='value')
    for node in Gnode.degree:
        size_dic[node[0]] = value_to_size(node[1])
        if value_to_size(node[1]) < size_min:
            size_dic[node[1]] = size_min
    #
    # if  "ADP amplitude"not in  size_dic: #exception fo KO
    #     size_dic["ADP amplitude"] = list_size[0]
    # node pos
    node_pos = {}
    n = len(np.concatenate(communities))
    r = radius
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = r * np.cos(t)
    y = r * np.sin(t)
    for id, node in enumerate(np.concatenate(communities)):
        node_pos[node] = np.array([x[id], y[id]])

    # node community
    node_to_community = dict()
    for id, node in enumerate(G.nodes()):
        print(node)
        for community_id, commu in enumerate(communities):
            if node in commu:
                node_to_community[node] = community_id

    community_to_color = {
        0: 'tab:blue',
        1: 'tab:orange',
        2: 'tab:red',
        3: 'tab:cyan',
        4: 'tab:grey',
        5: 'tab:olive',
        6: 'tab:brown',

    }
    if len(communities) == 8:
        community_to_color = {
            0: 'tab:blue',
            1: 'tab:orange',
            2: 'tab:green',
            3: 'tab:red',
            4: 'tab:cyan',
            5: 'tab:grey',
            6: 'tab:olive',
            7: 'tab:brown',

        }
    node_color = {node: adjust_lightness(community_to_color[community_id], 1.2) for node, community_id in
                  node_to_community.items()}
    node_edge_color = {node: community_to_color[community_id] for node, community_id in node_to_community.items()}
    # plotting
    fig, ax = plt.subplots(figsize=(15, 15))
    Graph(G,
          node_color=node_color, node_size=size_dic, node_edge_width=0.7, node_edge_color=node_edge_color,
          edge_alpha=alpha_dic, ax=ax,
          edge_color=color_dic,
          node_layout=node_pos, edge_layout_kwargs=dict(k=2000), node_labels=True, node_label_fontdict=dict(size=16),
          seed=19151, node_label_offset=0.08, edge_width=edges_width
          )
    plt.savefig(filename)


if __name__ == '__main__':

    # KO BMS
    # parameters_data = pd.read_excel("2022.04.19 - Combined Data Individual Cells.xlsx",sheet_name="FmKO Individual Cells")
    # data_kobms = parameters_data[parameters_data["GENOTYPE"] =="BMS-KO "]
    # data_kobms = data_kobms.replace('', np.nan)
    # to_exclude_kobms = ["GENOTYPE", "Unnamed: 0", "CELL NUMBER", "CELL ID", "CELL ID spontaneous"
    #                     , "AP halfwidth ratio (3/1)", "Alpha1", "Alpha2",
    #                  "Peak latency from Onset", "1st AP"]
    # pvalue_kobms, coeff_kobms = correlation_matrix(data_kobms, to_exclude_kobms)
    # corrKOBMS= pd.melt(pvalue_kobms.reset_index(), id_vars='index') # Unpivot the dataframe, to get pair of arrays for x and y
    # coeffBMS = pd.melt(coeff_kobms.reset_index(), id_vars='index')
    # corrKOBMS.columns = ['x', 'y', 'value']
    # heatmap(
    #     x=corrKOBMS['x'],
    #     y=corrKOBMS['y'],
    #     size=corrKOBMS['value'].abs(),
    #     color=coeffBMS['value'],
    #     filename="KOBMS_corrmatrix.pdf"
    # )
    #
    # KOBMS = corrKOBMS
    # KOBMS["coeff"] = coeffBMS["value"]
    # KOBMS = KOBMS.drop(KOBMS[KOBMS["x"] == KOBMS["y"]].index)
    # KOBMS = KOBMS.drop(KOBMS[KOBMS["x"] == "EPSP halfwidth"].index)
    # KOBMS = KOBMS.drop(KOBMS[KOBMS["y"] == "EPSP halfwidth"].index)
    # KOBMS = KOBMS.drop(KOBMS[KOBMS["x"] == "MAE"].index)
    # KOBMS = KOBMS.drop(KOBMS[KOBMS["y"] == "MAE"].index)
    # KOBMS_communities = [
    #     ["STD std baseline ", "STD 200 ms psd", "STD EPSP Amp", "STD EPSP hw"],
    #     ["UpstateDuration", "UpstateFreq", "DownstateDuration", "DownstateValue", "DownstateFreq","up-state down-state difference (mV)",
    #      "UpstateValue"], ["SD baseline"],
    #     ["AP-halfwidth", "3rd AP", "max firing"],
    #     ["Delta", "Theta", "Alpha", "Beta", "Gamma"],
    #     ["Dinstein SNR", "SNR baseline"],
    #     ["EPSP amplitude", "Epsp riseslope", "Onset latency", "Peak latency"]]
    # correlation_graph(KOBMS, KOBMS_communities, filename="KOBMS_node_graph.pdf", radius=0.9)

    # WT BMS
    parameters_data = pd.read_excel("2022.04.19 - Combined Data Individual Cells.xlsx",sheet_name="FmKO Individual Cells")
    data_wtbms = parameters_data[parameters_data["GENOTYPE"] =="WT-BMS"]
    data_wtbms = data_wtbms.replace('', np.nan)
    to_exclude_wtbms = ["GENOTYPE", "Unnamed: 0", "CELL NUMBER", "CELL ID", "CELL ID spontaneous"
                        , "AP halfwidth ratio (3/1)", "Alpha1", "Alpha2","Epsp riseslope","Dinstein SNR",
                     "Peak latency from Onset", "1st AP"]
    pvalue_wtbms, coeff_wtbms = correlation_matrix(data_wtbms, to_exclude_wtbms)
    corrwtbms= pd.melt(pvalue_wtbms.reset_index(), id_vars='index') # Unpivot the dataframe, to get pair of arrays for x and y
    coeffwtbms = pd.melt(coeff_wtbms.reset_index(), id_vars='index')
    corrwtbms.columns = ['x', 'y', 'value']
    heatmap(
        x=corrwtbms['x'],
        y=corrwtbms['y'],
        size=corrwtbms['value'].abs(),
        color=coeffwtbms['value'],
        filename="WTBMS_corrmatrix.pdf"
    )

    WTBMS = corrwtbms
    WTBMS["coeff"] = coeffwtbms["value"]
    # dropping some parameters with not enought n numbers to do corr
    WTBMS = WTBMS.drop(WTBMS[WTBMS["x"] == WTBMS["y"]].index)
    WTBMS = WTBMS.drop(WTBMS[WTBMS["x"] == "MAE"].index)
    WTBMS = WTBMS.drop(WTBMS[WTBMS["y"] == "MAE"].index)
    WTBMS = WTBMS.drop(WTBMS[WTBMS["x"] == "AP-halfwidth"].index)
    WTBMS = WTBMS.drop(WTBMS[WTBMS["y"] == "AP-halfwidth"].index)
    WTBMS = WTBMS.drop(WTBMS[WTBMS["x"] == "max firing"].index)
    WTBMS = WTBMS.drop(WTBMS[WTBMS["y"] == "max firing"].index)
    WTBMS = WTBMS.drop(WTBMS[WTBMS["x"] == "DownstateFreq"].index)
    WTBMS = WTBMS.drop(WTBMS[WTBMS["y"] == "DownstateFreq"].index)
    WTBMS = WTBMS.drop(WTBMS[WTBMS["x"] == "EPSP halfwidth"].index)
    WTBMS = WTBMS.drop(WTBMS[WTBMS["y"] == "EPSP halfwidth"].index)
    WTBMS = WTBMS.drop(WTBMS[WTBMS["x"] == "DownstateDuration"].index)
    WTBMS = WTBMS.drop(WTBMS[WTBMS["y"] == "DownstateDuration"].index)
    WTBMS = WTBMS.drop(WTBMS[WTBMS["x"] == "UpstateDuration"].index)
    WTBMS = WTBMS.drop(WTBMS[WTBMS["y"] == "UpstateDuration"].index)
    WTBMS_communities = [
        ["STD std baseline ", "STD 200 ms psd", "STD EPSP Amp", "STD EPSP hw"],
        ["UpstateFreq","DownstateValue",
         "up-state down-state difference (mV)","UpstateValue"],
        ["SD baseline"],
        ["AP-halfwidth", "3rd AP"],
        ["Delta", "Theta", "Alpha", "Beta", "Gamma"],
        ["Dinstein SNR", "SNR baseline"],
        ["EPSP amplitude", "Onset latency", "Peak latency"]]
    correlation_graph(WTBMS, WTBMS_communities, filename="WTBMS_node_graph.pdf", radius=0.9)


