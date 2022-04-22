"""Théo Gauvrit
12 Octobre 2021
Make a graph with circular layout for correlation matrices"""

import pandas as pd
import networkx as nx
import numpy as np
from outliers import smirnov_grubbs as grubbs
import scipy.stats as sc
from pyvis.network import Network
import matplotlib.pyplot as plt
import seaborn as sns
from netgraph import Graph

colorWT = "#808080"  # "#a6a6a6"
colorko = "#ff8000"  # "#cc8b00"#"#e69d00"#"#fdab00"
colorlightKO = "#ffdb8f"
colorlightWT = "#d9d9d9"
colorBms = "#0069CC"
colorlightBMS = "#cce6ff"
colorForepaw = "#00b050"
colorlightFP = "#4dff9d"
bw = 3
lwidth = 3


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def correlation_plots():
    GlobalDF = pd.read_csv("ArjunDataGlobal4.csv")
    to_exclude = ["RMSD", "Unnamed: 0", "CELL NUMBER", "CELL ID", "CELL ID spontaneous", "5th AP",
                  'AP halfwidth ratio (5/1)', "AP halfwidth ratio (3/1)", "Alpha1", "Alpha2", "EPSP SD response",
                  "Peak latency from Onset", 'Old SNR', "EPSP response var", "1st AP", "ADP amplitude", "MAE"]
    col_names = list(GlobalDF.columns)
    for label in to_exclude:
        col_names.remove(label)
        GlobalDF = GlobalDF.drop(label, axis=1)
    col_names.remove("GENOTYPE")

    dataKO = GlobalDF[GlobalDF["GENOTYPE"] == "KO"]

    dataWT = GlobalDF[GlobalDF["GENOTYPE"] == "WT"]

    ########P-value########

    def correlation_matrix(df, to_exclude):
        print(df["GENOTYPE"])
        col_names1 = list(df.columns)
        for col in to_exclude:
            col_names1.remove(col)
            df = df.drop(col, axis=1)
        pvalue_matrix = pd.DataFrame(index=col_names1, columns=col_names1)
        coeff_matrix = pd.DataFrame(index=col_names1, columns=col_names1)
        duo = []
        for (columnName, columnData) in df.iteritems():
            for (columnName1, columnData1) in df.iteritems():
                if columnName1 == columnName:
                    p_ = 1.
                    r_ = 1
                else:

                    columnData = columnData.astype(np.float)

                    columnData1 = columnData1.astype(np.float)
                    data = grubbs.test(columnData.values, alpha=0.05)
                    data1 = grubbs.test(columnData1.values, alpha=0.05)
                    col1Nan = set(np.argwhere(np.isnan(data)).flat)
                    col2Nan = set(np.argwhere(np.isnan(data1)).flat)
                    indicesNan = list(col1Nan) + list(col2Nan - col1Nan)
                    col1 = np.delete(data, indicesNan)
                    col2 = np.delete(data1, indicesNan)
                    print(columnName, columnName1)
                    print(sc.shapiro(col1))
                    print(sc.shapiro(col2))

                    r_, p_ = sc.pearsonr(col1, col2)
                    print(sc.pearsonr(col1, col2))
                    duo.append(columnName1 + "/" + columnName)
                    if p_ > 0.05:
                        p_ = 0.3
                    else:
                        p_ = 1.
                pvalue_matrix[columnName][columnName1] = p_
                coeff_matrix[columnName][columnName1] = r_
        return pvalue_matrix, coeff_matrix

    dataWT = GlobalDF[GlobalDF["GENOTYPE"] == "WT"]
    dataWT = dataWT.replace('', np.nan)
    to_exclude_WT = ["GENOTYPE", "spont.firing"]
    pValueMatrixWT, CoeffMatrixWT = correlation_matrix(dataWT, to_exclude_WT)
    dataKO = GlobalDF[GlobalDF["GENOTYPE"] == "KO"]
    dataKO = dataKO.replace('', np.nan)
    to_exclude_KO = ["GENOTYPE"]
    pValueMatrixKO, CoeffMatrixKO = correlation_matrix(dataKO, to_exclude_KO)

    corrWT = pd.melt(pValueMatrixWT.reset_index(), id_vars='index')
    coeffWT = pd.melt(CoeffMatrixWT.reset_index(),
                      id_vars='index')  # Unpivot the dataframe, so we can get pair of arrays for x and y
    # Unpivot the dataframe, so we can get pair of arrays for x and y
    corrWT.columns = ['x', 'y', 'value']

    corrKO = pd.melt(pValueMatrixKO.reset_index(), id_vars='index')
    coeffKO = pd.melt(CoeffMatrixKO.reset_index(),
                      id_vars='index')  # Unpivot the dataframe, so we can get pair of arrays for x and y
    # Unpivot the dataframe, so we can get pair of arrays for x and y
    corrKO.columns = ['x', 'y', 'value']
    return corrWT, coeffWT, corrKO, coeffKO


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
    # Graph(G,
    #       node_color=node_color, node_size=size_dic,node_edge_width=0, edge_alpha=alpha_dic,ax=ax,edge_color=color_dic,
    #       node_layout='community', node_layout_kwargs=dict(node_to_community=node_to_community),
    #       edge_layout='bundled', edge_layout_kwargs=dict(k=2000),node_labels=True,node_label_fontdict=dict(size=16),seed=19151
    # )
    Graph(G,
          node_color=node_color, node_size=size_dic, node_edge_width=0.7, node_edge_color=node_edge_color,
          edge_alpha=alpha_dic, ax=ax,
          edge_color=color_dic,
          node_layout=node_pos, edge_layout_kwargs=dict(k=2000), node_labels=True, node_label_fontdict=dict(size=16),
          seed=19151, node_label_offset=0.08, edge_width=edges_width
          )
    plt.savefig(filename)


if __name__ == '__main__':
    corrWT, coeffWT, corrKO, coeffKO = correlation_plots()
    corrWT = corrWT.drop(corrWT[corrWT["x"] == corrWT["y"]].index)
    corrKO = corrKO.drop(corrKO[corrKO["x"] == corrKO["y"]].index)
    KO = corrKO
    KO["coeff"] = coeffKO["value"]
    WT = corrWT
    WT["coeff"] = coeffWT["value"]
    WT = WT.drop(WT[WT["x"] == "max firing"].index)
    WT = WT.drop(WT[WT["y"] == "max firing"].index)
    """communities: list of list. Every list inside is a category of parameters to color
        and display together in the graph"""

    WT_communities = [
        ["STD std baseline ", "STD 200 ms psd", "STD EPSP Amp", "STD EPSP hw"],
        ["UpstateDuration", "UpstateFreq", "DownstateDuration", "DownstateValue", "DownstateFreq",
         "UpstateValue"], ["SD baseline"],
        ["AP-halfwidth", "3rd AP"],
        ["Delta", "Theta", "Alpha", "Beta", "Gamma"],
        ["Dinstein SNR", "SNR baseline"],
        ["EPSP amplitude", "EPSP halfwidth", "Epsp riseslope", "Onset latency", "Peak latency"]]
    KO_communities = [
        ["STD std baseline ", "STD 200 ms psd", "STD EPSP Amp", "STD EPSP hw"],
        ["UpstateDuration", "UpstateFreq", "DownstateDuration", "DownstateValue", "DownstateFreq",
         "UpstateValue"], ["spont.firing"], ["SD baseline"],
        ["max firing", "AP-halfwidth", "3rd AP"],
        ["Delta", "Theta", "Alpha", "Beta", "Gamma"],
        ["Dinstein SNR", "SNR baseline"],
        ["EPSP amplitude", "EPSP halfwidth", "Epsp riseslope", "Onset latency", "Peak latency"]]

    correlation_graph(WT, WT_communities, filename="WT_node_graph.pdf", radius=0.9)
    correlation_graph(KO, KO_communities, filename="KO_node_graph.pdf", radius=0.9)
