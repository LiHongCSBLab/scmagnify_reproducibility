"""Helper functions for single cell algorithm development."""


from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.preprocessing import MinMaxScaler

if TYPE_CHECKING:
    from typing import Literal, Union, Optional, List, Tuple
    from anndata import AnnData
    from mudata import MuData

__all__ = ["jaccard_similarity", "cosine_similarity", "plot_overlaps", "plot_scatter_with_error_bars", "show_color_dict", "get_tab20_colors_dict", "plot_boxplot", "plot_scatter_with_error_bars", "flatten_dict_values"]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity

def jaccard_similarity(data_dict, 
                       ax=None, 
                       fmt=".02f", 
                       figsize=(3.5, 3),
                       title=None,
                       save=None,
                       **kwargs):
    """
    Compute the Jaccard similarity between multiple sets of data and visualize the results as a heatmap.
    
    Parameters
    ----------
    data_dict : dict
        A dictionary where each key-value pair represents the name of a dataset and the data itself, respectively.
    figsize : tuple, optional (default: (4, 4))
        The size of the figure to create.
    title : str, optional (default: "Jaccard Similarity")
        The title of the plot.
    ax : matplotlib Axes, optional (default: None)
        The Axes on which to draw the plot. If None, a new figure and Axes will be created.
    fmt : str, optional (default: ".02f")
        The format string for the annotations in the heatmap.
    save : str, optional (default: None)
        The path to save the figure. If None, the figure will not be saved.
    
    Returns
    -------
    jaccard_matrix : np.ndarray
        A matrix containing the pairwise Jaccard similarities between the datasets.
    fig : matplotlib Figure or None
        The Figure on which the plot was drawn, or None if `ax` was provided.
    """
    
    # 1. 提取每个数据集的唯一元素，并将其转换为集合
    data_names = list(data_dict.keys())
    data_sets = [set(data) for data in data_dict.values()]

    # 2. 初始化一个空的矩阵用于存储成对的 Jaccard 相似度
    n = len(data_sets)
    jaccard_matrix = np.zeros((n, n))

    # 3. 计算成对的 Jaccard 相似度
    for i, j in combinations(range(n), 2):
        A = data_sets[i]
        B = data_sets[j]
        intersection = len(A.intersection(B))
        union = len(A.union(B))
        jaccard_similarity = intersection / union if union != 0 else 0
        jaccard_matrix[i, j] = jaccard_similarity
        jaccard_matrix[j, i] = jaccard_similarity

    # 对角线填充为 NaN（因为一个数据集与自身的相似度为 1）
    np.fill_diagonal(jaccard_matrix, np.NaN)

    # 4. 可视化 Jaccard 矩阵为热图
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = None
        
    overlaps_df = pd.DataFrame(jaccard_matrix, columns=data_names, index=data_names)
    ax = plot_overlaps(overlaps_df, 
                  feature_colors=None, 
                  ax=ax,
                  **kwargs)
    
    if title is not None:
        ax.set_title(title, pad=20)
    else:
        ax.set_title("Jaccard Similarity", pad=20)

    # 保存图像
    if save is not None:
        plt.savefig(save, bbox_inches='tight')

    return jaccard_matrix, ax

def cosine_similarity(data_dict, 
                      ax=None, 
                      figsize=(3.5, 3), 
                      fmt=".02f", 
                      title=None,
                      save=None,
                      **kwargs):
    """
    Compute the cosine similarity between multiple sets of data and visualize the results as a heatmap.
    
    Parameters
    ----------
    data_dict : dict
        A dictionary where each key-value pair represents the name of a dataset and the data itself, respectively.
    figsize : tuple, optional (default: (4, 4))
        The size of the figure to create.
    title : str, optional (default: "Cosine Similarity")
        The title of the plot.
    ax : matplotlib Axes, optional (default: None)
        The Axes on which to draw the plot. If None, a new figure and Axes will be created.
    fmt : str, optional (default: ".02f")  
        The format string for the annotations in the heatmap.
    
    Returns
    -------
    cosine_sim_matrix : np.ndarray
        A matrix containing the pairwise cosine similarities between the datasets.
    fig : matplotlib Figure or None
        The Figure on which the plot was drawn, or None if `ax` was provided.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 1. 提取每个数据集的唯一元素，并将其转换为集合
    data_names = list(data_dict.keys())
    data_sets = [set(data) for data in data_dict.values()]
    
    # 2. 创建一个包含所有可能链接（source-target）的并集，以形成一个“向量空间”
    all_links = set()
    for data in data_sets:
        all_links.update(data)
    all_links = sorted(list(all_links))  # Sort to keep consistent ordering
    
    # 3. 创建一个矩阵，其中每个条目表示每个数据集的链接权重
    weighted_matrix = np.zeros((len(data_sets), len(all_links)))
    for i, data in enumerate(data_sets):
        data_to_weight = {link: 1 for link in data}
        for j, link in enumerate(all_links):
            weighted_matrix[i, j] = data_to_weight.get(link, 0)  # Use 0 if link not present in data
    
    # 4. 计算基于权重的所有数据集之间的成对余弦相似度
    cosine_sim_matrix = cosine_similarity(weighted_matrix)
    for i in range(cosine_sim_matrix.shape[0]):
        for j in range(cosine_sim_matrix.shape[1]):
            if i==j:
                cosine_sim_matrix[i,j]=np.NaN
    
    # 5. 可视化余弦相似度矩阵为热图
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = None
    
    overlaps_df = pd.DataFrame(cosine_sim_matrix, columns=data_names, index=data_names)
    ax = plot_overlaps(overlaps_df,
                          feature_colors=None,
                          ax=ax,
                          **kwargs)
    if title is not None:
        ax.set_title(title, pad=20)
    else:
        ax.set_title("Cosine Similarity", pad=20)

    # 保存图像
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    
    return cosine_sim_matrix, ax

#### Plotting functions 

def plot_overlaps(
    overlaps: pd.DataFrame,
    feature_colors: dict = None,
    labels: List[str] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
):
    # if feature_colors is None:
    #     # tab10 colors
    #     feature_colors = get_tab20_colors_dict(overlaps.columns.tolist())
    # Plot overlaps
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    # vmin & vmax
    if "vmin" not in kwargs:
        kwargs["vmin"] = 0 if overlaps.min().min() >= 0 else None
        
    if "vmax" not in kwargs:
        kwargs["vmax"] = overlaps.max().max()  if overlaps.max().max() <= 1 else None

    if "labelsize" in kwargs:
        labelsize = kwargs.pop("labelsize")
        x_pad = labelsize
        y_pad = labelsize * 1.5
    else:
        labelsize = 10
        x_pad = labelsize
        y_pad = labelsize * 1.5
    
    sns.heatmap(
        overlaps,
        cmap="Reds",
        cbar_kws={"shrink": 0.2, "aspect": 6},
        linewidths=0.1,
        ax=ax,
        **kwargs
    )
    n = overlaps.shape[0]
    ax.hlines(range(n+1), -1, n+0.1, color="w", lw=3, clip_on=False)

    # Adjust tick labels
    ax.tick_params(which="major", length=0, labelsize=labelsize)
    ax.tick_params(axis="x", pad=x_pad)
    ax.tick_params(axis="y", pad=y_pad)
    if labels is None:
        labels = overlaps.columns.tolist()
    ax.set_yticklabels(labels, rotation=0)
    ax.set_xticks(np.arange(len(labels))+0.75)  # offset to align with center
    ax.set_xticklabels(labels, rotation=45, ha="right", va="top")

    # Add row and column color annotations
    if feature_colors is not None:
        for i, f in enumerate(overlaps):
            color = feature_colors[f]
            kwargs = dict(
                fill=True, facecolor=color, lw=1.5, edgecolor="w",
                clip_on=False, zorder=0
            )
            p = 0.075
            row_color = plt.Rectangle(
                (-p, i), p*0.75, 1, transform=ax.get_yaxis_transform(), **kwargs
            )
            ax.add_patch(row_color)
            p = 0.055
            col_color = plt.Rectangle(
                (i, -p), 1, p, transform=ax.get_xaxis_transform(), **kwargs
            )
            ax.add_patch(col_color)

    return ax