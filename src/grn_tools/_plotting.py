import os
import sys

import numpy as np
import pandas as pd

import mplscience
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from typing import List, Optional

__all__ = ["show_color", "plot_metrics", "plot_precision_recall", "plot_overlaps", "plot_pie_charts", "plot_violin", "plot_boxplot", "plot_paired_boxplot", "plot_horizontal_boxplot", "plot_line", "show_color_dict", "get_tab20_colors_dict", "quartile_to_level", "get_kde", "flatten_dict_values", "plot_scatter_with_error_bars"]

def plot_metrics(dfs, algo_list):
    
    dfs = {metric: dfs[metric].loc[dfs[metric].Algorithm.isin(algo_list) & dfs[metric].Lineage.isin(lin_list)] for metric in dfs.keys()}
    n_algo = len(algo_list)
    n_metrics = len(dfs)

    # palette = dict(zip(algo_list, sns.color_palette("colorblind").as_hex()[:n_algo]))
    palette = dict(zip(algo_list, [f"C{i}" for i in range(n_algo)]))

    with mplscience.style_context():
        fig, ax = plt.subplots(figsize=(10 * n_metrics, 6), ncols=n_metrics)

        for ax_id, metric in enumerate(dfs.keys()):
            _df = dfs[metric]
            _df["x_combined"] = _df["Lineage"] + "\n" + _df["Dataset"]
            sns.barplot(data=_df,
                        x="x_combined",
                        y="metric",
                        hue="Algorithm",
                        palette=palette,
                        ax=ax[ax_id],
                        )
            
            #ax[ax_id].set_title(metric)
            if ax_id == 0:
                ax[ax_id].set_ylabel(metric)
                handles, labels = ax[ax_id].get_legend_handles_labels()
            #ax[ax_id].set_xlabel("Lineage")
            ax[ax_id].text(0, -0.03, "PMID36973557_NatBiotechnol2023_CD34", )
            ax[ax_id].text(2.6, -0.03, "PMID36973557_NatBiotechnol2023_T-cell-depleted", )
            
            ax[ax_id].get_legend().remove()
            ax[ax_id].set_ylabel(metric)
            ax[ax_id].set_xticklabels(['Ery', 'Mono', 'CLP', 'Ery', 'Mono', 'NaiveB'])
            ax[ax_id].set_xlabel(None)
            
            #ax[ax_id].set_xticklabels(ax[ax_id].get_xticklabels(), rotation=45, ha='right')
        fig.legend(handles=handles, labels=labels, loc="lower center", ncol=n_algo, bbox_to_anchor=(0.5, -0.06))
        plt.tight_layout()

def plot_precision_recall(dfs, algo_list, lin_list):

    dfs = {lin: dfs[lin].loc[dfs[lin].Algorithm.isin(algo_list)] for lin in lin_list}

    n_algo = len(algo_list)
    n_lin = len(lin_list)

    palette = dict(zip(algo_list, sns.color_palette("colorblind").as_hex()[:n_algo]))

    with mplscience.style_context():
        sns.set_style("whitegrid")

        fig, ax = plt.subplots(figsize=(6 * n_lin, 4), ncols=n_lin)

        for ax_id, lin in enumerate(lin_list):
            _df = dfs[lin]
            sns.lineplot(data=_df, 
                         x="Recall", 
                         y="Precision", 
                         hue="Algorithm", 
                         ax=ax[ax_id]
                         )
            
            ax[ax_id].set_title(lin)
            if ax_id == 0:
                ax[ax_id].set_ylabel("Precision")
                handles, labels = ax[ax_id].get_legend_handles_labels()
            ax[ax_id].set_xlabel("Recall")
            ax[ax_id].get_legend().remove()
            ax[ax_id].set_ylim(-0.05, 0.4)
            ax[ax_id].set_xlim(-0.05, 0.4)

    #handles = [handles[0], handles[1], handles[2], handles[5], handles[4], handles[3]]
    #labels = [labels[0], labels[1], labels[2], labels[5], labels[4], labels[3]]
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=n_algo, bbox_to_anchor=(0.5, -0.15))
    plt.tight_layout()
    plt.show()
    

def plot_overlaps(
    overlaps: pd.DataFrame,
    feature_colors: dict = None,
    labels: List[str] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
):
    if feature_colors is None:
        # tab10 colors
        feature_colors = get_tab20_colors_dict(overlaps.columns.tolist())
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
        linewidths=1,
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

def plot_pie_charts(annot_df, colors, ncols=None, figsize=(16, 6), startangle=60, **kwargs):
    """
    Plot pie charts for each row in the input DataFrame.

    Parameters
    ----------
    annot_df : pd.DataFrame
        A DataFrame where each row represents a set of counts for the pie chart.
        The index of the DataFrame will be used as the title for each pie chart.
    colors : list
        A list of colors to be used for the pie chart wedges.
    ncols : int, optional (default: None)
        The number of columns in the subplot grid. If None, the number of columns
        will be set to the number of rows in the DataFrame.
    figsize : tuple, optional (default: (16, 6))
        The size of the figure to create.
    startangle : float, optional (default: 60)
        The starting angle for the first wedge in the pie chart.
    **kwargs : dict
        Additional keyword arguments to be passed to `plt.subplots`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Figure object containing the pie charts.
    axes : np.ndarray
        An array of Axes objects, one for each pie chart.
    """
    # Determine the number of rows and columns for subplots
    nrows = len(annot_df.index)
    if ncols is None:
        ncols = nrows  # Default to one column per row
    nrows = (nrows + ncols - 1) // ncols  # Calculate the number of rows needed

    # Create subplots with the specified number of columns
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    axes = np.array(axes).flatten()  # Flatten the axes array for easy iteration

    # Iterate over pie charts
    for i, (ax, idx) in enumerate(zip(axes, annot_df.index)):
        if i >= len(annot_df.index):  # Hide unused axes
            ax.axis("off")
            continue

        ax.set_title(idx, fontsize=12, pad=10)
        counts = annot_df.loc[idx]
        wedges, _ = ax.pie(counts, colors=colors, wedgeprops=dict(width=0.3), startangle=startangle)

        # Add annotations to the wedges
        for j, wedge in enumerate(wedges):
            if counts[j] == 0 or counts[j] < 1:  # Skip annotation for values less than 1%
                continue
            # Calculate angle and position for the annotation
            ang = (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1
            x = np.cos(np.deg2rad(ang))
            y = np.sin(np.deg2rad(ang))
            ax.annotate(f"{counts[j]:.1f}%", xy=(x, y), xytext=(1.25 * x, 1.25 * y),
                        ha="center", va="center", fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="none", ec="none"))

    # Hide unused axes if there are fewer rows than the grid size
    for ax in axes[len(annot_df.index):]:
        ax.axis("off")

    # Adjust spacing and add legend
    plt.subplots_adjust(wspace=0.2, hspace=0.5)

    # Create a single legend outside the pie charts
    patches = [mpatches.Patch(color=color, label=name) for color, name in zip(colors, annot_df.columns)]
    fig.legend(handles=patches, loc="center right", bbox_to_anchor=(1.15, 0.5), fontsize=10)

    return fig, axes

def plot_violin(
    data, x, y,
    ax=None,
    bw_adjust=1,
    grid_size=100,
    h=2,
    norm=True,
    min_q=0.01,
    lw=0.2,
    **kwargs
):
    # Calculate the median for each group
    medians = data.groupby(x)[y].median().sort_values(ascending=True)
    n_groups = len(medians)
    x_ticks = np.linspace(0, -h * (n_groups - 1), n_groups)
    
    sns.set_style("white")
    
    if "palette" in kwargs:
        palette = kwargs.pop("palette")
    else:
        palette = get_tab20_colors_dict(medians.index)
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    else:
        fig = None
        
    for i, (name, xpos) in enumerate(zip(medians.index, x_ticks)):
        group = data[data[x] == name]
        # Violin Plot
        n = group.shape[0]
        bw_method = group.shape[0] ** (-1/(5)) * bw_adjust
        pos, est = get_kde(group[y], min_q=min_q, bw_method=bw_method)
        if norm:
            est = MinMaxScaler().fit_transform(est.reshape(-1,1)).flatten()
        ax.fill_betweenx(
            pos, xpos - est, xpos + est,
            facecolor=palette[name], alpha=0.75, 
            lw=0.5, edgecolor="black",
            zorder=1,
        )
        ax.plot(xpos + est, pos, lw=lw, color="w", zorder=1)
        ax.plot(xpos - est, pos, lw=lw, color="w", zorder=1)
        # IQR Plot
        ymin, ymid, ymax = group[y].quantile([0.25, 0.5, 0.75])
        ax.plot([xpos, xpos], [ymin, ymax], lw=0.8, color="black", alpha=1)
        ax.scatter([xpos], [ymid], s=2, color="black", alpha=1)
    
    if "spline_linewidth" in kwargs:
        spline_linewidth = kwargs.pop("spline_linewidth")
        ax.spines["top"].set_linewidth(spline_linewidth)
        ax.spines["bottom"].set_linewidth(spline_linewidth)
        ax.spines["left"].set_linewidth(spline_linewidth)
        ax.spines["right"].set_linewidth(spline_linewidth)
    # Set xticks and xticklabels based on sorted medians
    
    
    if "xticklabels" in kwargs:
        ax.set_xticks([])
        xticklabels = kwargs.pop("xticklabels")
        ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=10)
    else:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(medians.index, rotation=45, ha="right", fontsize=10)

    return ax

def plot_boxplot(data, x, y, figsize=(4, 3), ax=None, **kwargs):
    
    if "palette" in kwargs:
        palette = kwargs.pop("palette")
    else:
        palette = get_tab20_colors_dict(data[x].unique())
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = None
        
    if "spline_linewidth" in kwargs:
        spline_linewidth = kwargs.pop("spline_linewidth")
        
    
    sns.boxplot(
        x=x, y=y, data=data, palette=palette, ax=ax, **kwargs
    )
    ax.spines["top"].set_linewidth(spline_linewidth)
    ax.spines["bottom"].set_linewidth(spline_linewidth)
    ax.spines["left"].set_linewidth(spline_linewidth)
    ax.spines["right"].set_linewidth(spline_linewidth)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    
    return ax


def plot_paired_boxplot(data, x, y, subject_id, figsize=(4, 3), ax=None, **kwargs):
    """
    Draw a paired boxplot with connecting lines for repeated measurements.

    Parameters:
    - data: pandas DataFrame containing the data.
    - x: str, name of the categorical column (group/condition).
    - y: str, name of the numerical value column.
    - subject_id: str, name of the column identifying paired samples (e.g., subject ID).
    - figsize: tuple, size of the figure.
    - ax: matplotlib Axes object to draw on.
    - kwargs: additional arguments passed to seaborn.boxplot.
    """
    if "palette" in kwargs:
        palette = kwargs.pop("palette")
    else:
        palette = sns.color_palette("Set2", n_colors=len(data[x].unique()))

    # Handle order of x axis
    if "order" in kwargs:
        x_order = kwargs["order"]
    elif pd.api.types.is_categorical_dtype(data[x]):
        x_order = data[x].cat.categories
    else:
        x_order = sorted(data[x].unique())
    kwargs["order"] = x_order

    spline_linewidth = kwargs.pop("spline_linewidth", 1.5)
    line_alpha = kwargs.pop("line_alpha", 0.3)
    line_width = kwargs.pop("line_width", 1.0)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    # 1. Draw paired lines (background)
    grouped = data.pivot(index=subject_id, columns=x, values=y)
    grouped = grouped[[col for col in x_order if col in grouped.columns]]

    for _, row in grouped.iterrows():
        if row.notna().sum() >= 2:
            ax.plot(range(len(row)), row.values, color="gray",
                    alpha=line_alpha, linewidth=line_width, zorder=0)

    # 2. Draw boxplot
    sns.boxplot(x=x, y=y, data=data, palette=palette, ax=ax, zorder=1, fliersize=0, **kwargs)

    # 3. Draw stripplot
    sns.stripplot(x=x, y=y, data=data, palette=palette, ax=ax,
                  size=3, alpha=0.6, dodge=True, zorder=2, jitter=0.2)

    # 4. Styling
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(spline_linewidth)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    return ax

def plot_horizontal_boxplot(
    data, x, y, figsize=(7, 3), palette="vlag", show_legend=False, ax=None
):
    """
    Draw a horizontal boxplot with overlaid stripplot points.
    
    Parameters:
    - data: pd.DataFrame, the input data.
    - x: str, name of the numeric column (e.g., "AUPR").
    - y: str, name of the categorical column (e.g., "Algorithm").
    - figsize: tuple, figure size.
    - palette: color palette for the boxplot.
    - show_legend: bool, whether to show the legend.
    - ax: matplotlib Axes object. If None, a new figure is created.
    
    Returns:
    - ax: the matplotlib Axes object with the plot.
    """
    sns.set_theme(style="ticks")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    # Boxplot
    sns.boxplot(
        data=data, x=x, y=y, hue=y,
        width=0.6, palette=palette, ax=ax, fliersize=0
    )

    # Stripplot
    sns.stripplot(
        data=data, x=x, y=y,
        color=".3", size=4, jitter=True, dodge=True, alpha=0.5, ax=ax
    )

    # Tweak presentation
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set(ylabel="", xlabel=x)

    if not show_legend:
        ax.legend_.remove()

    sns.despine(trim=True, left=True)
    
    return ax



def plot_scatter_with_error_bars(data, x_metric, y_metric, **kwargs):
    """
    Plot a scatter plot with error bars for the given metrics.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the data to plot.
    x_metric : str
        The column name for the x-axis metric.
    y_metric : str
        The column name for the y-axis metric.
    palette : list, optional
        A list of colors to use for the plot.
    spline_linewidth : float, optional
        Linewidth for the plot spines.
    xlim : tuple, optional
        Limits for the x-axis.
    ylim : tuple, optional
        Limits for the y-axis.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object containing the scatter plot.
    """
    
    required_columns = ["Algorithm", x_metric, y_metric]
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    
    if data.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Calculate summary statistics
    data_summ = data.groupby("Algorithm").agg({
        x_metric: ["mean", "std"],
        y_metric: ["mean", "std"]
    }).reset_index()
    data_summ.columns = ["Algorithm", "XMean", "XSD", "YMean", "YSD"]
    
    if data_summ.empty:
        raise ValueError("Summary DataFrame is empty after aggregation")
    
    # Extract additional plotting parameters
    spline_linewidth = kwargs.pop("spline_linewidth", 1.0)
    palette = kwargs.pop("palette", get_tab20_colors_dict(data["Algorithm"].unique()))
    xlim = kwargs.pop("xlim", [data[x_metric].min() - 0.01, data[x_metric].max() + 0.01])
    ylim = kwargs.pop("ylim", [data[y_metric].min() - 0.01, data[y_metric].max() + 0.01])
    
    # Create the scatter plot with error bars
    ax = sns.scatterplot(
        data=data_summ,
        x="XMean",
        y="YMean",
        hue="Algorithm",
        s=25,
        edgecolor="black",
        linewidth=0.8,
        zorder=2,
        palette=palette,
        **kwargs
    )

    # Add error bars
    for _, row in data_summ.iterrows():
        ax.errorbar(
            x=row["XMean"],
            y=row["YMean"],
            xerr=row["XSD"],
            yerr=row["YSD"],
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=0.8,
            capthick=0.8,
            zorder=1,
        )
        
    # Set plot limits and labels
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    
    ticks = np.linspace(xlim[0], xlim[1], 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    ax.set_xlabel(x_metric, fontsize=12)
    ax.set_ylabel(y_metric, fontsize=12)
    ax.tick_params(labelsize=10)
    
    for spine in ax.spines.values():
        spine.set_linewidth(spline_linewidth)

    # Remove legend title and adjust legend position
    ax.legend(title=None, loc="best", bbox_to_anchor=(1, 1), fontsize=8)
    
    return ax


def plot_line(metrics_df, 
              x="Algorithm", 
              y="AUPR", 
              figsize=(8, 4), 
              spline_linewidth=0.8, 
              palette=None,
              ylim=[0.18, 0.25],
              save=None,
              **kwargs):
    """
    Plot a line plot for the given metrics, with algorithms on the x-axis and metrics on the y-axis.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        A DataFrame containing the data to plot.
    x : str, optional (default: "Algorithm")
        The column name for the x-axis (algorithms).
    y : str, optional (default: "AUPR")
        The column name for the y-axis (metrics).
    figsize : tuple, optional (default: (8, 4))
        The size of the figure.
    spline_linewidth : float, optional (default: 0.8)
        Linewidth for the plot spines.
    palette : dict or list, optional
        A dictionary or list of colors to use for the plot.
    **kwargs : dict
        Additional keyword arguments passed to `sns.lineplot`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object containing the line plot.
    """
    
    # 验证输入数据
    required_columns = [x, y]
    if not all(col in metrics_df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    
    if metrics_df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # 设置颜色
    if palette is None:
        palette = sns.color_palette("tab10", n_colors=len(metrics_df[x].unique()))
    
    # 创建图形
    plt.figure(figsize=figsize)
    ax = sns.lineplot(
        data=metrics_df,
        x=x,
        y=y,
        hue=x,  # 使用算法作为 hue
        palette=palette,
        marker="o",  # 添加标记点
        markersize=8,  # 标记点大小
        linewidth=2,  # 线条宽度
        **kwargs
    )
    
    # 设置图形样式
    ax.set_xlabel(x, fontsize=12)
    ax.set_ylabel(y, fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_ylim(ylim[0], ylim[1]) # 设置 y 轴范围
    
    # 设置 spine 线宽
    for spine in ax.spines.values():
        spine.set_linewidth(spline_linewidth)
    
    # 调整图例
    # ax.legend(title=None, loc="best", bbox_to_anchor=(1, 1), fontsize=8)
    ax.legend([])
    
    # 自动调整布局
    plt.tight_layout()
    
    if save is not None:
        plt.savefig(save, bbox_inches="tight")
    
    return ax

def show_color_dict(color_dict, figsize=(4, 4)):
    """Show the color dictionary as a legend."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for i, (name, color) in enumerate(color_dict.items()):
        # Use 'facecolor' instead of 'color' to avoid the warning
        ax.add_patch(
            plt.Rectangle((0, i), 1, 1, facecolor=color, lw=1.5, edgecolor="w")
        )
        ax.text(1.1, i + 0.5, name, va="center", fontsize=8)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, len(color_dict))
    ax.axis("off")
    fig.show()

"""Plotting Utilities"""
def get_tab20_colors_dict(names=None):

    # 获取 tab10 颜色列表
    tab20_colors = plt.cm.tab10.colors

    # 如果没有提供名称，则使用默认的索引作为键名
    if names is None:
        names = [f"color{i}" for i in range(len(tab20_colors))]

    # 如果名称和颜色数量不匹配，截取colors列表
    n = min(len(names), len(tab20_colors))
    tab20_colors = tab20_colors[:n]

    # 生成颜色字典
    color_dict = {name: color for name, color in zip(names, tab20_colors)}
    return color_dict

def quartile_to_level(data, quantile):
    """Return data levels corresponding to quantile cuts of mass."""
    isoprop = np.asarray(quantile)
    values = np.ravel(data)
    sorted_values = np.sort(values)[::-1]
    normalized_values = np.cumsum(sorted_values) / values.sum()
    idx = np.searchsorted(normalized_values, 1 - isoprop)
    levels = np.take(sorted_values, idx, mode="clip")
    return levels

def get_kde(
    data,
    grid_size=500,
    min_q=0.0,
    **kwargs
):
    from scipy.stats import gaussian_kde
    
    kernel = gaussian_kde(data, **kwargs)
    positions = np.linspace(data.min(), data.max(), grid_size)
    estimate = kernel(positions)
    level = quartile_to_level(estimate, min_q)
    mask = estimate>=level
    return positions[mask], estimate[mask]-level


def flatten_dict_values(d):
    """
    Flatten the values of a dictionary. 
    
    Parameters
    ----------
    d : dict
        A dictionary to be flattened.
    
    Returns
    -------
    flat_values : list
        A list of the values of the dictionary.
    """
    flat_values = []
    for value in d.values():
        if isinstance(value, dict): 
            flat_values.extend(flatten_dict_values(value))
        else:  
            flat_values.append(value)
            
    flat_list = []
    for sublist in flat_values:
        flat_list.extend(sublist)
    return flat_list


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import is_color_like, Colormap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import is_color_like, Colormap

def show_color(color_input, title=None, n_colors=256):
    """
    Intelligently visualizes a color input.
    - If the input is a single color, it displays a block of that color.
    - If the input is a list (a palette), it displays the color blocks side-by-side.
    - If the input is a colormap (cmap), it displays its continuous color gradient.
    - If the input is a dictionary, it iterates through items, using keys as titles
      and values as the color input to display.

    Args:
        color_input (str, tuple, list, dict, or matplotlib.colors.Colormap):
            Can be a single color like 'red', '#FF0000', or (1, 0, 0);
            a palette (list) like ['red', 'blue', 'green'];
            a colormap name/object like 'viridis';
            or a dictionary where keys are titles and values are the colors/palettes/cmaps.
        title (str, optional): The title for the plot. If None, a default is generated.
            (Note: This is ignored if color_input is a dictionary).
        n_colors (int, optional): The number of colors to use when rendering a cmap gradient. Defaults to 256.
    """
    # --- NEW: Case 4: Handle a dictionary of colors ---
    if isinstance(color_input, dict):
        for key_title, value_color in color_input.items():
            # Call the function recursively for each item, using the key as the title
            show_color(value_color, title=key_title)
        return # Done with the dictionary

    # --- Case 1 & 2: Handle a single color or a palette (list of colors) ---
    colors_to_show = []
    default_title = ""

    if is_color_like(color_input):
        colors_to_show = [color_input]
        default_title = "Single Color"
    elif isinstance(color_input, list):
        colors_to_show = color_input
        default_title = "Color Palette"

    if colors_to_show:
        n = len(colors_to_show)
        plt.figure(figsize=(max(n, 2), 1))
        for i, color in enumerate(colors_to_show):
            plt.fill_between([i, i + 1], 0, 1, color=color)
        plt.xlim(0, n)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title(title if title else default_title, fontsize=16)
        plt.show()
        return

    # --- Case 3: Handle a colormap (cmap) ---
    try:
        if isinstance(color_input, str):
            cmap_name = color_input
        elif isinstance(color_input, Colormap):
            cmap_name = color_input.name
        else:
            raise ValueError("Input type is not a list, dict, color-like, or Colormap")

        cmap = plt.get_cmap(color_input)
        gradient = np.linspace(0, 1, n_colors)
        gradient = np.vstack((gradient, gradient))

        plt.figure(figsize=(8, 1.5))
        plt.imshow(gradient, aspect='auto', cmap=cmap)
        plt.axis('off')
        final_title = title if title else f"Colormap: '{cmap_name}'"
        plt.title(final_title, fontsize=16)
        plt.show()
        return

    except (ValueError, TypeError) as e:
        print(f"Input '{color_input}' could not be identified as a valid color, palette, or cmap. Error: {e}")