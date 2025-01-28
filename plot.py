import torch
import numpy
import h5py
import matplotlib.pyplot as plt
import numpy as np
import math as m

from typing import List, Optional, Tuple, Union

from variable_autoregression.util import pearson_correlation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_one_metric(
        data: List,
        data_std: List,
        legends: List,
        x_label: List,
        x_label_rotate: float = 0,
        color: List = [],
        width_bar: float = 0.1,
        x_title: str = "",
        y_title: str = "",
        legend_title: str = "",

        text_fontsize: int = 10,
        text_vspacing: float = 0,
        
        y_lim: List = None,
        y_scale: str = "linear",

        is_save: bool = False,
        save_loc: str = None,
        save_format: str = None,


):

  # Option 1: Return figure
    fig = plt.figure()  # Uncomment to return the figure


    x = np.arange(len(x_label))

    bars = []
    for i in range(len(legends)):
        shift = shift = i*width_bar
        bars.append(plt.bar(x+ shift, data[i], color=color[i], yerr=data_std[i], capsize=7, zorder=10, label=str(legends[i]), width=width_bar))


    for i in range(len(data)):
        bar_heights = data[i]
        shift = i*width_bar
        for j, height in enumerate(bar_heights):
            plt.text(j+shift, height + text_vspacing + data_std[i][j] , r"$" + str(np.round(height, 2) ) + "$", ha='center', va='bottom', fontsize=text_fontsize-2)

    plt.yscale(y_scale)
    plt.xlabel(x_title, fontsize=text_fontsize+2 )
    plt.ylabel(y_title, fontsize=text_fontsize+2 )
  
    plt.xticks(x + width_bar/len(legends), x_label, fontsize=text_fontsize-2, rotation=x_label_rotate)  # Rotate x-axis labels for better readability (optional)

    plt.grid(axis='y', zorder=0)  # Add gridlines (optional)
    plt.ylim(y_lim)
    # Display the legend (optional)

    legend = plt.legend(loc ="upper center", ncol=2, fontsize=text_fontsize-2, title=legend_title)
    legend.get_title().set_fontsize(str(text_fontsize-1))


    if is_save:
        plt.tight_layout()
        plt.savefig(save_loc, format=save_format)
        

    return fig
