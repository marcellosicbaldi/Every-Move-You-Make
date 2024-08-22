import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def stripplot_with_lines(df, jitter=0.06, ax=None):
    """
    Create a stripplot with lines connecting the same subject across different conditions

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data to plot
    jitter : float
        Amount of jitter to apply to the data points
    ax : matplotlib.axes._subplots.AxesSubplot
        Axes where to plot the data
    """
    if ax is None:
        ax = plt.gca()
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=df.values.shape), columns=df.columns,
                                index = df.index)
    df_x_jitter += np.arange(len(df.columns))
    # Plot the markers
    for col in df_x_jitter:
        for i in range(len(df)):
            ax.scatter(df_x_jitter[col][i], df[col][i], s = 150, color = sns.color_palette("Set2")[0],
                        alpha=.6, edgecolors = "black", linewidth = 1.2)
            if col == df.columns[1]:
                ax.scatter(df_x_jitter[col][i], df[col][i], s = 150, color = sns.color_palette("Set2")[1],
                        alpha=.6, edgecolors = "black", linewidth = 1.2)
            
    for i, idx in enumerate(df.index):
        ax.plot(df_x_jitter.loc[idx,[df.columns[0], df.columns[1]]], df
                .loc[idx,[df.columns[0], df.columns[1]]], color = "k", linewidth = 0.6, alpha = 0.3, linestyle = '--', zorder = -1)
        
def stripplot_with_lines_3(df, jitter=0.06, ax=None):
    """
    stripplot_with_lines for 3 conditions
    """
    if ax is None:
        ax = plt.gca()
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=df.values.shape), columns=df.columns,
                                index = df.index)
    df_x_jitter += np.arange(len(df.columns))
    # Plot the markers
    for col in df_x_jitter:
        for i in range(len(df)):
            ax.scatter(df_x_jitter[col][i], df[col][i], s = 150, color = sns.color_palette("Set2")[0],
                        alpha=.6, edgecolors = "black", linewidth = 1.2)
            if col == df.columns[1]:
                ax.scatter(df_x_jitter[col][i], df[col][i], s = 150, color = sns.color_palette("Set2")[1],
                        alpha=.6, edgecolors = "black", linewidth = 1.2)
            if col == df.columns[2]:
                ax.scatter(df_x_jitter[col][i], df[col][i], s = 150, color = sns.color_palette("Set2")[2],
                        alpha=.6, edgecolors = "black", linewidth = 1.2)
            
    for i, idx in enumerate(df.index):
        ax.plot(df_x_jitter.loc[idx,[df.columns[0], df.columns[1]]], df
                .loc[idx,[df.columns[0], df.columns[1]]], color = "k", linewidth = 0.9, alpha = 0.3, linestyle = '--', zorder = -1)
        ax.plot(df_x_jitter.loc[idx,[df.columns[1], df.columns[2]]], df
                .loc[idx,[df.columns[1], df.columns[2]]], color = "k", linewidth = 0.9, alpha = 0.3, linestyle = '--', zorder = -1)