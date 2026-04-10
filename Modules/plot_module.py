# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# PLOT MODULE
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# Requirements setup 
import warnings
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller

# =======================================================
# Aggregate Box-plot
# =======================================================
def aggregate_box_plot(df, palette, title):
    

# General Layout (column and rows enumeration, figure's size, sub_plot)
# Sorting variables by std (so that logs variable move to the right-hand side, for better readibility)
    df = df.copy()
    Xy = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    df = df[Xy].dropna()

# Variance sorting
    statistics = pd.DataFrame({
        "var": df.var(ddof=1, skipna=True),
        "n": df.count()
    })
    statistics = statistics[statistics["n"] >= 2].sort_values("var", ascending=True)

# List of explanatory variables to plot from the general train dataset jp_aggregated_kde
    X = statistics.index.tolist()

# We define the palette following past graphs design
    pal = sns.color_palette(palette, n_colors=len(X))

# General Layout (column and rows enumeration, figure's size, sub_plot)
    cols = 7
    rows = int(np.ceil(len(X) / cols))
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(4 * cols, 5 * rows))
    axes = np.array(axes).flatten()

# Boxplot setup
    for i, col in enumerate(X):
        sns.boxplot(
            data=df,
            y=col,
            ax=axes[i],
            color=pal[i],        
            showmeans=True
        )
        axes[i].set_title(f"{col}")
        axes[i].set_xlabel("")
        axes[i].grid(True, linestyle="--", linewidth=0.7, alpha=0.7)

        for spine in axes[i].spines.values():
            spine.set_visible(False)

# Deletion of unused subplots (we have less variables than available slots for subplots on the page)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"{title}", y=1.02, fontsize=15)
    plt.tight_layout()
    plt.show()








# =======================================================
# Aggregate Violin-plot
# =======================================================
def aggregate_violin_plot(df, palette, title):

# General Layout (column and rows enumeration, figure's size, sub_plot)
# Sorting variables by std (so that logs variable move to the right-hand side, for better readibility)
    df = df.copy()
    Xy = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    df = df[Xy].dropna()

# Variance sorting
    statistics = pd.DataFrame({
        "var": df.var(ddof=1, skipna=True),
        "n": df.count()
    })
    statistics = statistics[statistics["n"] >= 2].sort_values("var", ascending=True)

# List of explanatory variables to plot from the general train dataset jp_aggregated_kde
    X = statistics.index.tolist()

# We define the palette following past graphs design
    pal = sns.color_palette(palette, n_colors=len(X))

# General Layout (column and rows enumeration, figure's size, sub_plot)
    cols = 7
    rows = int(np.ceil(len(X) / cols))
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

# Violin plot setup
    for i, col in enumerate(X):
        sns.violinplot(
            data=df,
            y=col,
            ax=axes[i],
            color=pal[i]
        )
        axes[i].set_title(f"{col}")
        axes[i].set_xlabel("")
        axes[i].grid(True, linestyle="--", linewidth=0.7, alpha=0.7)

        for spine in axes[i].spines.values():
            spine.set_visible(False)

# Deletion of unused subplots (we have less variables than available slots for subplots on the page)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"{title}", y=1.02, fontsize=15)
    plt.tight_layout()
    plt.show()








# =======================================================
# Aggregate Kernel Distribution
# =======================================================

def aggregate_kernel_distribution(df, cols, palette, title):

# Data Plotting (variable distribution with respect to the theoretical normal)
# General Layout (column and rows enumeration, figure's size, sub_plot)
    cols = cols
    df = df.copy()
    Xy = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    aggregated_kde = df[Xy].copy()
    num_vars = aggregated_kde.shape[1]
    rows = int(np.ceil(num_vars / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.ravel(axes)

# Colormap
    cmap = cm.get_cmap(palette)

# Iteration per each variable in jp_aggregated_df (var_i=location of the variable based on index, col_name = variable name)
    for i, col in enumerate(aggregated_kde.columns):
        data = aggregated_kde[col].dropna()
# Kernel density distribution of i
# We discard the NaN observations we mentioned earlier
        counts, bins = np.histogram(data, bins=30, density=True)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        widths = np.diff(bins)
# Normalized densities for color mapping
        vmin, vmax = counts.min(), counts.max()
        if np.isclose(vmin, vmax):
            colors = cmap(np.zeros_like(counts))
        else:
            norm_cmap = Normalize(vmin=vmin, vmax=vmax)
            colors = cmap(norm_cmap(counts))
# Bins definition
        axes[i].bar(bin_centers, counts, width=widths, color=colors, edgecolor="black", alpha=0.9)
# KDE
        sns.kdeplot(data, ax=axes[i], color="black", linewidth=2, label="KDE")
# Normal PDF
        mu, std = data.mean(), data.std(ddof=1)
        x = np.linspace(bins.min(), bins.max(), 200)
        axes[i].plot(x, norm.pdf(x, mu, std), "r--", label="Normal PDF")

        axes[i].set_title(f"Distribution of {col}")
        axes[i].legend()
# Deletion of unused subplots (we have less variables than available slots for subplots on the page)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"{title}", y=1.02, fontsize=15)
    plt.tight_layout()
    plt.show()






# =======================================================
# Correlation Heatmap
# =======================================================

def correlation_heatmap(df, palette, title):

# Keep only numeric columns
    df = df.copy()
    Xy = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

# Correlation matrix (restricted to selected variables)
    corr_matrix = df[Xy].corr()

# Sample size
    n = df.shape[0]

# t-statistics derived from correlation values
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat_matrix = corr_matrix * np.sqrt((n - 2) / (1 - corr_matrix**2))
        t_stat_matrix = t_stat_matrix.round(2)

# For each cell, we want to have both the correlation index, as well as the just computed t-statistics
    annot_matrix = corr_matrix.copy().astype(str)

    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
# We only want to keep the lower triangle and diagonal of the full correlation matrix
            if i >= j: 
                r = corr_matrix.iloc[i, j]
                t = t_stat_matrix.iloc[i, j]
                annot_matrix.iloc[i, j] = f"{r:.2f}\n({t:.2f})"
            else:
                annot_matrix.iloc[i, j] = ""

# We manually hide the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Heat-map plot
# General Layout (figure's size and style)
    plt.figure(figsize=(12, 10))
    sns.set(style="white")

    sns.heatmap(corr_matrix,
                mask=mask,
                annot=annot_matrix,
                fmt="",               
                cmap=palette,         
                vmin=-1, vmax=1,       
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .8},
                annot_kws={"size": 8}) 

    plt.title(f"{title}", y=1.02, fontsize=15)
    plt.tight_layout()
    plt.show()








# =======================================================
# AR(1) Unit-Root Circle Plotting 
# =======================================================

def ar1_unitroot_circle(df, palette, title):

# Unit-root Testing - Adfuller Test 
# Drop non-numeric columns and handle missing data
    df = df.copy()
    Xy = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    df = df[Xy].dropna()
# (!!!) We need to initialize the results as empty list before execuding the test
    results = []
    for col in df.columns:
        series = df[col]
# As before, we extract the AR(1) coefficients
        ar1 = series.autocorr(lag=1)
# Augmented Dickey-Fuller (ADF) unit root test 
        adf_result = adfuller(series, autolag="AIC")
        adf_stat = adf_result[0]
        p_value = adf_result[1]
        crit_values = adf_result[4]
        results.append({
            "Variable": col,
            "AR(1)": ar1,
            "ADF Statistic": adf_stat,
            "p-value": p_value,
            "Stationary - Absence of unit-root (HP1)": "Yes" if p_value < 0.05 else "No"
        })
    adf_df = pd.DataFrame(results)


# General settings & Parameters
# We extract values and parameters from the ADF unit root tests
    core_variables = adf_df["Variable"].tolist()
    ar1_values = adf_df["AR(1)"].values         
    stationary  = adf_df["Stationary - Absence of unit-root (HP1)"].tolist()
# Angle of the circle
    theta = np.linspace(0, 2 * np.pi, 100)
# Color Palette 
    colors  = [palette[i % len(palette)] for i in range(len(core_variables))]
# Figure settings
    fig = plt.figure(figsize=(13, 5.5), facecolor="#ffffff")
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.38)

# Unit Root Circle
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#ffffff")

# Plot components:
# 1) Unit circle
    ax1.plot(np.cos(theta), np.sin(theta), color="#393939", lw=1.2, zorder=1)
# 2) Vertical and horizontal axes (0,0)
    ax1.axhline(0, color="#393939", lw=0.8, zorder=0)
    ax1.axvline(0, color="#393939", lw=0.8, zorder=0)
# 3) +1, -1 Vertical Stationarity limits (-1, +1)
    ax1.axvline(1.0,  color="#ff0000", lw=0.9, linestyle="--", alpha=0.5, zorder=0)
    ax1.axvline(-1.0, color="#ff0000", lw=0.9, linestyle="--", alpha=0.5, zorder=0)
# Variables lambda on the circle 
# AR(1) coefficients live on the real axis (imaginary = 0)
    for i, (ar1, var, stat, col) in enumerate(zip(ar1_values, core_variables, stationary, colors)):
# (!!!) To plot the lambda point we can use the scatter function
        ax1.scatter(ar1, 0, color=col, s=100, zorder=4, linewidths=1.0, label=var)
# Subplot settings
# (!!!) Set .set_aspect to "equal" and not "auto" or it deforms the unit root circle into an ellipse
    ax1.set_xlim(-1.35, 1.35)
    ax1.set_ylim(-1.35, 1.35)
    ax1.set_aspect("equal")
    ax1.set_title("AR(1) Coefficients on Unit Circle\n(Imaginary = 0 for univariate series)", pad=10)
    ax1.set_xlabel("AR(1) Coefficient")
    ax1.set_ylabel("Imaginary Part (= 0)")
    ax1.tick_params(labelsize=8)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.legend(loc="lower right")

# Simulated AR(1) processes 
# Grid settings
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.38) 
    gs_right = gridspec.GridSpecFromSubplotSpec(
        len(core_variables), 1,
        subplot_spec=gs[1],
        hspace=0.55
    )
# Simulation Parameters
# Random seed
    np.random.seed(42)
# Number of steps to simulate
    N = 300
# Simulation Plotting
    for i, (var, ar1, col, stat) in enumerate(zip(core_variables, ar1_values, colors, stationary)):
        ax = fig.add_subplot(gs_right[i])
        ax.set_facecolor("#ffffff")
# Simulation of the AR(1) process: x_t = ar1 * x_{t-1} + u_t
        eps = np.random.normal(0, 1, N)
        y = np.zeros(N)
        y[0] = eps[0]
        for t in range(1, N):
            y[t] = ar1 * y[t - 1] + eps[t]
# Stem-style plot (squared dots with hortogonal lines to the 0 axis)
        markerline, stemlines, baseline = ax.stem(
            np.arange(N), y,
            linefmt=col,
            markerfmt=f"s",           
            basefmt=" "               
        )
        plt.setp(stemlines, lw=0.4, alpha=0.5, color=col)
        plt.setp(markerline, markersize=2.0, color=col, alpha=0.85)
# Horizontal zero line
        ax.axhline(0, color="#393939", lw=0.6)
# Cosmetics
        ax.set_title(
            f"{var} - AR(1)={ar1:.4f}",
            fontsize=6.8, loc="left", pad=3
        )
# ylabel: x(t)
        ax.set_ylabel("x(t)", fontsize=8, labelpad=2)
        ax.tick_params(labelsize=6)
        for spine in ax.spines.values():
            spine.set_visible(False)
# xlabel (we only plot the x label on the last bottom subplot)
        if i == len(core_variables) - 1:
            ax.set_xlabel("lag t", fontsize=8)
    plt.suptitle(f"{title}", y=1.02, fontsize=15)
    plt.show()