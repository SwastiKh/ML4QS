import sys
import os
# Add the absolute path to the 'plot-misc' folder
sys.path.insert(0, '/gpfs/work2/0/aus20644/project/ksharma/brugada/plot-misc')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plot_misc.utils as pm_utils
import seaborn as sns
from plot_misc import volcano
from IPython.display import display
from project_constants import Project_paths as local_paths
from project_constants import Constant as local_c
from project_constants import (
    OUTCOME_ORDER,
    OUTCOMES,
    EXPOSURES,
    EXPOSURE_ORDER,
    CLASSES,
    CLASS_COL,
    )

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Load data
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
metab = pd.read_csv('/gpfs/work2/0/aus20644/project/ksharma/brugada/python_scripts/metabolite_results_n5.tsv', sep = '\t', header = 0, index_col = 0,
    low_memory = False).drop_duplicates()
# Step 1: Convert scientific notation strings to numeric
metab['pvalue'] = pd.to_numeric(
    metab['P-value - Supported'].astype(str).str.replace('×10⁻', 'e-', regex=False),
    errors='coerce'
)
card = pd.read_csv('/gpfs/work2/0/aus20644/project/ksharma/brugada/data/cardiac_results_n5.tsv', sep = '\t', header = 0, index_col = 0).drop_duplicates()
card.loc[card['P-value (float) - Supported'] == 0, 'pvalue'] = 1e-100
card.loc[:, 'pvalue_log10'] = -1*np.log10(card['P-value (float) - Supported'])
card.loc[card['pvalue_log10'] > 16, 'pvalue_log10'] = 16

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Constants
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

plt.ion()
# The y, x axes labels,title, point size, text size,
# ytick size, x ticks size
ANNOT_SIZE = [6.0, 6.0, 5.0, 5.0, 3.4, 4.0, 4.0]
# using standard colours
COLOURS = ('#1191FA','grey','#FC6039')
# setting size (in cm)
FIG_SIZE = [float(r) * local_c.cmtoinch for r in  [4.5,5.5]]
# print(metab.columns.tolist())
# print(metab.head())

# setting threshold
SIGNIFICANCE = metab['pvalue_strict'].unique()[0]
YLIM = [0, 17]
# Truncate p-values
# Tick size
TL = 3
# plot defaults, line between bins
plt.rcParams["patch.force_edgecolor"] = True
sns.set(style = 'ticks')
sns.set_context('paper')

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Cardiac plots
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Adjusting constants
# print(card.columns.tolist())
# print(card.head())
SIGNIFICANCE = card['pvalue_strict'].unique()[0]

# making xlim dict
vol_dict = {
    'brugada' : [-1.5, 1.5],
    'QRS'     : [-0.75, 0.75]
}   

# making subplots
fig, axes = plt.subplots(1, 2,
        figsize=(10 * local_c.cmtoinch, 10 * local_c.cmtoinch),  # more square
        sharex=False, sharey=True, constrained_layout=False)

plt.subplots_adjust(wspace=0.25, hspace=0.28)  

# figure dict and plotting order
order = np.array([x for x in OUTCOME_ORDER if x in card['Outcome'].unique()])

for i, s in enumerate(order):
    # subsetting
    subset = card[card['Outcome'] == s].copy()

    # dynamic xlim
    x_min = subset['Point estimate'].min() - 0.1 * abs(subset['Point estimate'].min())
    x_max = subset['Point estimate'].max() + 0.1 * abs(subset['Point estimate'].max())
    axes[i].set_xlim(x_min, x_max)

    # dynamic ylim
    y_min = 0
    y_max = subset['pvalue_log10'].max() * 1.1
    axes[i].set_ylim(y_min, y_max)

    # get indices to plot
    text_idx = subset.loc[subset["Outcome"] == s, "uniprot_display_label"].unique()
    subset.set_index('uniprot_display_label', inplace=True)
    subset['uniprot_display_label'] = subset.index.copy()

    _, _ = volcano.plot_volcano(subset, y_column='pvalue_log10',
            x_column='Point estimate', fsize=FIG_SIZE,
            alpha=SIGNIFICANCE,
            col=COLOURS, xlab='', ylab='', ylim=(y_min, y_max), msize=ANNOT_SIZE[3],
            ax=axes[i],
            label_kwargs_dict = {
                'force_text': (0.1, 0.5),
                'lim': 1500,
                'precision': 0.1,
                'only_move': {'points': 'xy', 'text': 'xy', 'objects': 'xy'},
                'expand_text': (1.05, 1.15)
            }
        )

    # set title above the subplot
    axes[i].set_title(str(OUTCOMES[s]), fontsize=ANNOT_SIZE[2]+2,
                      fontweight='bold', pad=15)  # pad moves title above plot

    axes[i].set_xlabel('')  
    axes[i].xaxis.labelpad = 5

    # tick font size
    axes[i].tick_params(axis='y', labelsize=ANNOT_SIZE[5]+2, length=TL)
    axes[i].tick_params(axis='x', labelsize=ANNOT_SIZE[6]+2, length=TL)

    print('Finished: ', str(s))

# common axis-labels
ylab = r'$-log_{10}(pvalue)$'
fig.text(0.06, 0.5, ylab, va='center', rotation='vertical', fontsize=ANNOT_SIZE[0]+2)
fig.text(0.44, 0.03, 'Effect size [log(OR)]', va='center', rotation='horizontal', fontsize=ANNOT_SIZE[0]+2)

# save plot
plt.savefig('/gpfs/work2/0/aus20644/project/ksharma/brugada/plots/cardiac_volcano.pdf', bbox_inches='tight')
plt.close('all')
