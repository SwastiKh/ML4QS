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
from math import ceil
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

metab = pd.read_csv('/gpfs/work2/0/aus20644/project/ksharma/brugada/python_scripts/metabolite_results_n5.tsv', sep = '\t', header = 0, index_col = 0,
    low_memory = False).drop_duplicates()
# Step 1: Convert scientific notation strings to numeric
# Step 1: Replace zeros with a tiny number to avoid log10(0)
metab['pvalue'] = metab['P-value (float) - Supported'].replace(0, 1e-100)

# Step 2: Take -log10 of the cleaned p-values
metab['pvalue_log10'] = -np.log10(metab['pvalue'])


# Truncate p-values
metab.loc[metab['pvalue_log10'] > 16, 'pvalue_log10'] = 16
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
## Metabolite plots
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Use SIGNIFICANCE threshold to subset significant metabolites 
# sig_metab = metab[metab['pvalue'] < SIGNIFICANCE].copy()
# # Normalize Outcome values in the data
# sig_metab['Outcome_clean'] = sig_metab['Outcome'].astype(str).str.strip().str.lower(
# Normalize Outcome values in the data
metab['Outcome_clean'] = metab['Outcome'].astype(str).str.strip().str.lower()


# Normalize EXPSOURE_ORDER
EXPOSURE_ORDER_clean = [x.lower() for x in EXPOSURE_ORDER]

# Build order only if it exists in both EXPOSURE_ORDER and CLASSES
order = [x for x in EXPOSURE_ORDER_clean if x in metab['Outcome_clean'].values and x in CLASSES]


# Fix the CLASSES dictionary first
if pd.isna(CLASSES.get('hexose')):
    CLASSES['hexose'] = 'hexose'

# Fix CLASS_COL for hexose
if 'hexose' not in CLASS_COL:
    CLASS_COL['hexose'] = '#FFA500'  # choose a color you like

# Make xlim and color dict for each metabolite
vol_dict = {}
for s in order:
    if s not in CLASSES:
        print("Skipping missing CLASSES key:", s)
        continue

    # Assign colors based on class
    cls = CLASSES[s]
    if cls in ['Amino acids', 'Acylcarnitines']:
        mid_col = '#004285'
    else:
        mid_col = COLOURS[2]

    vol_dict[s] = [[-5, 5], [CLASS_COL[cls], COLOURS[1], mid_col]]

# Parameters for multi-page plotting
plots_per_page = 30
rows, cols = 5, 6  # 5x6 grid = 30 plots per page
num_pages = ceil(len(order) / plots_per_page)

pdf = PdfPages('/gpfs/work2/0/aus20644/project/ksharma/brugada/plots/metabolite_volcano.pdf')

# Loop through pages
for page in range(num_pages):
    start_idx = page * plots_per_page
    end_idx = min(start_idx + plots_per_page, len(order))
    chunk = order[start_idx:end_idx]
    
    fig, axes = plt.subplots(rows, cols,
                             figsize=(25 * local_c.cmtoinch, 20 * local_c.cmtoinch),
                             sharex=False, sharey=True)
    axes = axes.flatten()
    
    for i, s in enumerate(chunk):
        subset = metab[metab['Outcome_clean'] == s].copy()
        subset['Point estimate'] = subset['Point estimate']
        # Prepare indices for labels
        text_idx = subset['uniprot_display_label'].unique()
        subset.set_index('uniprot_display_label', inplace=True)
        subset['uniprot_display_label'] = subset.index.copy()
        
        # Plot volcano
        _, _ = volcano.plot_volcano(subset, y_column='pvalue_log10',
                                    x_column='Point estimate', fsize=FIG_SIZE, alpha=SIGNIFICANCE,
                                    col=vol_dict[s][1], xlab='', ylab='', ylim=YLIM,
                                    msize=ANNOT_SIZE[3], ax=axes[i],
                                    index_label=text_idx,
                                    lsize=ANNOT_SIZE[4])
        
        axes[i].set_title(str(EXPOSURES[s]), loc='left', y=.94,
                          fontsize=ANNOT_SIZE[2], fontdict={'fontweight': 'bold'})
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='y', labelsize=ANNOT_SIZE[5], length=TL)
        axes[i].tick_params(axis='x', labelsize=ANNOT_SIZE[6], length=TL)
    
    # Remove unused axes
    for j in range(len(chunk), len(axes)):
        fig.delaxes(axes[j])
    
    # Common axis labels
    fig.text(0.09, 0.5, r'$-log_{10}(pvalue)$', va='center', rotation='vertical', fontsize=ANNOT_SIZE[0])
    fig.text(0.48, 0.07, 'Effect size (MD)', va='center', rotation='horizontal', fontsize=ANNOT_SIZE[0])
    
    # Save current page
    pdf.savefig(fig)
    plt.close(fig)

pdf.close()