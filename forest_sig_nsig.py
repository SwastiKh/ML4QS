#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Imports
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import sys
import os
# Add the absolute path to the 'plot-misc' folder
sys.path.insert(0, '/gpfs/work2/0/aus20644/project/ksharma/brugada/plot-misc')
import unicodedata
import math
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
import matplotlib
import re
from matplotlib import colors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import plot_misc.utils.utils as pm_utils
from plot_misc.utils.colour import Colours as pm_col
import plot_misc.forest as forest
import plot_misc.heatmap as heatmap
from itertools import cycle
from IPython.display import display  # Updated import per DeprecationWarning
from project_constants import Project_paths as local_paths
from project_constants import Constant as local_c
from project_constants import (
    OUTCOME_ORDER,
    OUTCOMES,
    EXPOSURES,
    EXPOSURE_ORDER,
    CLASSES,
    CLASS_COL,
    CLASS_ORDER
)

plt.ion()
matplotlib.rcParams['axes.linewidth'] = .3
COL = pm_col()

# =========================
# 1) Load BRUGADA + prep
# =========================
fin_b = pd.read_csv(
    '/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/TableX_combined_test.tsv',
    sep='\t', header=0
)
pre_b = pd.read_csv(
    '/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/cardiac_results_n5_p2.04_05.tsv',
    sep='\t', header=0, index_col=0
)
# tidy column names
pre_b.columns = pre_b.columns.str.strip()

# map to display outcome label
pre_b['format_out'] = pre_b['Outcome'].map(OUTCOMES)

# keep needed columns for Brugada merge
pre_b_sub = pre_b[['uniprot_id', 'Outcome', 'uniprot_display_label', 'OR', 'LCI', 'UCI']].copy()

# Merge to attach OR/CI info to fin_b
fin_b = fin_b.merge(
    pre_b_sub,
    left_on=['uniprot_id', 'Protein name', 'cardiac cis Outcome'],
    right_on=['uniprot_id', 'uniprot_display_label', 'Outcome'],
    how='left'
)

# Build out_b and met_b
out_b = fin_b[['Protein name', 'cardiac cis Outcome', 'OR', 'LCI', 'UCI']].drop_duplicates()
out_b.columns = ['protein', 'outcome', 'OR', 'LCI', 'UCI']

met_b = fin_b[['Protein name', 'GW Metabolite', 'Metabolite class', 'cardiac cis Outcome']].drop_duplicates()
met_b.columns = ['protein', 'metabolite', 'class', 'outcome']

# =========================
# 2) Load QRS + prep
# =========================
fin_q = pd.read_csv(
    '/gpfs/work2/0/aus20644/project/ksharma/brugada/qrs_forest.tsv',
    sep='\t', header=0
)
pre_q = pd.read_csv(
    '/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/cardiac_results_n5.tsv',
    sep='\t', header=0, index_col=0
)
# Normalize and map outcome display names
pre_q.columns = pre_q.columns.str.strip()
pre_q['format_out'] = pre_q['Outcome'].map(OUTCOMES).fillna(pre_q['Outcome'])

# Keep only QRS duration rows
pre_q_qrs = pre_q[pre_q['format_out'].eq('QRS duration')].copy()

# Ensure fin_q has the join key set consistently
fin_q['cardiac cis Outcome'] = 'QRS duration'

# Merge the relevant columns from pre_q_qrs into fin_q — include P-value for completeness if present
merge_cols_q = ['uniprot_display_label', 'format_out', 'Point estimate', 'Lower bound', 'Upper bound']
if 'P-value' in pre_q_qrs.columns:
    merge_cols_q.append('P-value')

fin_q = fin_q.merge(
    pre_q_qrs[merge_cols_q],
    left_on=['Protein name', 'cardiac cis Outcome'],
    right_on=['uniprot_display_label', 'format_out'],
    how='left'
)

# Build out_q (unique per protein) and metabolite mapping met_q
out_q = fin_q[['Protein name', 'format_out', 'Point estimate', 'Lower bound', 'Upper bound']].drop_duplicates()
out_q.columns = ['protein', 'outcome', 'Point estimate', 'Lower bound', 'Upper bound']

# Build met_q from fin_q
met_q = fin_q[['Protein name', 'GW Metabolite', 'Metabolite class', 'cardiac cis Outcome']].drop_duplicates()
met_q.columns = ['protein', 'metabolite', 'class', 'outcome']

# =========================
# 3) Keep only proteins present in BOTH (POST-MERGE labels)
# =========================
common_proteins = (
    set(out_b['protein'].astype(str).str.strip())
    & set(out_q['protein'].astype(str).str.strip())
)
print("Common proteins (post-merge):", len(common_proteins))

# filter both tables and metabolites to common proteins
out_b = out_b[out_b['protein'].astype(str).str.strip().isin(common_proteins)].copy()
met_b = met_b[met_b['protein'].astype(str).str.strip().isin(common_proteins)].copy()
out_q = out_q[out_q['protein'].astype(str).str.strip().isin(common_proteins)].copy()
met_q = met_q[met_q['protein'].astype(str).str.strip().isin(common_proteins)].copy()

if out_b.empty or out_q.empty:
    raise RuntimeError("After intersecting proteins, one panel has no data. Check your inputs.")

# =========================
# 4) Colors / shapes
# =========================
# Brugada colours
col_dic_b = {}
pal_b = [COL.colsweetie[5], COL.colsweetie[11], COL.colpico8[10]]
for i, o in enumerate(out_b['outcome'].unique()):
    col_dic_b[o] = pal_b[i % len(pal_b)]
out_b['col'] = out_b['outcome'].map(col_dic_b)
out_b['shape'] = 'o'
out_b['a_col'] = 1

# QRS colours
col_dic_q = {}
pal_q = [COL.colsweetie[5], COL.colsweetie[11], COL.colpico8[10]]
for i, o in enumerate(out_q['outcome'].unique()):
    col_dic_q[o] = pal_q[i % len(pal_q)]
out_q['col'] = out_q['outcome'].map(col_dic_q)
out_q['shape'] = 'o'
out_q['a_col'] = 1

# =========================
# 5) Build order (ORD) AFTER filtering, then assign y and x
# =========================

# ---- Brugada ----
candidates_b = [o for o in out_b['outcome'].unique() if 'brug' in str(o).lower()]
if not candidates_b:
    raise RuntimeError("No Brugada-like outcome found in filtered data.")
s_b = candidates_b[0]

ORD_b = {}
res_b = []
for o in met_b['outcome'].unique():
    suc = met_b[met_b['outcome'] == o]
    sor = []
    for prot in suc['protein'].unique():
        tmp = suc[suc['protein'] == prot].copy()
        tmp['sort'] = tmp['class'].map({c:i for i, c in enumerate(tmp['class'].value_counts().index)})
        tmp.sort_values('sort', inplace=True)
        res_b.append(tmp)
        first_class = tmp['class'].iloc[0] if not tmp.empty else CLASS_ORDER[0]
        sor.append([prot, tmp.shape[0], CLASS_ORDER.index(first_class)])
    if sor:
        sor = pd.DataFrame(sor, columns=['protein', 'no_met', 'class']).sort_values(['no_met', 'class'], ascending=[False, True])
        ORD_b[o] = list(sor['protein'])
    else:
        ORD_b[o] = []
met_b = pd.concat(res_b) if res_b else met_b

ordered_b = list(ORD_b.get(s_b, []))
for p in out_b.loc[out_b['outcome'] == s_b, 'protein'].astype(str).str.strip().unique():
    if p not in ordered_b:
        ordered_b.append(p)
ORD_b[s_b] = ordered_b

# ---- QRS ----
s_q = 'QRS duration'

ORD_q = {}
res_q = []
for o in met_q['outcome'].unique():
    suc = met_q[met_q['outcome'] == o]
    sor = []
    for prot in suc['protein'].unique():
        tmp = suc[suc['protein'] == prot].copy()
        tmp['sort'] = tmp['class'].map({c:i for i, c in enumerate(tmp['class'].value_counts().index)})
        tmp.sort_values('sort', inplace=True)
        res_q.append(tmp)
        first_class = tmp['class'].iloc[0] if not tmp.empty else CLASS_ORDER[0]
        sor.append([prot, tmp.shape[0], CLASS_ORDER.index(first_class)])
    if sor:
        sor = pd.DataFrame(sor, columns=['protein', 'no_met', 'class']).sort_values(['no_met', 'class'], ascending=[False, True])
        ORD_q[o] = list(sor['protein'])
    else:
        ORD_q[o] = []
met_q = pd.concat(res_q) if res_q else met_q

existing_q = ORD_q.get(s_q, [])
for p in out_q.loc[out_q['outcome'] == s_q, 'protein'].astype(str).str.strip().unique():
    if p not in existing_q:
        existing_q.append(p)
ORD_q[s_q] = existing_q

# -----------------------
# Enforce same y-order for both panels
# -----------------------
# Build a master order from Brugada ordering
master_order = list(ORD_b.get(s_b, []))

# Keep only proteins present in the intersection
master_order = [p for p in master_order if p in common_proteins]

# Append any QRS proteins that are in the intersection but missing from master_order
extra_q = [p for p in ORD_q.get(s_q, []) if (p in common_proteins and p not in master_order)]
master_order += extra_q

# Safety-net: append any remaining proteins deterministically (sorted)
missing = [p for p in sorted(common_proteins) if p not in master_order]
if missing:
    master_order += missing

# Now use this single sort_dict for both panels
sort_dict_master = {p: i for i, p in enumerate(master_order)}

# Re-assign distances (y_axis) using the common sort order and same between-pad
# For Brugada
sub_b = out_b[out_b['outcome'] == s_b].copy()
sub_b['protein'] = sub_b['protein'].astype(str).str.strip()
sub_b = forest.assign_distance(
    sub_b,
    group='protein',
    sort_dict=sort_dict_master,
    between_pad=2
)

# For QRS
sub_q = out_q[out_q['outcome'] == s_q].drop_duplicates(subset=['protein'], keep='first').copy()
sub_q['protein'] = sub_q['protein'].astype(str).str.strip()
sub_q = forest.assign_distance(
    sub_q,
    group='protein',
    sort_dict=sort_dict_master,
    between_pad=2
)

# Update metabolite data merges so they use the new y_axis coordinates:
suc_b = met_b[met_b['outcome'] == s_b].copy()
suc_b['protein'] = suc_b['protein'].astype(str).str.strip()
suc_b = suc_b.merge(sub_b[['protein', 'y_axis']], on='protein', how='left')
suc_b['col'] = suc_b['class'].map(CLASS_COL).fillna('#808080')
suc_b['x'] = suc_b.groupby('protein').cumcount() + 1
suc_plot_b = suc_b.dropna(subset=['y_axis']).copy()

suc_q = met_q[met_q['outcome'] == s_q].copy()
suc_q['protein'] = suc_q['protein'].astype(str).str.strip()
suc_q = suc_q.merge(sub_q[['protein', 'y_axis']], on='protein', how='left')
suc_q['col'] = suc_q['class'].map(CLASS_COL).fillna('#808080')
suc_q['x'] = suc_q.groupby('protein').cumcount() + 1

# Set axis ticks/limits info similar to before
XTIX_b = {s_b: [.85, 1, 1.5, 2, 2.5]}
max_x_b = int(suc_plot_b['x'].max()) if not suc_plot_b.empty else 4
TLIM_b  = {s_b: [0.5, max(4, max_x_b + 0.5)]}

# =========================
# 6) Plot — one PDF page
# =========================
fig = plt.figure(figsize=(20 * local_c.cmtoinch, 8 * local_c.cmtoinch))
gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.08, width_ratios=[1, 0.28, 1, 0.28])

axB_forest = fig.add_subplot(gs[0]); axB_meta = fig.add_subplot(gs[1])
axQ_forest = fig.add_subplot(gs[2]); axQ_meta = fig.add_subplot(gs[3])

# Brugada forest (log OR, ref=1)
ax = axB_forest
ax.axvline(1, linewidth=.3, linestyle='--', c='black', zorder=20)
sub_b = sub_b.sort_values('y_axis')
_, ax, _ = forest.plot_forest(
    df=sub_b, x_col='OR', lb_col='LCI', ub_col='UCI',
    g_col='protein', c_col='col', s_col='shape', shape_size=6,
    ci_lwd=.3, ci_colour='black', span=True, ax=ax,
    kwargs_scatter_dict={'zorder':10,'edgecolors':'black','linewidth':.3},
    kwargs_plot_ci_dict={'zorder':5,'solid_capstyle':'round','linestyle':'-'}
)
ax.set_xscale('linear')

# Set fixed ticks for Brugada forest (as before)
ax.set_xlim(0.2, 2.2)  # adjust limits as needed
ax.set_xticks([0.25,0.5, 0.75, 1, 1.25 ,1.5, 1.75, 2, 2.25, 2.50, 2.75, 3])
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))

ax.spines['top'].set_visible(False)
ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.tick_params(axis='both', which='minor', bottom=False, labelsize=4, width=.3)
ax.tick_params(axis='both', labelsize=4, width=.3)
ax.set_title('Brugada syndrome', loc='left', y=1, fontsize=5, fontweight='bold')

# Brugada metabolites
ax2 = axB_meta
ax2.spines[['top','right','bottom','left']].set_visible(False)
ax2.scatter(suc_plot_b['x'], suc_plot_b['y_axis'], c=suc_plot_b['col'],
            edgecolors='black', linewidth=.3, s=20, marker='s', zorder=8)
ax2.set_xlim(TLIM_b[s_b])
ax2.set_ylim([sub_b['y_axis'].min()-2, sub_b['y_axis'].max()+2])
ax2.set(yticks=[], xticks=[])

# QRS forest (beta, linear, ref=0)
ax = axQ_forest

# -------------------------
# QRS: robustly retrieve 'cardiac cis p-value' from fin_q, parse & mark Bonferroni significance
# -------------------------
# Bonferroni alpha
alpha = 0.05 / max(1, len(common_proteins))
print(f"Debug: Bonferroni alpha = {alpha:.3e} (0.05 / {len(common_proteins)})")

# Robust parser for p-value strings (handles '2.0×10⁻²', '<1e-6', '3.5e-06*', '1.0×10⁻¹⁰⁰', etc.)
superscript_map = {
    '⁰':'0','¹':'1','²':'2','³':'3','⁴':'4','⁵':'5','⁶':'6','⁷':'7','⁸':'8','⁹':'9',
    '⁻':'-','⁺':'+'
}
def replace_superscripts(s):
    return ''.join(superscript_map.get(ch, ch) for ch in s)

def parse_pval(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == '':
        return np.nan
    # remove trailing annotations like '*' or '†'
    s = re.sub(r'[\*\†\s]+$', '', s)
    # normalize unicode chars
    s = s.replace('−', '-').replace('×', 'x').replace('·', '.')
    s = replace_superscripts(s)
    # case: '<1e-6' -> use numeric part
    m = re.match(r'^\s*<\s*([0-9.+-eE]+)\s*$', s)
    if m:
        try:
            return float(m.group(1))
        except:
            return np.nan
    # case: '2.0x10-2' or '2.0x10^-2' -> convert to numeric
    m = re.match(r'^\s*([0-9.+-]+)\s*[xX]\s*10\^?([+-]?\d+)\s*$', s)
    if m:
        try:
            base = float(m.group(1))
            exp = int(m.group(2))
            return base * (10.0 ** exp)
        except:
            return np.nan
    m = re.match(r'^\s*([0-9.+-]+)\s*[xX]\s*10([+-]?\d+)\s*$', s)
    if m:
        try:
            base = float(m.group(1))
            exp = int(m.group(2))
            return base * (10.0 ** exp)
        except:
            return np.nan
    # final attempt: standard float parsing
    try:
        return float(s)
    except:
        # fallback: extract first numeric substring
        m2 = re.search(r'([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?\d+)?)', s)
        if m2:
            try:
                return float(m2.group(1))
            except:
                return np.nan
        return np.nan

# Preferred column names (explicit preference for 'cardiac cis p-value')
preferred_cols = ['cardiac cis p-value']
# build lowercase map of fin_q columns for case-insensitive matching
lower_map = {c.lower(): c for c in fin_q.columns}
chosen_col = None
for col in preferred_cols:
    if col.lower() in lower_map:
        chosen_col = lower_map[col.lower()]
        break

print("Debug: using P-value column from fin_q:", chosen_col)

# Build pv_map (protein -> parsed P-value)
if chosen_col is None:
    pv_map = pd.DataFrame(columns=['protein', 'P-value'])
    print("Debug: no cardiac cis p-value column found in fin_q; proceeding with empty pv_map")
else:
    pv_map = fin_q[['Protein name', chosen_col]].drop_duplicates().copy()
    pv_map.columns = ['protein', 'P-value']
    pv_map['protein'] = pv_map['protein'].astype(str).str.strip()
    pv_map['P-value'] = pv_map['P-value'].apply(parse_pval)
    n_pvals = pv_map['P-value'].notna().sum()
    print(f"Debug: parsed {n_pvals} p-values from column '{chosen_col}' (out of {len(pv_map)})")

# Merge P-values into sub_q (left join so sub_q keeps its rows/order)
sub_q['protein'] = sub_q['protein'].astype(str).str.strip()
if not pv_map.empty:
    pv_map['protein'] = pv_map['protein'].astype(str).str.strip()
    sub_q = sub_q.merge(pv_map, on='protein', how='left', validate='m:1')
else:
    sub_q['P-value'] = np.nan

# Coerce and compute significance
sub_q['P-value'] = pd.to_numeric(sub_q['P-value'], errors='coerce')
num_non_na = sub_q['P-value'].notna().sum()
print(f"Debug: sub_q contains {num_non_na} non-NaN P-values after merge (out of {len(sub_q)})")

n_signif = (sub_q['P-value'] < alpha).sum()
print(f"Debug: number of proteins with P < alpha: {n_signif}")

sub_q['is_signif_qrs'] = sub_q['P-value'] < alpha
sub_q['shape'] = sub_q['is_signif_qrs'].map({True: 'o', False: '^'}).fillna('^')

# Debug table
debug_tbl = sub_q[['protein', 'P-value', 'is_signif_qrs']].copy()
debug_tbl = debug_tbl.sort_values(['is_signif_qrs', 'P-value'], ascending=[False, True])
print("Debug: top rows of P-value table (significant first):")
print(debug_tbl.head(50).to_string(index=False))
sig_list = list(debug_tbl.loc[debug_tbl['is_signif_qrs'], 'protein'])
print("Debug: proteins flagged significant (Bonferroni):", sig_list)



# Now plot QRS forest
sub_q = sub_q.sort_values('y_axis')
_, ax, _ = forest.plot_forest(
    df=sub_q,
    x_col='Point estimate', lb_col='Lower bound', ub_col='Upper bound',
    g_col='protein', c_col='col', s_col='shape', shape_size=6,
    ci_lwd=.3, ci_colour='black', span=True, ax=ax,
    kwargs_scatter_dict={'zorder':10, 'edgecolors':'black', 'linewidth':.3},
    kwargs_plot_ci_dict={'zorder':5, 'solid_capstyle':'round', 'linestyle':'-' }
)

# ensure LINEAR scale
ax.set_xscale('linear')

# Set ticks symmetric around 0 (visually similar density to BrS)
ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0, 1.5])
ax.set_xlim(-1.2, 1.7)

# Reference line at 0
ax.axvline(0, linewidth=0.6, linestyle='--', c='black', zorder=50)

ax.spines['top'].set_visible(False)
ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.tick_params(axis='both', which='minor', bottom=False, labelsize=4, width=.3)
ax.tick_params(axis='both', labelsize=4, width=.3)
ax.set_title('QRS duration', loc='left', y=1, fontsize=5, fontweight='bold')

# # # -------------------------
# # Replace QRS metabolites panel with a protein-significance legend
# # -------------------------
# ax2 = axQ_meta

# # Clear and hide the axis (no grid, no spines, no ticks)
# ax2.clear()
# ax2.set_frame_on(False)
# ax2.axis('off')

# # Representative color for legend markers (use pal_q if available)
# rep_col = pal_q[0] if ('pal_q' in globals() and pal_q) else '#808080'


# sig_handle = Line2D([0], [0],
#                     marker='o', color='black',
#                     label='Significant (P < Bonferroni α)',
#                     markerfacecolor=rep_col, markeredgecolor='black',
#                     markersize=6, linestyle='None')
# nsig_handle = Line2D([0], [0],
#                      marker='^', color='black',
#                      label='Non-significant (P ≥ Bonferroni α)',
#                      markerfacecolor=rep_col, markeredgecolor='black',
#                      markersize=6, linestyle='None')

# # Place the legend centered in the (now-empty) QRS-meta panel
# ax2.legend(handles=[sig_handle, nsig_handle],
#            loc='right', frameon=False,
#            fontsize=4, title='Protein significance', title_fontsize=4,
#            ncol=1)



# # -------------------------
# # Shared legend (classes present) — only from Brugada (suc_plot_b)
# # -------------------------
# legend_classes = sorted(set(suc_plot_b['class'].dropna()))
# patches = [Patch(facecolor=CLASS_COL[c], edgecolor='black', label=c)
#            for c in legend_classes if c in CLASS_COL]

# # keep the main class legend centered below the figure (still shared), but derived only from Brugada
# fig.subplots_adjust(bottom=0.18)  # add a touch more space for two legends
# if patches:
#     fig.legend(handles=patches, loc='lower center', ncol=min(len(patches), 6),
#                frameon=False, title='Metabolite class',
#                title_fontsize=6, fontsize=5)


# # If you want the marker facecolor in the legend to reflect the QRS protein colors (one representative color),
# # you can pick a common color (e.g., pal_q[0]) or build multiple handles per-class+signif (more crowded).
# # Example for a single representative color (uncomment to use):
# # rep_col = pal_q[0]
# # sig_handle = Line2D([0],[0], marker='o', color='black', label='Significant', markerfacecolor=rep_col, markeredgecolor='black', markersize=5, linestyle='None')
# # nsig_handle = Line2D([0],[0], marker='^', color='black', label='Non-significant', markerfacecolor=rep_col, markeredgecolor='black', markersize=5, linestyle='None')
# # ax2.legend(handles=[sig_handle, nsig_handle], loc='upper right', frameon=False, fontsize=5, title='Protein significance', title_fontsize=6)
# -------------------------
# Place legends under each forest panel:
#  - Metabolite-class legend under Brugada forest (axB_forest)
#  - Protein-significance legend under QRS forest (axQ_forest)
# -------------------------

# 0) Clear/hide the QRS meta axis (we won't draw metabolites there)
axQ_meta.clear()
axQ_meta.axis('off')

# 1) Metabolite-class legend (only from Brugada suc_plot_b)
legend_classes = sorted(set(suc_plot_b['class'].dropna())) if ('class' in suc_plot_b.columns) else []
patches = [Patch(facecolor=CLASS_COL[c], edgecolor='black', label=c)
           for c in legend_classes if c in CLASS_COL]

# Add the metabolite-class legend under the Brugada forest (axB_forest)
if patches:
    # bbox_to_anchor uses axes coordinates (0..1) when bbox_transform=axB_forest.transAxes
    leg = axB_forest.legend(handles=patches,
                      loc='upper center',
                      bbox_to_anchor=(0.5, -0.15),  # centered below the axes
                      bbox_transform=axB_forest.transAxes,
                      ncol=min(len(patches), 6),
                      frameon=False,
                      title='Metabolite class',
                      title_fontsize=6,
                      fontsize=5)
# make the legend title bold
leg.get_title().set_fontweight('bold')
# 2) Protein-significance legend under QRS forest (axQ_forest)
# Representative facecolor for legend markers
rep_col = pal_q[0] if ('pal_q' in globals() and pal_q) else '#808080'

sig_handle = Line2D([0], [0], marker='o', color='black',
                    label='Significant',
                    markerfacecolor=rep_col, markeredgecolor='black',
                    markersize=4, linestyle='None')
nsig_handle = Line2D([0], [0], marker='^', color='black',
                     label='Non-significant',
                     markerfacecolor=rep_col, markeredgecolor='black',
                     markersize=4, linestyle='None')

leg_q = axQ_forest.legend(handles=[sig_handle, nsig_handle],
                  loc='upper center',
                  bbox_to_anchor=(0.5, -0.15),
                  bbox_transform=axQ_forest.transAxes,
                  ncol=1,
                  frameon=False,
                  title='Protein significance',
                  title_fontsize=6,
                  fontsize=5)
# make the legend title bold
leg_q.get_title().set_fontweight('bold')
# 3) Make room at the bottom for both legends
# Increase bottom margin (tweak value if legends are clipped or too far)
fig.subplots_adjust(bottom=0.20)
fig.text(0.25, 0.12, 'OR (95% CI)', ha='center', va='baseline', fontsize=6)
fig.text(0.68, 0.12, 'Mean difference (95% CI)', ha='center', va='baseline', fontsize=6)

# Save
out_pdf = '/gpfs/work2/0/aus20644/project/ksharma/brugada/results/final_forest_Brugada_QRS_sig_nsig.pdf'
plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
plt.close('all')
print(f"Saved side-by-side figure (common proteins): {out_pdf}")
