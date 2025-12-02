## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Imports
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import pandas as pd
from project_constants import Project_paths as local_paths
from project_constants import Constant as local_c
import matplotlib
import matplotlib.pyplot as plt
plt.ion()

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Loading data
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

expression = pd.read_csv('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/HPA-exp_prio_ensemblids.tsv', sep = '\t', index_col = 0, low_memory = False)
prmp = pd.read_csv('/gpfs/work2/0/aus20644/project/ksharma/brugada/data/protein_name_mapper.tsv', sep = '\t', header = 0)
# map names
expression.loc[:, 'ensemblid'] = expression.index
expression = expression.merge(prmp, on = 'ensemblid', how = 'left')
expression.index = expression['ensemblid']

# extract the correct tissues
expression = expression.loc[expression.assayType == 'consensusTissue'].copy()
others = expression[expression['tissue_name'].str.startswith('organ')]
expression = expression.loc[expression.additional_information == 'Heart muscle'].copy()

# calculate ratios
hrat = pd.DataFrame()
for prot in others.uniprot_display_label.unique():
    # extract data for this protein
    tmp = others[others['uniprot_display_label'] == prot]
    tmp = tmp.dropna(subset = ['normalizedRNAExpression'])
    # calculate total expression
    tot = tmp['normalizedRNAExpression'].sum()
    # extract heart and calculate ratio
    hrt = tmp.loc[tmp['additional_information'] == 'Heart muscle', ]
    rat = hrt.loc[hrt.index.unique()[0], 'normalizedRNAExpression'] / tot * 100
    # save results
    if hrat.shape[0] > 0:
        hrat = pd.concat([hrat, pd.DataFrame({'protein' : [prot],
                    'total' : [tot], 'heart' : [hrt], 'ratio' : [rat]})])
    else:
        hrat = pd.DataFrame({'protein' : [prot],
                'total' : [tot], 'heart' : [hrt], 'ratio' : [rat]})

expression = expression[['uniprot_display_label', 'normalizedRNAExpression']].drop_duplicates()
heart = pd.read_csv(os.path.join(local_paths.results,
    'HPA-exp_diff_expression_heart_p0.05.tsv'), sep = '\t', index_col = 0)
heart = list(heart['uniprot_display_label'].unique())

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Visualise absolute cardiac values
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

expression = expression.sort_values('normalizedRNAExpression',
        ascending = False)

fig = plt.figure(figsize = (17 * local_c.cmtoinch, 6 * local_c.cmtoinch))
matplotlib.rcParams['axes.linewidth'] = .5

ax = fig.add_subplot()
plt.bar(expression['uniprot_display_label'], expression['normalizedRNAExpression'],
        width = .8, color = '#f9bc39', edgecolor = 'black', linewidth = .5)
ax.spines[['top', 'right']].set_visible(False)
plt.xticks(rotation = 45, ha = 'right')
ax.tick_params(axis = 'both', labelsize = 3.2, width = .5)
ax.set_ylabel('RNA expression (nTPM)', loc = 'center', size = 4)
ax.set_xlim([-1, expression.shape[0] + 1])
for prot in heart:
    xti = ax.get_xticklabels()
    if prot in list(expression['uniprot_display_label']):
        xti[list(expression['uniprot_display_label']).index(prot)].set_weight('bold')

plt.savefig(os.path.join('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs', 'Figure3_expression.pdf'), bbox_inches = 'tight')
plt.close('all')

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Visualise ratios
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

hrat = hrat[['protein', 'ratio']].drop_duplicates()
hrat = hrat.sort_values('ratio', ascending = False)

fig = plt.figure(figsize = (17 * local_c.cmtoinch, 6 * local_c.cmtoinch))
matplotlib.rcParams['axes.linewidth'] = .5

ax = fig.add_subplot()
ax.axhline(1, linewidth = .3, linestyle = '-', c = 'black', zorder = 0)
plt.bar(hrat['protein'], hrat['ratio'], width = .8, color = '#FF77A8',
        edgecolor = 'black', linewidth = .5)
ax.spines[['top', 'right']].set_visible(False)
plt.xticks(rotation = 45, ha = 'right')
ax.tick_params(axis = 'both', labelsize = 3.2, width = .5)
ax.set_ylabel('Percentage cardiac tissue\nmRNA expression', loc = 'center', size = 4)
ax.set_xlim([-1, expression.shape[0] + 1])
for prot in heart:
    xti = ax.get_xticklabels()
    if prot in list(hrat['protein']):
        xti[list(hrat['protein']).index(prot)].set_weight('bold')

plt.savefig(os.path.join('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs', 'Figure3_expression_ratio.pdf'), bbox_inches = 'tight')
plt.close('all')
