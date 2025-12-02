#!/usr/bin/env python3
# #############################################################################
# Process HPA expression data
#
# #############################################################################

# imports
import os
import pandas as pd
import numpy as np
import scipy.stats as ss
import statistics as stat
from statsmodels.stats.proportion import test_proportions_2indep, confint_proportions_2indep
import statsmodels.stats.multitest as smm
from pathlib import Path
from project_constants import Project_paths as local_paths
from project_constants import Constant as local_c

###################################################################
expression = pd.read_csv('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/HPA-exp_prio_ensemblids.tsv', sep = '\t', header = 0)
prmp = pd.read_csv('/gpfs/work2/0/aus20644/project/ksharma/brugada/data/protein_name_mapper.tsv', sep = '\t', header = 0)
print(expression.head())
print(prmp.head())

# Rename Ensembl_ID in expression to match prmp
expression = expression.rename(columns={'Ensembl_ID': 'ensemblid'})
# Merge with protein mapper
expression = expression.merge(prmp, on='ensemblid', how='left')
# Set index to ensemblid
expression.index = expression['ensemblid']
# Column used for expression
colexpres = 'normalizedRNAExpression'

# Quick check
for gene in expression['Name'].unique():
    temp = expression[(expression['Name'] == gene) & (expression['assayType'] == 'consensusTissue')]
    try:
        if len(temp['additional_information'].unique()) != temp.shape[0]:
            print(f"Lengths are not equal for gene {gene}")
    except ValueError:
        print(f"ValueError for gene {gene}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Tissue specific data

# Subsetting based on assayType
expression = expression.loc[expression['assayType'] == 'consensusTissue'].copy()

# Add ID column for later merging
expression['ID'] = expression.index

# Clean up additional_information column (remove trailing numbers if any)
expression['additional_information'] = expression['additional_information']\
    .str.replace(r' [0-9]+', "", regex=True)

# Create tissue_group column by removing 'organ:' prefix
expression['tissue_group'] = expression['tissue_name'].str.replace('organ:', "", regex=True)

# Save unique tissues for later
all_tissues = pd.Series(expression['additional_information'].unique(),
                        name='Tissues').to_csv(
                            os.path.join(local_paths.results, 'all_tissues.tsv'),
                            index=False, sep='\t'
                        )

# Columns to use for calculations
cols = ['uniprot_display_label', 'normalizedRNAExpression', 'tissue_name',
        'additional_information', 'ID', 'tissue_group']
sel = ['ID', 'uniprot_display_label', 'RNAExpression_tissues_med',
       'RNAExpression_tissues_mean', 'RNAExpression_heart']

res = []

# Calculate heart vs median/mean of other tissues
for gen in expression['uniprot_display_label'].unique():
    # Non-heart tissues
    tmp = expression[(expression['uniprot_display_label'] == gen) &
                     (expression['additional_information'] != 'Heart muscle')]
    # Heart tissues
    hrt = expression[(expression['uniprot_display_label'] == gen) &
                     (expression['additional_information'] == 'Heart muscle')].drop_duplicates()

    tmp = tmp[cols].drop_duplicates()
    hrt = hrt[cols].drop_duplicates()

    # Heart median expression
    if hrt.shape[0] > 1:
        med = stat.median(hrt[colexpres])
    else:
        med = hrt[colexpres].unique()[0]

    # Median and mean for other tissues
    tmp['RNAExpression_tissues_med'] = stat.median(tmp[colexpres])
    tmp['RNAExpression_tissues_mean'] = stat.mean(tmp[colexpres])
    tmp['RNAExpression_heart'] = med

    res.append(tmp[sel].drop_duplicates())

# Create final dataframe from tissue-specific results
res = pd.concat(res)
res.to_csv(os.path.join(local_paths.results, 'HPA-exp_median_heart.tsv'),
           sep='\t', index=False)

### Prioritise based on heart expression ###
pri = res[res['RNAExpression_heart'] > 0].iloc[:, :2]
pri.columns = ['ensemblid', 'uniprot_display_label']
pri.to_csv(os.path.join(local_paths.results, 'prioritised_proteins.tsv'),
           sep='\t', index=False)

### Calculating Tau values and differential expression p-values ###
expression['prioritized_pval'] = 1

# Dictionaries to store results
tau_dict, tissues_dict, uniq_tissues_dict, tissues_pos_dict, \
    uniq_tissues_pos_dict, tissues_neg_dict, uniq_tissues_neg_dict, \
    group_dict, uniq_group_dict, tissue_max = ({} for s in range(10))

differential_expression = pd.DataFrame()

for i in expression.index.unique():
    gene = expression.loc[i].copy()
    
    ### Tau calculation
    maximum = gene[colexpres].max()
    m = gene[colexpres] / maximum
    tau = np.sum(1 - m) / (gene.shape[0] - 1)
    tau_dict[i] = tau

    ### Differential expression p-values
    mean_val = gene[colexpres].mean()
    sd_val = np.sqrt(gene[colexpres].var())
    zvalues = (gene[colexpres] - mean_val) / sd_val
    gene['diff_expression_zstat'] = zvalues
    gene['diff_expression_pval'] = 2 * (1 - ss.norm.cdf(np.abs(zvalues)))
    
    differential_expression = differential_expression.append(gene, ignore_index=True)

    ### Tissues that are differentially expressed
    sub_sg = gene[gene['diff_expression_pval'] < 0.05]
    string1 = ', '.join(sub_sg['additional_information'])
    uniq1 = ', '.join(set(string1.split(', ')))
    tissues_dict[i] = string1
    uniq_tissues_dict[i] = uniq1

    string2 = ', '.join(sub_sg['tissue_group'])
    uniq2 = ', '.join(set(string2.split(', ')))
    group_dict[i] = string2
    uniq_group_dict[i] = uniq2

    # Tissues with expression above/below mean
    string3 = ', '.join(sub_sg[sub_sg['diff_expression_zstat'] > 0]['additional_information'])
    uniq3 = ', '.join(set(string3.split(', ')))
    string4 = ', '.join(sub_sg[sub_sg['diff_expression_zstat'] < 0]['additional_information'])
    uniq4 = ', '.join(set(string4.split(', ')))
    
    tissues_pos_dict[i] = string3
    uniq_tissues_pos_dict[i] = uniq3
    tissues_neg_dict[i] = string4
    uniq_tissues_neg_dict[i] = uniq4

    ### Tissue with largest expression
    tissue_max[i] = gene[gene[colexpres] == maximum]['additional_information'].iloc[0]

# Apply FDR correction across all p-values
fdr_results = smm.fdrcorrection(differential_expression['diff_expression_pval'],
                                alpha=0.05,
                                method='indep',
                                is_sorted=False)

# Add corrected p-values and significance flag
differential_expression['diff_expression_pval_adj'] = fdr_results[1]
differential_expression['diff_expression_significant'] = fdr_results[0]
# Check number of significant results before and after FDR
print("Significant before FDR:", sum(differential_expression['diff_expression_pval'] < 0.05))
print("Significant after FDR:", sum(differential_expression['diff_expression_significant']))


# Map calculated metrics back to expression dataframe
expression['tau'] = expression.index.map(tau_dict)
expression['diff_express_tissues'] = expression.index.map(tissues_dict)
expression['uniq_diff_express_tissues'] = expression.index.map(uniq_tissues_dict)
expression['diff_more_express_tissues'] = expression.index.map(tissues_pos_dict)
expression['uniq_diff_more_express_tissues'] = expression.index.map(uniq_tissues_pos_dict)
expression['diff_less_express_tissues'] = expression.index.map(tissues_neg_dict)
expression['uniq_diff_less_express_tissues'] = expression.index.map(uniq_tissues_neg_dict)
expression['grouped_tissues'] = expression.index.map(group_dict)
expression['uniq_grouped_tissues'] = expression.index.map(uniq_group_dict)
expression['max_express_tissue'] = expression.index.map(tissue_max)

# Merge differential expression p-values
expression = pd.merge(expression,
                      differential_expression[['ID', 'additional_information', 'diff_expression_pval_adj', 'diff_expression_significant']],
                      on=['ID', 'additional_information'],
                      how='left')

# expression = pd.merge(expression,
#                       differential_expression[['ID', 'additional_information', 'diff_expression_pval']],
#                       on=['ID', 'additional_information'])
expression.index = expression['ID']
del expression['ID']

## Save results
expression = expression.drop_duplicates()
expression.to_csv('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/HPA-exp_diff_expression.tsv.gz', sep='\t')
expression_fdr = expression[expression['diff_expression_significant']].drop_duplicates()
expression_fdr.to_csv('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/HPA-exp_diff_expression_FDR.tsv', sep='\t')

# Select and order columns (remove 'assayType')
cols = ['uniprot_display_label', 'normalizedRNAExpression',
        'proteinCodingRNAExpression', 'RNAExpression', 'tissue_name',
        'additional_information', 'tissue_group',
        'prioritized_pval', 'tau', 'uniq_diff_express_tissues',
        'uniq_diff_more_express_tissues', 'uniq_diff_less_express_tissues',
        'uniq_grouped_tissues', 'max_express_tissue', 'diff_expression_pval_adj']
expression = expression[cols].drop_duplicates()

# Filter and save heart-specific results
hrt = expression[expression['additional_information'] == 'Heart muscle']
hrt.to_csv('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/HPA-exp_diff_expression_heart.tsv', sep='\t')

# Filter significant differential expression
expression = expression[expression['diff_expression_pval_adj'] < 0.05].drop_duplicates()
expression.to_csv('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/HPA-exp_diff_expression_p0.05.tsv', sep='\t')

# Heart-specific significant results
hrt = expression[expression['additional_information'] == 'Heart muscle']
hrt.to_csv('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/HPA-exp_diff_expression_heart_p0.05.tsv', sep='\t')



