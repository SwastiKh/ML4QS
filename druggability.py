#!/usr/bin/env python3
##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Script information
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##
## Script name: druggability
##
## Purpose of script: Summarise druggability of proteins
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Imports
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import pandas as pd
import numpy as np
from project_constants import Project_paths as local_paths

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Loading data
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

summ = pd.read_csv('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/drug_summary.tsv', sep = '\t', header = 0,
        index_col = None).drop_duplicates()
eff = pd.read_csv('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/drug_effects.tsv', sep = '\t', header = 0,
        index_col = None).drop_duplicates()
subd = pd.read_csv('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/TableX_combined_test.tsv', sep = '\t',
         header = 0, index_col = None)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Prepare data
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Merge columns required for druggability information of both
cols = ['uniprot_id', 'max_phase', 'drug_effect_type', 'target_max_compound_phase',
     'target_type', 'target_n_chembl_indications',]
dref = eff[[x for x in cols if x in eff.columns]]
drum = summ[[x for x in cols if x in summ.columns]]
phase = dref.merge(drum, on = 'uniprot_id', how = 'outer', suffixes = ['_eff', '_sum'])

# fill empty cells of max_phase
phase['phase_eff'] = phase['max_phase'].fillna(0)
phase['phase_sum'] = phase['target_max_compound_phase'].fillna(0)
# group data by UniProt, due to >1 effect per target
uni_grouped = phase.groupby('uniprot_id')
# obtain max_phase over all effects
drugged = uni_grouped['phase_eff'].max()
phase['max_phase_eff'] = phase['uniprot_id'].map(drugged)
sumdrug = uni_grouped['phase_sum'].max()
phase['max_phase_sum'] = phase['uniprot_id'].map(sumdrug)
# combine
# phase['max_phase_tot'] = phase.apply(lambda x: x['max_phase_eff'] if
#         x['max_phase_eff'] >= x['max_phase_sum'] else
#         x['max_phase_sum'], axis = 1)

phase['max_phase_tot'] = phase[['max_phase_eff', 'max_phase_sum']].max(axis=1)
# collect druggable targets by looking for no NA in drug_effect_type column
# druggable = uni_grouped.drug_effect_type.apply(lambda x: x.notna().any())
druggable = phase.groupby('uniprot_id')['drug_effect_type'].apply(lambda x: x.notna().any())
phase['effect_type'] = phase['uniprot_id'].map(druggable)
# sumabl = uni_grouped.target_type_sum.apply(lambda x: x.notna().any())
# phase['effect_type_sum'] = phase['UniProt'].map(sumabl)
# phase['effect_type_tot'] = phase['effect_type_eff'] | phase['effect_type_sum']

# Also summarise target information
cols = summ.columns[summ.columns.str.startswith(pat = 'target_n_')]
for col in cols:
    new = summ.groupby('uniprot_id')[col].sum()
    summ[col] = summ['uniprot_id'].map(new)

# Ensure unique rows in each DataFrame before merging
phase = phase.drop_duplicates(subset=['uniprot_id'])
eff = eff.drop_duplicates(subset=['uniprot_id', 'drug_effect'])
summ = summ.drop_duplicates(subset=['Ensembl ID', 'uniprot_id', 'target_n_assays'])

# Define druggability category efficiently
phase['Druggability'] = 'Not yet druggable'
phase.loc[phase['effect_type'], 'Druggability'] = 'Druggable'
phase.loc[phase['target_n_chembl_indications'] == 0, 'Druggability'] = 'Not yet druggable'
phase.loc[(phase['max_phase_tot'] > 3) & (phase['effect_type']), 'Druggability'] = 'Drugged'

# Keep only relevant columns for merging
phase_merge = phase[['uniprot_id', 'max_phase_tot', 'Druggability']]

# Merge druggability info into effects
comb = eff.merge(phase_merge, how='left', on='uniprot_id')
print("Shape after merging druggability into effects:", comb.shape)

# Merge summary with effects+druggability
summ_cols = ['Ensembl ID', 'uniprot_id', 'target_n_assays']  # adjust as needed
summ_subset = summ[summ_cols].drop_duplicates()
print(summ_subset.columns.tolist())
print(comb.columns.tolist())

# Left join to avoid large Cartesian products
data = summ_subset.merge(comb, how='left', on=['Ensembl ID', 'uniprot_id'])
print("Final shape after merging summary:", data.shape)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Define druggability
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define druggability category
phase['Druggability'] = 'Not yet druggable'
phase.loc[phase.effect_type == True, 'Druggability'] = 'Druggable'
# Set to not druggable if no chembl indications intended target
phase.loc[phase['target_n_chembl_indications'] == 0, 'Druggability'] = 'Not yet druggable'
# Set to drugged if phase>3 and indication
phase.loc[(phase['max_phase_tot'] > 3) & (phase.effect_type == True), 'Druggability'] = 'Drugged'

print("Unique uniprot_id in eff:", eff['uniprot_id'].nunique())
print("Unique uniprot_id in phase:", phase['uniprot_id'].nunique())


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Merge data
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ## Merge results
# # Merge druggability with effects
phase = phase[['uniprot_id', 'max_phase_tot', 'Druggability']]
# comb = eff.merge(phase, how = 'outer', on = ['uniprot_id']).drop_duplicates()
comb = eff.merge(phase, how='left', on='uniprot_id')
print("Shape after merge:", comb.shape)
comb = comb.drop_duplicates(subset=['uniprot_id'])
# From summary extract only relevant information
cols = ['Ensembl ID', 'uniprot_id', 'target_n_assays'] + list(cols)
summ = summ[cols].drop_duplicates()
# Merge summary with results
data = summ.merge(comb, how = 'outer',
        on = ['Ensembl ID', 'uniprot_id'])
print("Final shape after merge:", data.shape)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Summarise effects
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Capitalize all side-effects and indications
data['drug_effect'] = data['drug_effect'].str.capitalize()
data['drug_effect'] = data['drug_effect'].str.strip()
data['drug_effect'].replace(to_replace = ',', value = np.nan, regex = False,
        inplace = True)
data['drug_effect'].replace(to_replace = '.in patients.*', value = '',
        regex = True, inplace = True)
data['drug_effect'].replace(to_replace = '\(.*\)', value = '',
        regex = True, inplace = True)
data['drug_effect'].replace(to_replace = '\[.*\]', value = '',
        regex = True, inplace = True)
data['drug_effect'].replace(to_replace = 'aem',
        value = 'em', regex = True, inplace = True)

data['drug_effect'].replace(to_replace = '[Aa]lopecia.*',
        value = 'Alopecia', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = '[Aa]ngioedema.*',
        value = 'Angiodema', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = 'disorders',
        value = 'disorder', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = 'infections',
        value = 'infection', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = 'Hallucinations',
        value = 'Hallucination', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = 'Pneumonia,.*',
        value = 'Pneumonia', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = 'Scleroderma,.*',
        value = 'Scleroderma', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = 'Tachycardia,.*',
        value = 'Tachycardia', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = 'Uveitis,.*',
        value = 'Uveitis', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = 'Dementia,.*',
        value = 'Dementia', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = 'Depressive disorder,.*',
        value = 'Depressive disorder', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = 'Essential hypertension',
        value = 'Hypertension', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = 'Hearing loss,.*',
        value = 'Hearing loss', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = 'Hepatitis c,.*',
        value = 'Hepatitis c', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = 'Hyperlipidemia.*',
        value = 'Hyperlipidemia', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = 'Hypertension, pulmonary',
        value = 'Pulmonary hypertension', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = 'Mastocytosis,.*',
        value = 'Mastocytosis', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = 'Myasthenia gravis.*',
        value = 'Myasthenia gravis', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = 'Renal insufficiency.*',
        value = 'Renal insufficiency', regex = True, inplace = True)
data['drug_effect'].replace(to_replace =
        'Established atherosclerotic cardiovascular disease',
        value = 'Atherosclerosis', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = 'Hypophosphatemia.*',
        value = 'Hypophosphatemia', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = '(?i).*hypophosphatemic.*',
        value = 'Hypophosphatemic rickets', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = 'Pulmonary disease, chronic obstructive',
        value = 'Chronic obstructive pulmonary disease', regex = True,
        inplace = True)

data['drug_effect'].replace(to_replace = '(?i).*neoplasm.*',
        value = 'Neoplasms', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = '(?i).*carcinoma.*',
        value = 'Carcinoma', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = '(?i).*lymphoma.*',
        value = 'Lymphoma', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = '(?i).*sarcoma.*',
        value = 'Sarcoma', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = '.*leukemia.*',
        value = 'leukemia', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = 'Leukemia,.*',
        value = 'Leukemia', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = 'tumour',
        value = 'tumor', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = '.*umor.*',
        value = 'Tumors', regex = True, inplace = True)

data['drug_effect'].replace(to_replace = '(?i).*malaria.*',
        value = 'Malaria', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = 'Anticoagulation.*',
        value = 'Anticoagulation', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = '.*abscess', value = 'Abscess',
        regex = True, inplace = True)
data['drug_effect'].replace(to_replace = '(?i).*anemia.*',
        value = 'anemia', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = '(?i).*arthritis.*',
        value = 'arthritis', regex = True, inplace = True)
data['drug_effect'].replace(to_replace = 'Angina.*',
        value = 'Angina', regex = True, inplace = True)

can = ['Neoplasms', 'Carcinoma', 'Cancer', 'Leukemia', 'Lymphoma', 'Sarcoma', 'Melanoma',
        'Multiple myeloma', 'Tumors', 'Glioblastoma', 'Mesothelioma',
        'Oligodendroglioma', 'Astrocytoma', 'Glioma', 'Neuroblastoma',
        'Hematoma', 'Diffuse intrinsic pontine glioma', 'Glaucoma',
        'Papilloma, inverted', 'Esthesioneuroblastoma, olfactory',
        'Granuloma, lethal midline', 'Adenoma, islet cell', 'Thymoma',
        'Myelodysplastic syndromes', ]
data.loc[data['drug_effect'].isin(can), 'drug_effect'] = 'Cancer'


cis = ['Non-st elevated myocardial infarction', 'Pulmonary hypertension',
        'Hypertension', 'Ischemic stroke', 'Heart failure',
        'Takotsubo cardiomyopathy', 'Aortic valve stenosis', 'Atrial flutter',
        'Ischemia', 'Acute coronary syndrome', 'Atrial fibrillation',
        'Aortic valve disease', 'Heart diseases', 'Coronary disease',
        'Myocardial infarction',
        'Prophylaxis of stroke and systemic embolism in non-valvular atrial fibrillation and at least one risk factor',
        'Prophylaxis of atherothrombotic events following an acute coronary syndrome with elevated cardiac biomarkers',
        'Tachycardia', 'Myocardial ischemia', 'Atherosclerosis',
        'Cardiovascular diseases', 'Coronary artery disease',
        'St elevation myocardial infarction', 'Endocarditis',
        'Chest discomfort', 'Palpitations', 'Cardiac conduction disorder',
        'Bradycardia', 'Arrhythmias', 'Angina', 'Atrioventricular block',
        'Ventricular dysfunction, left', 'Cardiomyopathy, dilated',
        'Hypertension if used in volume depletion, cardiac decompensation, or renovascular hypertension',
        'Prophylaxis of symptomatic heart failure after myocardial infarction in clinically stable patients with asymptomatic left ventricular dysfunction',
        'Short-term treatment within 24 hours of onset of myocardial infarction in clinically stable patients',
        'Chest pain', 'Cardiac arrest', 'Cardiogenic shock',
        'Prevention of symptomatic heart failure', 'Congestive heart failure',
        'Hypertensive crisis', 'Heart failure',
        'Hypertension, when used in addition to diuretic, in cardiac decompensation or in volume depletion',
        'Short-term treatment following myocardial infarction in hemodynamically stable patients‚Äîsystolic blood pressure 100‚Äì120‚ÄØmmhg',
        'Short-term treatment following myocardial infarction in hemodynamically stable patients‚Äîsystolic blood pressure over 120‚ÄØmmhg',
        'Prophylaxis of cardiac events following myocardial infarction or revascularisation in stable coronary artery disease',
        'Symptomatic heart failure',
        'Hypertension not adequately controlled by perindopril alone',
        'Prevention of cardiovascular events',
        'Prophylaxis after myocardial infarction',
        'Prophylaxis of stroke and systemic embolism in non-valvular atrial fibrillation and at least one risk factor',
        'Atherosclerosis',
        'Unstable angina or non-st-segment elevation myocardial infarction',
        'Cardiac tamponade', 'Cardiac disorder', 'Qt interval prolongation',
        'Emergency loading dose, for atrial fibrillation or flutter',
        'Rapid digitalisation, for atrial fibrillation or flutter',
        'Cardiomyopathies', 'Cardiomyopathy, hypertrophic']
data.loc[data['drug_effect'].isin(cis), 'effect_class'] = 'cardiac'
data['drug_effect'] = data['drug_effect'].str.capitalize()
data['drug_effect'] = data['drug_effect'].str.strip()
data.to_csv('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/drug_table.tsv',sep = '\t', index = False)

# Prepare full druggability IDs file
druggability_ids = phase[['uniprot_id', 'max_phase_tot', 'Druggability']].drop_duplicates()

# If you want, you can merge Ensembl IDs from the summary
druggability_ids = druggability_ids.merge(
    summ[['Ensembl ID', 'uniprot_id']].drop_duplicates(),
    on='uniprot_id',
    how='left'
)

# Save full druggability IDs
druggability_ids.to_csv(
    '/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/final_druggability_IDs_full.tsv',
    sep='\t',
    index=False
)

print("Saved full druggability IDs:", druggability_ids.shape)

# ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ## Subset
# ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ## Create Brugada syndrome subset and save
# ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# # Merge Ensembl IDs into phase_merge
# phase_merge = phase[['uniprot_id', 'max_phase_tot', 'Druggability']].merge(
#     summ[['Ensembl ID', 'uniprot_id']].drop_duplicates(),
#     on='uniprot_id',
#     how='left'
# )

# # Now subset for Brugada proteins
# brugada_proteins = subd[(subd['GW outcome'] == "Brugada syndrome") |
#                         (subd['cardiac cis Outcome'] == "Brugada syndrome")][['Uniprot ID']]
# print(subd.columns.tolist())

# comp_brugada = phase_merge[phase_merge['uniprot_id'].isin(brugada_proteins['Uniprot ID'])]

# # Save
# comp_brugada.to_csv(
#     os.path.join(local_paths.here, 'results', 'final_druggability_IDs_brugada.tsv'),
#     sep='\t', index=False
# )

# ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ## Create QRS subset and save
# ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# # Merge Ensembl IDs into phase_merge
# phase_merge = phase[['uniprot_id', 'max_phase_tot', 'Druggability']].merge(
#     summ[['Ensembl ID', 'uniprot_id']].drop_duplicates(),
#     on='uniprot_id',
#     how='left'
# )

# # # Subset for QRS proteins
# # qrs_proteins = subd[(subd['GW outcome'] == "QRS duration") |
# #                     (subd['cardiac cis Outcome'] == "QRS duration")][['uniprot_id']].drop_duplicates()

# # comp_qrs = phase_merge[phase_merge['uniprot_id'].isin(qrs_proteins['uniprot_id'])]

# # # Save
# # comp_qrs.to_csv(
# #     os.path.join(local_paths.here, 'results', 'final_druggability_IDs_QRS.tsv'),
# #     sep='\t', index=False
# # )





# # ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # ## Create frequencies of indications and side-effects
# # ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Save unique side-effects and indications
sei = data['drug_effect'].unique().tolist()
# Initiate list to create results df from
res = []
# Iterate over effects to count
for s in sei:
    # subset
    sal = data[data['drug_effect'] == s]
    # count indications and side-effects
    ind = sal[sal['drug_effect_type'] == 'indication'].shape[0]
    eff = sal[sal['drug_effect_type'] == 'side_effect'].shape[0]
    # save in new row
    res.append([s, ind, eff, sal.shape[0]])

# Make results df
freq = pd.DataFrame(res,
        columns = ['Phenotype', 'Indications', 'Side-effects', 'Total'])

# Tidy up
freq = freq[freq['Total'] > 0].sort_values('Total', ascending = False)
freq.to_csv('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/druggability_frequencies.tsv', sep = '\t',index = False)

# #evaluating frequencies in Brugada subset
# sei = data_brugada['drug_effect'].unique().tolist()
# res = []
# for s in sei:
#     sal = data_brugada[data_brugada['drug_effect'] == s]
#     ind = sal[sal['drug_effect_type'] == 'indication'].shape[0]
#     eff = sal[sal['drug_effect_type'] == 'side_effect'].shape[0]
#     res.append([s, ind, eff, sal.shape[0]])

# freq = pd.DataFrame(res, columns=['Phenotype', 'Indications', 'Side-effects', 'Total'])
# freq = freq[freq['Total'] > 0].sort_values('Total', ascending=False)
# freq.to_csv('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/druggability_frequencies_brugada.tsv', sep='\t', index=False)
