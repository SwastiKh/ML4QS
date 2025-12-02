#!/usr/bin/env python3
##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Script information
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##
## Script name: extract_results.py
##
## Purpose of script: Extract all MR results in chronological order
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Imports
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import numpy as np
import pandas as pd
import formatting as fm
import mygene
from project_constants import Project_paths as local_paths
from project_constants import Constant as local_c
from project_constants import (
    OUTCOME_ORDER,
    OUTCOMES,
    EXPOSURES,
    CLASSES,
    EXPOSURE_ORDER,
    STUDY_ORDER,
    PROTEINS
)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Process genome-wide MR results
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Processing genome-wide MR results')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
gw = pd.read_csv(os.path.join(local_paths.here, 'gw_mr','extracted_results', 'results_file.tsv.gz'),
        sep='\t', header=0 , index_col=0)

print(str(len(gw.Exposure.unique().tolist())) + ' metabolites tested') #print number of metabolites tested
print(gw.columns)

# Inspect unique Outcome values to see exact spelling
print("Unique Outcome values containing 'Brugada':")
print([v for v in gw['Outcome'].dropna().unique() if 'brug' in str(v).lower()])

# Clean the Outcome column: remove extra spaces
outcome_clean = gw['Outcome'].astype(str).str.strip()

# Filter rows where Outcome contains "Brugada" (case-insensitive)
brugada = gw[outcome_clean.str.contains('brugada', case=False, na=False)]
qrs = gw[outcome_clean.str.contains('QRS', case=False, na=False)]

# Check how many rows were found
print("Number of rows with Brugada outcome:", len(brugada))
print("Number of rows with QRS duration:", len(qrs))


brugada.loc[:, 'OR'] = np.exp(brugada['Point estimate']) #converts log estimates to odds ratios: OR: Main effect size
brugada.loc[:, 'LCI'] = np.exp(brugada['Lower bound']) #lower confidence interval
brugada.loc[:, 'UCI'] = np.exp(brugada['Upper bound'])#upper confidence 

qrs.loc[:, 'OR'] = np.exp(qrs['Point estimate']) #converts log estimates to odds ratios: OR: Main effect size
qrs.loc[:, 'LCI'] = np.exp(qrs['Lower bound']) #lower confidence interval
qrs.loc[:, 'UCI'] = np.exp(qrs['Upper bound'])#upper confidence interval

dor = {}
brugada['format_OR'] = brugada.apply(lambda row: fm._format_estimates(point = row['Point estimate'], se = row['Standard error'], exp = True), axis = 1)
qrs['format_OR'] = qrs.apply(lambda row: fm._format_estimates(point = row['Point estimate'], se = row['Standard error'], exp = True), axis = 1)

# Filter for outcomes in list/dict OUTCOMES
brugada = brugada[brugada['Outcome'].isin(OUTCOMES.keys())]
qrs = qrs[qrs['Outcome'].isin(OUTCOMES.keys())]
# Calculate both thresholds
p_strict = 0.05 / len(gw['Exposure'].unique().tolist()) / local_c.variance_npc
p_thres = 0.05 / len(gw['Exposure'].unique().tolist()) / len(OUTCOMES)

# Add both thresholds as new columns (same value for all rows)
brugada.loc[:, 'pvalue_threshold_loose'] = p_thres
brugada.loc[:, 'pvalue_threshold_strict'] = p_strict

# Add formatted versions for display
brugada.loc[:, 'pvalue_format_loose'] = fm.sci_notation(p_thres)
brugada.loc[:, 'pvalue_format_strict'] = fm.sci_notation(p_strict)

# Save unfiltered data copy before filtering
nor = brugada.copy()
# Add both thresholds as new columns (same value for all rows)
qrs.loc[:, 'pvalue_threshold_loose'] = p_thres
qrs.loc[:, 'pvalue_threshold_strict'] = p_strict

# Add formatted versions for display
qrs.loc[:, 'pvalue_format_loose'] = fm.sci_notation(p_thres)
qrs.loc[:, 'pvalue_format_strict'] = fm.sci_notation(p_strict)

# Save unfiltered data copy before filtering
nor = qrs.copy()

local_paths.results = "/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs"
gw.to_csv(os.path.join(local_paths.results, 'gw_results_1.tsv'), sep='\t', index=False)
brugada.to_csv(os.path.join(local_paths.results, 'gw_results_brugada.tsv'), sep='\t', index=False)
qrs.to_csv(os.path.join(local_paths.results, 'gw_results_qrs.tsv'), sep='\t', index=False)

# Filter using only the loose p-value threshold
print('Filter for metabolites with p < ' + fm.sci_notation(p_thres))
gw = gw[gw['P-value (float)'] < p_thres]
brugada = brugada[brugada['P-value (float)'] < p_thres]
qrs = qrs[qrs['P-value (float)'] < p_thres]

# Save filtered results (loose p-value)
filt_l = "{:.0e}".format(p_thres).replace('e-', '_')
file_1 = f"gw_results_p{filt_l}.tsv"
brugada.to_csv(os.path.join(local_paths.results, file_1), sep='\t', index=False)

# Save filtered results (strict p-value)
filt_s = "{:.0e}".format(p_strict).replace('e-', '_')
file_2 = f"gw_results_p{filt_s}.tsv"
brugada.to_csv(os.path.join(local_paths.results, file_2), sep='\t', index=False)
qrs.to_csv(os.path.join(local_paths.results, file_2), sep='\t', index=False)

# Filter for minimum number of variants
print('Filter metabolites with number of variants > ' + str(local_c.var_threshold))
gw = gw[gw['No. variants'] > local_c.var_threshold]
brugada = brugada[brugada['No. variants'] > local_c.var_threshold]
qrs = qrs[qrs['No. variants'] > local_c.var_threshold]

# Save filtered results after variant filtering
file_3 = f"gw_results_var_p{filt_l}_n{local_c.var_threshold}.tsv"
brugada.to_csv(os.path.join(local_paths.results, file_3), sep='\t', index=False)
qrs.to_csv(os.path.join(local_paths.results, file_3), sep='\t', index=False)
file_4 = f"gw_results_var_p{filt_s}_n{local_c.var_threshold}.tsv"
brugada.to_csv(os.path.join(local_paths.results, file_4), sep='\t', index=False)
qrs.to_csv(os.path.join(local_paths.results, file_4), sep='\t', index=False)

# Save list of significant metabolites
met = pd.DataFrame(brugada['Exposure'].unique(), columns=['metabolite'])
file_sig_1 = f"significant_metabolites_p{filt_l}_n{local_c.var_threshold}.tsv"
met.to_csv(os.path.join(local_paths.results, file_sig_1), sep='\t', index=False)
file_sig_2 = f"significant_metabolites_p{filt_s}_n{local_c.var_threshold}.tsv"
met.to_csv(os.path.join(local_paths.results, file_sig_2), sep='\t', index=False)

# Print how many significant metabolites were found
met_list = brugada['Exposure'].unique()
met_list_1 = qrs['Exposure'].unique()
print(f"{len(met_list)} metabolites associated with cardiac outcomes")
print(f"{len(met_list_1)} metabolites associated with cardiac outcomes")

# Extract and save all results for these metabolites from unfiltered copy 'nor', for heatmap
hm = nor[nor['Exposure'].isin(met_list)]
hm_1 = nor[nor['Exposure'].isin(met_list_1)]
hm.to_csv(os.path.join(local_paths.results, 'gw_results_heat.tsv'), sep='\t', index=False)
hm_1.to_csv(os.path.join(local_paths.results, 'gw_results_heat_qrs.tsv'), sep='\t', index=False)
exit()
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Process metabolite cis-MR results
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### Load results
print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Processing metabolite cis-MR results')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

# Load bad somamers
somamer = pd.read_csv(os.path.join(local_paths.data,
    'protein_panels_druggability_and_secreted_SOMAmers_status_13_04_21.csv'),
            sep = '\t')
somamer = somamer[somamer['bad_somamer'] == 1]
# initiate list with ensembl IDs
enun = []
# initiate dataframe for results
dups = []
metab = []

# Load protein mapping
prot = pd.read_csv(os.path.join(local_paths.data, 'protein_name_mapper.tsv'), sep='\t')

# Clean protein names
prot['gene_name_clean'] = prot['gene_name'].str.strip().str.upper()
gene_to_ensembl = prot.set_index('gene_name_clean')['ensemblid'].to_dict()

# Load list of exposures to keep
gw_output = pd.read_csv('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/gw_results_brugada.tsv', sep='\t')
met = list(gw_output['Exposure'])

# Initialize MyGeneInfo to obtain the ensembl IDs
mg = mygene.MyGeneInfo()

# Initialize tracking variables
enun = []
dups = []
metab = []

# Initialize MyGeneInfo
mg = mygene.MyGeneInfo()

for study in STUDY_ORDER:
    print(f"\nProcessing study: {study}")
    
    # Load study file
    filepath = os.path.join(local_paths.metmr, study, 'extracted_results', "results_file.tsv.gz")
    res = pd.read_csv(filepath, sep='\t', header=0, index_col=0, low_memory=False)

    # Clean Exposure
    res['Exposure_clean'] = res['Exposure'].str.strip().str.upper()

    # Map using protein file
    res['ensemblid'] = res['Exposure_clean'].map(gene_to_ensembl)

    # Fallback mapping using mygene for unmapped
    unmapped = res[res['ensemblid'].isna()]['Exposure_clean'].unique()
    if len(unmapped) > 0:
        print(f" - Mapping {len(unmapped)} missing gene symbols via MyGeneInfo")
        result = mg.querymany(unmapped, scopes='symbol', fields='ensembl.gene', species='human')

        # Build mapping
        symbol_to_ensembl={}
        for item in result:
            if 'ensembl' in item and not item.get('notfound', False):
                ens = item['ensembl']
                if isinstance(ens, list):
                    symbol_to_ensembl[item['query']] = ens[0]['gene']
                else:
                    symbol_to_ensembl[item['query']] = ens['gene']

        # Update unmapped rows
        still_missing = res['ensemblid'].isna()
        res.loc[still_missing, 'ensemblid'] = res.loc[still_missing, 'Exposure_clean'].map(symbol_to_ensembl)

    # Final check
    mapped_count = res['ensemblid'].notna().sum()
    print(f" - Mapped {mapped_count} / {len(res)} exposures")

    # Drop unmapped
    res = res.dropna(subset=['ensemblid'])

    # Remove bad somamers if applicable
    if study in ['decode', 'gudjonsson', 'interval', 'gilly', 'yang']:
        res = res[~res['ensemblid'].isin(somamer['geneID'])]

    # Keep only relevant outcomes
    res = res[res['Outcome'].isin(met)]

    # Keep only rows with sufficient variants
    res = res[res['No. variants'] > local_c.var_threshold]

    # Separate duplicates (already selected in previous studies)
    dup = res[res['ensemblid'].isin(enun)]
    res = res[~res['ensemblid'].isin(enun)]

    # Update tracking list
    enun += res['ensemblid'].tolist()
    enun = list(set(enun))  # Ensure uniqueness

    # Add study info
    res['study'] = study
    dup['study'] = study

    # Collect results
    dups.append(dup)
    metab.append(res)

# Done
print("\nFinished processing all studies.")


# combine results
metab = pd.concat(metab, sort = False)
# Format metabolite cis-MR results
dor_metab = {}
for index, row in metab.iterrows():
    ids = row['ensemblid']
    dor_metab[ids] = fm._format_estimates(
        point=row['Point estimate'],
        se=row['Standard error'],
        exp=False  # or True if you want OR format
    )

metab['format_results'] = metab['ensemblid'].map(dor_metab)
# map protein names
if os.path.exists(os.path.join(local_paths.data, 'protein_name_mapper.tsv')):
    # check if complete mapper already exists
    prmp = pd.read_csv(os.path.join(local_paths.data,
        'protein_name_mapper.tsv'), sep = '\t', header = 0)
    # map names
    metab = metab.merge(prmp, on = 'ensemblid', how = 'left')

else:
    # if doesn't exist create it -- load full mapper
    prmp = pd.read_csv(os.path.join(local_paths.data,
        'homo_sapiens_prot.v102.b37.txt.gz'), sep = '\t', header = 0,
        index_col = 0)
    # remove suffix from protein names
    prmp['uniprot_display_label'].replace(to_replace = '_HUMAN', value = '',
            regex = True, inplace = True)
    # save all unique uniprot ids
    pid = prmp.uniprot_accession.unique()
    # filter for uniprot id in our study
    ids = [x for x in metab.uniprot_id.unique() if x in pid]
    prmp = prmp[prmp['uniprot_accession'].isin(ids)]
    # remove redundant columns
    cols = ['uniprot_accession', 'uniprot_display_label', 'uniprot_description']
    prmp = prmp[cols].drop_duplicates().rename(columns =
        {'uniprot_accession' : 'uniprot_id'})
    # map names
    metab['uniprot_id'] = metab['ensemblid'].map(ENSEMBL)
    metab = metab.merge(prmp, left_on = 'uniprot_id',
            right_on = 'uniprot_accession', how = 'left')
    del metab['uniprot_accession']
    metab['gene_name'] = metab['uniprot_id'].map(PROTEINS)
    # save mapper for all proteins
    mas = metab[['ensemblid',  'uniprot_id', 'gene_name',
        'uniprot_display_label', 'uniprot_description']].drop_duplicates()
    mas.to_csv(os.path.join(local_paths.data, 'protein_name_mapper.tsv'),
            index = None, sep = '\t')

# report number of proteins
print(str(len(enun)) + ' proteins tested')

# define p-values in df
p_thres = 0.05 / len(enun)
metab.loc[:, 'pvalue_threshold'] = p_thres
metab.loc[:, 'pvalue_format'] = fm.sci_notation(p_thres)
p_stric = 0.05 / len(enun) / len(met)
metab['pvalue_strict'] = p_stric

# Define p-value thresholds
p_thres = 0.05 / len(enun)                    # less strict threshold
p_stric = 0.05 / len(enun) / len(met)         # stricter threshold
print("Strict p-value threshold:", p_stric)
print("Loose p-value threshold:", p_thres)
# Add thresholds as columns for reference (in the full dataframe)
metab.loc[:, 'pvalue_threshold_loose'] = p_thres
metab.loc[:, 'pvalue_format_loose'] = fm.sci_notation(p_thres)
metab.loc[:, 'pvalue_threshold_strict'] = p_stric
metab.loc[:, 'pvalue_format_strict'] = fm.sci_notation(p_stric)

# Save full unfiltered data with thresholds for reference
file = ''.join(['metabolite_results_n', str(local_c.var_threshold), '.tsv'])
metab.to_csv(os.path.join(local_paths.data, file), sep='\t', index=False)

# Save combined duplicates if you have those
dups = pd.concat(dups, sort=False)
dups.to_csv(os.path.join(local_paths.results, 'metabolite_duplicate_protein_results.tsv'), sep='\t', index=False)

### Filtering with loose p-value threshold ###

print('Filter for proteins with p<' + fm.sci_notation(p_thres))

# Filter by loose p-value only
metab_filtered = metab[metab['P-value (float)'] < p_thres].copy()

# Save filtered results (loose threshold)
filt_loose = "{:.0e}".format(p_thres).replace('e-', '_')
file_loose = ''.join(['metabolite_results_n', str(local_c.var_threshold), '_p', filt_loose, '.tsv'])
metab_filtered.to_csv(os.path.join(local_paths.results, file_loose), sep='\t', index=False)

# Save proteins from loose filtered set
cols = ['ensemblid', 'uniprot_id', 'gene_name', 'uniprot_display_label', 'uniprot_description']
prot = metab_filtered[cols].drop_duplicates().rename(columns={
    'uniprot_display_label': 'protein_name',
    'uniprot_description': 'protein_description'
})
file_prot_loose = ''.join(['metabolite_proteins_n', str(local_c.var_threshold), '_p', filt_loose, '.tsv'])
prot.to_csv(os.path.join(local_paths.results, file_prot_loose), sep='\t', index=False)

prot_list = metab_filtered['ensemblid'].unique()  # Avoid reusing 'prot' as variable name

### Filtering with strict p-value threshold ###

print('Filter for proteins with p<' + fm.sci_notation(p_stric))

strict_filtered = metab[metab['P-value (float)'] < p_stric].copy()
strict_filtered.loc[:, 'pvalue_threshold'] = p_stric
strict_filtered.loc[:, 'pvalue_format'] = fm.sci_notation(p_stric)

# Save filtered strict results
filt_strict = "{:.2e}".format(p_stric).replace('e-', '_')
file_strict = ''.join(['metabolite_results_n', str(local_c.var_threshold), '_p', filt_strict, '.tsv'])
strict_filtered.to_csv(os.path.join(local_paths.results, file_strict), sep='\t', index=False)

# Save proteins from strict filtered set
cols = ['ensemblid', 'uniprot_id', 'gene_name', 'uniprot_display_label', 'uniprot_description']
prot = strict_filtered[cols].drop_duplicates().rename(columns={
    'uniprot_display_label': 'protein_name',
    'uniprot_description': 'protein_description'
})
file_prot_strict = ''.join(['metabolite_proteins_n', str(local_c.var_threshold), '_p', filt_strict, '.tsv'])
prot.to_csv(os.path.join(local_paths.results, file_prot_strict), sep='\t', index=False)

prot_list = strict_filtered['ensemblid'].unique()  # Avoid reusing 'prot' as variable name

# Count proteins per metabolite (strict filtered)
pr_gr = strict_filtered.groupby('Outcome')['Exposure'].nunique()
pr_ran = pr_gr.describe()
print(f"{pr_ran['50%']} proteins [IQR {pr_ran['25%']}, {pr_ran['75%']}] per metabolite")

# Clean 'Outcome' column for mapping to classes
strict_filtered['Outcome_clean'] = (
    strict_filtered['Outcome']
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
    .str.replace(':', '')
)

# Clean CLASSES keys similarly
CLASSES_clean = {k.strip().lower().replace(' ', '_').replace(':', ''): v for k, v in CLASSES.items()}

# Map to class
strict_filtered['class'] = strict_filtered['Outcome_clean'].map(CLASSES_clean)

# Group by class and describe
pr_gr_class = strict_filtered.groupby('class')['Exposure'].nunique()
pr_ran_class = pr_gr_class.describe()
print(f"{pr_ran_class['50%']} proteins [IQR {pr_ran_class['25%']}, {pr_ran_class['75%']}] per metabolite class")
# Proteins associated with a cardiac metabolite for Brugada
metab_brugada = strict_filtered[strict_filtered['Outcome'].str.contains("Brugada", case=False, na=False)]
n_proteins_metab = metab_brugada['ensemblid'].nunique()
print("Proteins associated with a cardiac metabolite (Brugada):", n_proteins_metab)

### Load cardiac cis-MR results
print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Processing cardiac cis-MR results')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

# initiate list for ensembl IDs
enid = []
# initiate dataframe for results
card = []
dups = []

strict = pd.read_csv('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/metabolite_results_n5_p2_07.tsv', sep='\t')
ensembl_ids = strict['ensemblid'].unique()
# Reload or regenerate protein mapper
prot = pd.read_csv('/gpfs/work2/0/aus20644/project/ksharma/brugada/data/protein_name_mapper.tsv', sep='\t')
# Clean mapping
if 'gene_name' in prot.columns:
    prot['gene_name_clean'] = prot['gene_name'].astype(str).str.strip().str.upper()
elif 'gene_name_clean' not in prot.columns:
    raise ValueError("Missing 'gene_name' column in protein mapping file.")

if 'ensembl_id' in prot.columns:
    prot = prot.rename(columns={'ensembl_id': 'ensemblid'})
elif 'ensemblid' not in prot.columns:
    raise ValueError("Missing 'ensemblid' column in protein mapping file.")

# Create mapping dictionary
gene_to_ensembl = prot.set_index('gene_name_clean')['ensemblid'].to_dict()
# Initialize storage
enid = []
card = []
dups = []

# Initialize storage
enid = []
card = []
dups = []

# Loop through cardiac studies
for study in STUDY_ORDER:
    crs_path = os.path.join(local_paths.carmr, study, 'extracted_results', 'results_file.tsv.gz')
    crs = pd.read_csv(crs_path, sep="\t")

    # Clean Exposure and map to Ensembl IDs
    crs['Exposure_clean'] = crs['Exposure'].astype(str).str.strip().str.upper()
    crs['ensemblid'] = crs['Exposure_clean'].map(gene_to_ensembl)

    # Use MyGeneInfo for unmapped genes
    unmapped = crs[crs['ensemblid'].isna()]['Exposure_clean'].unique()
    if len(unmapped) > 0:
        mg = mygene.MyGeneInfo()
        result = mg.querymany(unmapped, scopes='symbol', fields='ensembl.gene', species='human')
        symbol_to_ensembl = {}
        for entry in result:
            if 'ensembl' in entry and not entry.get('notfound', False):
                symbol_to_ensembl[entry['query']] = entry['ensembl'][0]['gene'] if isinstance(entry['ensembl'], list) else entry['ensembl']['gene']
        still_missing = crs['ensemblid'].isna()
        crs.loc[still_missing, 'ensemblid'] = crs.loc[still_missing, 'Exposure_clean'].map(symbol_to_ensembl)

    # Filter by outcomes and metabolite-strict proteins
    crs = crs[crs['Outcome'].isin(OUTCOMES.keys())]
    crs = crs[crs['ensemblid'].isin(ensembl_ids)]
    crs = crs[crs['No. variants'] > local_c.var_threshold]

    # Skip study if no proteins remain
    if crs.empty:
        print(f"Skipping study {study} (no metabolite-strict proteins)")
        continue

    # Separate duplicates
    dup = crs[crs['ensemblid'].isin(enid)]
    crs = crs[~crs['ensemblid'].isin(enid)]
    enid = list(set(enid + crs['ensemblid'].dropna().tolist()))

    crs['study'] = study
    dup['study'] = study
    dups.append(dup)
    card.append(crs)

# Combine final results
card = pd.concat(card, sort=False)

if 'prmp' not in globals():
    prmp = pd.read_csv(prot_path, sep='\t')

ovc = [x for x in prmp.columns if x in card.columns]
card = card.merge(prmp, on=ovc, how='left')

# Format ORs
card.loc[:, 'OR'] = np.exp(card['Point estimate'])
card.loc[:, 'LCI'] = np.exp(card['Lower bound'])
card.loc[:, 'UCI'] = np.exp(card['Upper bound'])
# Create formatted OR (like in genome-wide MR)
dor_card = {}
for index, row in card.iterrows():
    ids = row['ensemblid']
    dor_card[ids] = fm._format_estimates(
        point=row['Point estimate'],
        se=row['Standard error'],
        exp=True
    )

card['format_OR'] = card['ensemblid'].map(dor_card)


# Optional formatting if 'RUCKER_uniq_analysis' is available
if 'RUCKER_uniq_analysis' in card.columns:
    dor = {}
    for index, row in card.iterrows():
        ids = row['RUCKER_uniq_analysis']
        dor[ids] = fm._format_estimates(point=row['point'], se=row['se'], exp=True)
    card.loc[:, 'format_OR'] = card['RUCKER_uniq_analysis'].map(dor)
else:
    print("Warning: 'RUCKER_uniq_analysis' column missing. Skipping OR formatting.")


# Define loose and strict p-value thresholds
pvalue_loose = 0.05 / len(ensembl_ids)
pvalue_strict = pvalue_loose / len(card['Outcome'].unique())
print("Loose p-value threshold:", pvalue_loose)
print("Strict p-value threshold:", pvalue_strict)   
# Add both thresholds to dataframe
card['pvalue_threshold_loose'] = pvalue_loose
card['pvalue_threshold_strict'] = pvalue_strict
card['pvalue_format_loose'] = fm.sci_notation(pvalue_loose)
card['pvalue_format_strict'] = fm.sci_notation(pvalue_strict)

# Save unfiltered results (with both p-values for reference)
file = f'cardiac_results_n{local_c.var_threshold}.tsv'
card.to_csv(os.path.join('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs', file), sep='\t', index=False)

print("Cardiac cis-MR results processed and saved successfully.")

# -------------------------
# Filter using loose threshold
# -------------------------
print('Filter for proteins with p<' + fm.sci_notation(pvalue_loose))
card_loose = card[card['P-value (float)'] < pvalue_loose]
filt_loose = "{:.0e}".format(pvalue_loose).replace('e-', '_')

# Save results with loose threshold
file = f"cardiac_results_n{local_c.var_threshold}_p{filt_loose}.tsv"
card_loose.to_csv(os.path.join('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs', file), sep='\t', index=False)

# Save proteins (loose)
cols = ['ensemblid', 'uniprot_id', 'gene_name', 'uniprot_display_label', 'uniprot_description']
prot_loose = card_loose[cols].drop_duplicates().rename(columns={
    'uniprot_display_label': 'protein_name',
    'uniprot_description': 'protein_description'
})
file = f"cardiac_proteins_n{local_c.var_threshold}_p{filt_loose}.tsv"
prot_loose.to_csv(os.path.join('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs', file), sep='\t', index=False)

# -------------------------
# Filter using strict threshold
# -------------------------
print('Filter for proteins with p<' + fm.sci_notation(pvalue_strict))
card_strict = card[card['P-value (float)'] < pvalue_strict]
filt_strict = "{:.2e}".format(pvalue_strict).replace('e-', '_')

# Save results with strict threshold
file = f"cardiac_results_n{local_c.var_threshold}_p{filt_strict}.tsv"
card_strict.to_csv(os.path.join('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs', file), sep='\t', index=False)

# Save proteins (strict)
prot_strict = card_strict[cols].drop_duplicates().rename(columns={
    'uniprot_display_label': 'protein_name',
    'uniprot_description': 'protein_description'
})
file = f"cardiac_proteins_n{local_c.var_threshold}_p{filt_strict}.tsv"
prot_strict.to_csv(os.path.join('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs', file), sep='\t', index=False)
# Cardiac proteins for Brugada
card_brugada = card_strict[card_strict['Outcome'].str.contains("Brugada", case=False, na=False)]
n_proteins_cardiac = card_brugada['ensemblid'].nunique()
print("Proteins associated with Brugada outcome:", n_proteins_cardiac)

# --- Begin: compute intersection of proteins associated with metabolites and cardiac outcomes ---

import pandas as pd

# Defensive checks: ensure the dataframes exist
if 'strict_filtered' not in globals():
    raise RuntimeError("strict_filtered (metabolite->protein results) not found. Run metabolite section first.")
if 'card_strict' not in globals():
    raise RuntimeError("card_strict (cardiac cis-MR results) not found. Run cardiac section first.")

# Choose the column containing a stable protein identifier (ensembl id preferred)
prot_col = 'ensemblid'
if prot_col not in strict_filtered.columns or prot_col not in card_strict.columns:
    # try alternatives
    candidates = ['uniprot_id', 'ensembl_id', 'gene_name']
    found = None
    for c in candidates:
        if c in strict_filtered.columns and c in card_strict.columns:
            found = c
            break
    if found is None:
        raise RuntimeError("No shared protein identifier column found in strict_filtered and card_strict. "
                           "Available columns strict_filtered: {}, card_strict: {}".format(
                               list(strict_filtered.columns), list(card_strict.columns)))
    prot_col = found
print(f"Using protein identifier column: {prot_col}")

# Create sets of unique proteins for each dataset
metab_proteins = set(strict_filtered[prot_col].dropna().unique())
cardiac_proteins = set(card_strict[prot_col].dropna().unique())

# Intersection and unique sets
both_proteins = metab_proteins.intersection(cardiac_proteins)
only_metab = metab_proteins - cardiac_proteins
only_card = cardiac_proteins - metab_proteins

# Print counts
print(f"Metabolite-associated proteins (strict): {len(metab_proteins)}")
print(f"Cardiac-associated proteins (strict):    {len(cardiac_proteins)}")
print(f"Proteins associated with BOTH:          {len(both_proteins)}")
print(f"Proteins only in metabolite set:        {len(only_metab)}")
print(f"Proteins only in cardiac set:           {len(only_card)}")

# Produce a dataframe listing the intersecting proteins with metadata (if available)
# Prefer to pull gene_name / uniprot_display_label / description if present
md_cols = [prot_col]
for c in ['gene_name', 'uniprot_id', 'uniprot_display_label', 'uniprot_description']:
    if c in strict_filtered.columns or c in card_strict.columns:
        md_cols.append(c)

# build metadata by combining both tables (prefer strict_filtered for metadata, fallback to card_strict)
meta_df = strict_filtered[[c for c in md_cols if c in strict_filtered.columns]].drop_duplicates(subset=[prot_col])
if prot_col in card_strict.columns:
    extra = card_strict[[c for c in md_cols if c in card_strict.columns]].drop_duplicates(subset=[prot_col])
    # merge missing fields
    meta_df = meta_df.merge(extra, on=prot_col, how='outer', suffixes=('_metab','_card'))

# filter to intersection
both_df = meta_df[meta_df[prot_col].isin(both_proteins)].copy()

# If gene_name not present, try to create from columns
if 'gene_name' not in both_df.columns:
    if 'uniprot_display_label' in both_df.columns:
        both_df['gene_name'] = both_df['uniprot_display_label']
    elif 'uniprot_id' in both_df.columns:
        both_df['gene_name'] = both_df['uniprot_id']

# Add simple provenance columns: count of significant associations per dataset
both_df['n_metab_hits'] = both_df[prot_col].map(lambda x: int((strict_filtered[prot_col] == x).sum()))
both_df['n_card_hits'] = both_df[prot_col].map(lambda x: int((card_strict[prot_col] == x).sum()))

# Save results to file(s)
out_dir = local_paths.results if hasattr(local_paths, 'results') else '/tmp'
both_file = os.path.join(out_dir, 'proteins_both_metabolite_and_cardiac_strict.tsv')
both_df.to_csv(both_file, sep='\t', index=False)
print(f"Saved intersection table to: {both_file}")

# Also save lists
with open(os.path.join(out_dir, 'proteins_both_list.txt'), 'w') as f:
    for p in sorted(both_proteins):
        f.write(str(p) + '\n')

with open(os.path.join(out_dir, 'proteins_only_metab_list.txt'), 'w') as f:
    for p in sorted(only_metab):
        f.write(str(p) + '\n')

with open(os.path.join(out_dir, 'proteins_only_card_list.txt'), 'w') as f:
    for p in sorted(only_card):
        f.write(str(p) + '\n')

print("Also saved plain lists for both / only_metab / only_card in results folder.")

# Optional: show top examples (first 10)
print("\nExample intersecting proteins (first 10 rows):")
print(both_df.head(10))




# ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ## Subset proteins for associating with same outcome as metabolite
# ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
# print('Subset proteins for associating with same outcome as metabolite')
# print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

# # Filter genome-wide MR results by p-value threshold
# p_thresh = brugada['pvalue_threshold_loose'].unique()[0] if 'pvalue_threshold_loose' in brugada.columns else brugada['pvalue_threshold'].unique()[0]
# metout = brugada[brugada['P-value (float)'] < p_thresh].copy()

# par = {}
# res = []
# mets = []
# meta = []
# fin = []

# for m in met:
#     meto = metout[metout['Exposure'] == m]
#     par[m] = meto['Outcome'].tolist()

#     metpro = metab[metab['Outcome'] == m]
#     prom = metpro['ensemblid'].tolist()
#     prout = card[card['ensemblid'].isin(prom)].copy()
#     prout.loc[:, 'metabolite'] = m
#      # Keep only proteins that appear in BOTH
#     common_proteins = set(metpro['ensemblid']).intersection(set(prout['ensemblid']))
#     metpro = metpro[metpro['ensemblid'].isin(common_proteins)]
#     prout  = prout[prout['ensemblid'].isin(common_proteins)]


#     for out in par[m]:
#         new = prout[prout['Outcome'] == out]
#         prot = new['ensemblid'].unique().tolist()

#         mmr = metpro[metpro['ensemblid'].isin(prot)].copy()
#         mmr.loc[:, 'cardiac'] = out

#         for pr in prot:
#             tem = meto[(meto['Outcome'] == out)].copy()
#             tem.loc[:, 'ensemblid'] = pr
#             if tem.empty:
#                 continue
#             oem = tem.sort_values('P-value (float)').iloc[0]  # Use row, not dataframe

#             rpm = mmr[(mmr['ensemblid'] == pr) & (mmr['Outcome'] == m)].copy()
#             if rpm.empty:
#                 continue
#             opm = rpm.sort_values('P-value (float)').iloc[0]

#             rpy = new[(new['ensemblid'] == pr) & (new['Outcome'] == out)].copy()
#             if rpy.empty:
#                 continue
#             opy = rpy.sort_values('P-value (float)').iloc[0]

#             dmy = oem['OR']
#             dpm = opm['Point estimate']
#             dpy = opy['OR']

#             gwr = [
#                 oem['Exposure'],
#                 oem['No. variants'],
#                 oem['format_OR'],
#                 oem['P-value - Supported'],
#                 # oem['format_out'],
#                 "GW"
#             ]
#             prn = [
#                 pr,
#                 opm['gene_name'],
#                 opm['uniprot_id'],
#                 opm['uniprot_display_label'],
#                 opm['uniprot_description'],
#                 opm['study']
#             ]
#             cmr = [
#                 opm['gene_name'],
#                 opm['No. variants'],
#                 # opm['format_results'],
#                 opm['P-value - Supported'],
#                 opm['Exposure'],
#                 opm['study']
#             ]
#             ccr = [
#                 opy['gene_name'],
#                 opy['No. variants'],
#                 opy['format_OR'],
#                 opy['P-value - Supported'],
#                 # opy['format_out'],
#                 opy['study']
#             ]

#             if dmy > 1:
#                 if dpm > 0 and dpy > 1:
#                     mets.append(tem)
#                     meta.append(rpm)
#                     res.append(rpy)
#                     fin.append(prn + gwr + cmr + ccr)
#                 elif dpm < 0 and dpy < 1:
#                     mets.append(tem)
#                     meta.append(rpm)
#                     res.append(rpy)
#                     fin.append(prn + gwr + cmr + ccr)
#             elif dmy < 1:
#                 if dpm > 0 and dpy < 1:
#                     mets.append(tem)
#                     meta.append(rpm)
#                     res.append(rpy)
#                     fin.append(prn + gwr + cmr + ccr)
#                 elif dpm < 0 and dpy > 1:
#                     mets.append(tem)
#                     meta.append(rpm)
#                     res.append(rpy)
#                     fin.append(prn + gwr + cmr + ccr)

# # # Combine prn, ccr, and cmr into a new DataFrame, keeping only rows where 'gene_name' is present in all three

# # # Convert prn, ccr, cmr to DataFrames for easier handling
# # prn_df = pd.DataFrame([prn], columns=[pr, 'gene_name', 'uniprot_id', 'uniprot_display_label', 'uniprot_description', 'study'])
# # ccr_df = pd.DataFrame([ccr], columns=['gene_name', 'No. variants', 'format_OR', 'P-value - Supported', 'study'])
# # cmr_df = pd.DataFrame([cmr], columns=['gene_name', 'No. variants', 'P-value - Supported', 'Exposure', 'study'])

# # # Find intersection of gene_name across all three
# # gene_names = set(prn_df['gene_name']) & set(ccr_df['gene_name']) & set(cmr_df['gene_name'])

# # print(f"Number of common gene_names across all three datasets: {len(gene_names)}")

# # # Filter each DataFrame to only those gene_names
# # prn_filt = prn_df[prn_df['gene_name'].isin(gene_names)]
# # ccr_filt = ccr_df[ccr_df['gene_name'].isin(gene_names)]
# # cmr_filt = cmr_df[cmr_df['gene_name'].isin(gene_names)]

# # # Merge on gene_name (inner join to keep only those present in all)
# # combined_df = prn_filt.merge(ccr_filt, on='gene_name', suffixes=('_prn', '_ccr'))
# # combined_df = combined_df.merge(cmr_filt, on='gene_name', suffixes=('', '_cmr'))

# # print("Combined DataFrame shape after merging: ", combined_df.head())

# # # Now combined_df contains only rows where gene_name is present in all three

# # combined_df.to_csv('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/test_table.tsv', sep = '\t', header = 0)

# # combine results
# res_df = pd.concat(res, sort = False)
# mets_df = pd.concat(mets, sort = False)
# meta_df = pd.concat(meta, sort = False)

# ovc = [x for x in prmp.columns if x in mets_df.columns]
# mets_df = mets_df.merge(prmp, on = ovc, how = 'left')
# ovc = [x for x in prmp.columns if x in meta_df.columns]
# meta_df = meta_df.merge(prmp, on = ovc, how = 'left')
# only_in_met = set(prot['ensemblid']) - set(card['ensemblid'])
# print("Proteins only in metabolite cis-MR:", only_in_met)

# only_in_card = set(card['ensemblid']) - set(prot['ensemblid'])
# print("Proteins only in cardiac cis-MR:", only_in_card)

# overlap = set(prot['ensemblid']).intersection(set(card['ensemblid']))
# print("Proteins in both:", overlap)

# # # Prepare final combined summary dataframe
# # fin_df = pd.DataFrame(
# #     fin,
# #     columns=[
# #         'Ensembl ID', 'Gene name', 'Uniprot ID', 'Protein name', 'Protein description', 'Cardiac cis Study',
# #         'GW Metabolite', 'GW No. variants', 'GW OR', 'GW p-value', 'GW outcome', 'GW Study',
# #         'metabolite cis Gene name', 'metabolite cis No. variants', 'metabolite cis MD', 'metabolite cis p-value', 'metabolite cis Metabolite', 'metabolite cis Study',
# #         'cardiac cis Gene name', 'cardiac cis No. variants', 'cardiac cis OR', 'cardiac cis p-value', 'cardiac cis Outcome', 'cardiac cis Study'
# #     ]
# # )
# # # Extract all genes from final combined results
# # genes_final = fin_df['Ensembl ID'].unique()

# # # Genes present in cardiac cis-MR
# # genes_card = card['ensemblid'].unique()
# # # Genes present in metabolite cis-MR
# # genes_metab = metab['ensemblid'].unique()

# # # Check if any genes in final results are missing in cardiac or metabolite cis-MR
# # missing_in_card = [g for g in genes_final if g not in genes_card]
# # missing_in_metab = [g for g in genes_final if g not in genes_metab]

# # print(f"Genes in final results missing in cardiac cis-MR: {missing_in_card}")
# # print(f"Genes in final results missing in metabolite cis-MR: {missing_in_metab}")


# # # # Save results
# # # res_df.to_csv(os.path.join('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs', 'cardiac_results_metab.tsv'), sep='\t')
# # # mets_df.to_csv(os.path.join('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs', 'gw_results_carmet.tsv'), sep='\t')
# # # meta_df.to_csv(os.path.join('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs', 'metab_results_cardiac.tsv'), sep='\t')
# # fin_df.to_csv(os.path.join('/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs', 'TableX_combined.tsv'), sep='\t', index=False)

# # # Summary output
# # unique_proteins = res_df['ensemblid'].nunique()
# # print(f"{unique_proteins} unique proteins concordant with metabolite outcome")