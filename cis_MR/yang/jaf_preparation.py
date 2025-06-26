#!/usr/bin/env python
'''
Preparing Job array file to perform cis-MR to assess the effect of protein expression on Brugada syndrome

'''
#####################################################
# preparing job array file
#####################################################

#imports
import os
import pandas as pd
from pathlib import Path

#constants
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
HOME = '/gpfs/work2/0/aus20644/project/ksharma/brugada'
# NOTE: the output directory
OUTPUT_PATH =os.path.join(HOME,'cis_MR', 'yang', 'results')
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

# ordered columns
selcol =['rowidx', 'infile', 'outfile', 'exposure', 'exposure_path',
         'ensemblid',  'up', 'down', 'pvalue', 'logpval', 'maf',
         'ld_cut', 'ld_sample','ld_seed', 'ref_pop',
         'sample_list', 'proximity_dist', 'drop_variants',
         'models', 'models-kwargs', 'steiger_pvalue'
         ]
###############################################################################
# reading in basic info
###############################################################################
generic_jaf = pd.read_csv( os.path.join(ROOT_DIR,'yang_plasma_pqtl_mapper.tsv'),
            sep='\t', index_col=0)

# remove proteins without data
#empty_prots = pd.read_csv( os.path.join(ROOT_DIR, 'mr_empty_files.txt'))
#index_del = np.unique([r[0] for r in  empty_prots.iloc[:,0].str.split('/')])
#generic_jaf.drop(labels=index_del, inplace=True)

###########################################################################
# preparing input files
###########################################################################
# outcome data file
generic_jaf['infile'] = os.path.join(HOME, 'outcomes.txt')
generic_jaf['ensemblid'] = generic_jaf['ensembl_id']

# exposure index
# exposure index
generic_jaf['exposure_path'] = os.path.join(HOME, 'cis_MR', 'yang', 'yang_plasma_pqtl_mapper.tsv')
generic_jaf['exposure'] = generic_jaf.index

# instrument selection
kb = 1000
generic_jaf['up'] = 200*kb
generic_jaf['down'] =200*kb
generic_jaf['pvalue'] = 4
generic_jaf['logpval'] = True
generic_jaf['maf'] = 0.01
generic_jaf['ld_cut'] = 'None'
generic_jaf['ld_sample'] = 5000
generic_jaf['ld_seed'] = 12052018
generic_jaf['ref_pop'] = 'EUR_UKB_GRCh38_without_homozygosity'
generic_jaf['sample_list'] = 'EUR_UKB_WO_RELATED'

# ### Stuff not used in the actual analysis
generic_jaf['proximity_dist'] = 'None'
generic_jaf['drop_variants'] = 'None'

# models to use
generic_jaf['models'] = "IVW;IVW|IVW;Egger;Egger|Egger"
generic_jaf['models-kwargs'] = 'None'
generic_jaf['steiger_pvalue'] = 0.05

###########################################################################
# outcome specific parts
###########################################################################
generic_jaf['outfile'] = [OUTPUT_PATH + '/' +  str(i).strip() + '.tar.gz' for\
        i in generic_jaf.index.to_list()]

# adding index
jaf = generic_jaf.copy()
jaf['rowidx'] = list(range(1,jaf.shape[0] + 1))
jaf = jaf[selcol]

###############################################################################
# Saving out the jaf file
###############################################################################
jaf.to_csv(os.path.join(HOME,'cis_MR', 'yang', 'jaf.txt'), sep='\t', header=True, index=False)




