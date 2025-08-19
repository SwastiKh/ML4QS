import os #interaction with the operating system
import pandas as pd #to work with tables and spreadsheets in python
from scipy.stats import norm #imports normal distribution functions from SciPy's statistics module (probability calculations and stats involving bell curves)
from typing import Any, List, Type, Union, Tuple, Dict #imports tools to label variable types in the code, making it easier to understand and debug
from collections import OrderedDict # remembers the order in which the items are added (sp. type of dictionary)
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##Paths
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Project_paths:
    '''Setting project paths'''
    here = '/gpfs/work2/0/aus20644/project/ksharma/brugada'
    scratch = '/lustre/scratch/scratch/rmgpmva'
    data = os.path.join(here, 'data')
    results = os.path.join(here, 'results', 'outputs')
    figures = os.path.join(here,  'results', 'figures')
    metmr = os.path.join(here, 'cis_MR_metabolites')
    carmr = os.path.join(here, 'cis_MR')
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Constants
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Constant(object):
    """constants used 
    """
    cmtoinch      = 0.393700787 #converting centimeters to inches
    exposure      = 'exposure' #keyword to identify exposure variable
    outcome       = 'outcome'  #keyword to identify outcome variable
    var_threshold = 5 #variance threshold for filtering variables
    variance_npc  = 86 #number of principal components to use for variance filtering (total metabolites/2)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Outcomes
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OUTCOMES = {
    'brugada' : 'Brugada syndrome',
    'QRS'     : 'QRS duration',
}
 
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Exposures and Classes
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Loading exposure and class data...")
exp = pd.read_csv(os.path.join(Project_paths.data, 'exp_file.txt'), encoding = 'ISO-8859-1', sep='\t', header=0, index_col=0)
print("exp.head(): \n")
print(exp.head())
print("\n")
print("exp.index: \n")
print(exp.index)
EXPOSURES = dict(zip(exp.index, exp.short_name))
CLASSES = dict(zip(exp.index, exp.met_class))

print(f"Found {len(EXPOSURES)} exposures and {len(CLASSES)} classes in the exposure file.")
print(f"EXPOSURES: {EXPOSURES}")
print(f"CLASSES: {CLASSES}")

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Proteins
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

prot = pd.read_csv(os.path.join(Project_paths.data, 'proteins.tsv'),sep = '\t', header = 0, index_col = 0)
ENSEMBL = dict(zip(prot.index, prot.uniprot_id))
PROTEINS = dict(zip(prot.uniprot_id, prot.protein_name))

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Orders
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OUTCOME_ORDER = ['brugada', 'QRS']
EXPOSURE_ORDER = exp.index.to_list()
STUDY_ORDER = [ 'decode', 'ukb', 'ahola_olli', 'framingham_combined', 'gudjonsson', 'interval', 'bretherick', 'scallop', 'gilly', 'yang']
STUDY_NAMES = { 'decode'             :  'DECODE',
               'ukb'                 :  'UK Biobank',
               'ahola_olli'          :  'Ahola-Olli',
               'framingham_combined' : 'Framingham',
               'gudjonsson'          : 'AGES-Reykjavik',
               'interval'            : 'Interval',
               'bretherick'          : 'Bretherick',
               'scallop'             : 'SCALLOP',
               'gilly'               : 'Gilly',
               'yang'                : 'Yang' 
               }

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Instrument column names
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

INST_NAMES = {'chr_name'                :    'Chromosome',
              'start_pos'               :    'Start position',
              'effect_allele'           :    'Effect allele',
              'other_allele'            :    'Other allele',
              'effect_size_exposure'    :   'Effect size exposure',
               'standard_error_exposure': 'Standard error exposure',
               'p_value_exposure'       : 'P-value exposure',
               'effect_size_outcome'    : 'Effect size outcome',
               'standard_error_outcome' : 'Standard error outcome',
               'p_value_outcome'        : 'P-value outcome',
               'q_i'                    : 'Variant-specific Q statistic',
               'leverage_i'             : 'Variant-specific leverage statistic',
               'cooks_i'                : 'Variant-specific Cook\'s distance',
               'merit_idx'              : 'Merit index',
                'filename'              : "File name" ,
              }

INSTYPES = {'Chromosome'                :    'Int64',
              'Start position'          :    'Int64',
              'Effect allele'           :     'str',
              'Other allele'            :     'str',
              'Effect size exposure'    :    'float64',
              'Standard error exposure' : 'float64',
              'P-value exposure'        : 'float64',
              'Effect size outcome'     : 'float64',
              'Standard error outcome'  : 'float64',
              'P-value outcome'         : 'float64',
              'Q statistic'             : 'float64',
              'Leverage statistic'      : 'float64',
              'Cook\'s distance'        : 'float64',
              'Merit index'             : 'float64',
             }

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Colors and shapes
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CLASS_COL = {
        'Acylcarnitines'              : '#BF3429',
        'Amino acids'                 : '#006BAD',
        'Biogenic amines'             : '#E87930',
        'Lysophosphatidylcholines'    : '#027949',
        'Phosphatidylcholines'        : '#6B6CA3',
        'Sphingomyelins'              : '#5D90A0',
        }

