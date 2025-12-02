#!/bin/bash
#SBATCH --job-name Druggability
#SBATCH --output=/hpc/dhl_ec/mvanvugt/scripts/logs/Druggability.log
#SBATCH --time= 30:00:00
#SBATCH --mem=30G

# --- Set paths ---
INPUT="/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/TableX_combined_.tsv"   # your input file
RESULTS="/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs"    # directory for output
CHEMBL_DB="/gpfs/work2/0/aus20644/data/resources/chembl_35.db"
BNF_DB="/gpfs/work2/0/aus20644/data/resources/bnf_20210409.db"

# --- Fix column name in input file ---
sed -i '1s/Uniprot ID/uniprot_id/' "${INPUT}"

# --- Extract drug summary ---
bm-drug-target-summary \
	-i ${INPUT} \
	-o ${RESULTS}/drug_summary_final.tsv \
	-v \
	-p uniprot_id \
	--chembl ${CHEMBL_DB} \
	--bnf ${BNF_DB} \
	-M

# --- Extract drug effects ---
bm-drug-target-effects \
	-i ${INPUT} \
	-o ${RESULTS}/drug_effects_final.tsv \
	-v \
	-p uniprot_id \
	--chembl ${CHEMBL_DB} \
	--bnf ${BNF_DB} \
	-M

echo "Drug summary and drug effects extraction completed."
echo "Files saved to ${RESULTS}/drug_summary_2.tsv and ${RESULTS}/drug_effects_2.tsv"

