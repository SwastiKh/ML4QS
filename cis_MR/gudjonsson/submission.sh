#!/bin/bash
# Source in the task setup, this will deal with the command line input
# and cleanup functions, there are several variables that it will give
# access to:
# JA="$1"
# TMP_LOC="$2"
# STEP=$3
# IDX=${4:-$SGE_TASK_ID}

# WORKING_DIR : A location

# Store all the paths to the intermediate files, these will all be deleted when
# exitting cleanly or with an error. The "" is to stop unbound errors
# TEMPFILES=("")

# Files that should be moved should go into these two arrays. This will then be
# moved to their final location on exit.
# MOVE_FROM=()
# MOVE_TO=()

# PROCESS_LINE : An array extracted from the job array file, this provides
# input for the current STEP of the starting from the current TASK

# ROW_IDX : Element 0 from the process line, this is a requirement that all
# job arrays have this

# Enable nounset option
set -u

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
run_command() {
    # Decompose the line array, this is really for code clarity
    # Make sure order matches columns in jaf.txt file (ordered in selcol object in jaf_preparation.py)
    INFILE=($(echo "${PROCESS_LINE[1]}" | tr '|' ' '))
    OUTFILE="${PROCESS_LINE[2]}"
    EXPOSURE="${PROCESS_LINE[3]}"
    EXP_PATH="${PROCESS_LINE[4]}"
    GENE="${PROCESS_LINE[5]}"
    UP="${PROCESS_LINE[6]}"
    DOWN="${PROCESS_LINE[7]}"
    PVAL="${PROCESS_LINE[8]}"
    LPVAL="${PROCESS_LINE[9]}"
    MAF="${PROCESS_LINE[10]}"
    LDCUT="${PROCESS_LINE[11]}"
    LDSAMPLE="${PROCESS_LINE[12]}"
    LDSEED="${PROCESS_LINE[13]}"
    REF_POP="${PROCESS_LINE[14]}"
    SAMPLE_LIST="${PROCESS_LINE[15]}"
    PROXY_DIST="${PROCESS_LINE[16]}"
    DROP_VAR="${PROCESS_LINE[17]}"
    MODELS="${PROCESS_LINE[18]}"
    MODELS_KWARGS="${PROCESS_LINE[19]}"
    STEIGER_PVAL="${PROCESS_LINE[20]}"
   
   
    info_msg "# infiles=""${#INFILE[@]}"
    info_msg "outfile="$OUTFILE
    info_msg "idx (SGE_TASK_ID) [${IDX}] == rowidx [${ROW_IDX}]"
    
    # This generates a temp outfile name in out working directory that we
    # will then move to the final location upon successful completion of the
    # task
    OUTFILE_BASE="$(basename "$OUTFILE")"
    TEMP_OUTFILE="$WORKING_DIR"/"$OUTFILE_BASE"
    
    info_msg "output file basename=""$OUTFILE_BASE"
    info_msg "temporary output file=""$TEMP_OUTFILE"
    
    # Add the outfiles to our moving arrays
    MOVE_FROM+=("$TEMP_OUTFILE")
    MOVE_TO+=("$OUTFILE")
    
    # Also add to temp files just in case we exit with an error
    TEMPFILES+=("$TEMP_OUTFILE")
    # The call to any other scripts that are being run
    # NOTE: CHECK if env set, otherwise use sensible default
    # Allowing for unset variables
    set +u
    if [ ! -v "$MERIT_INTERPRETER" ]; then
        MERIT_INTERPRETER="${HOME}/miniforge3/envs/merit_py39/bin/python"
    fi
    if [ ! -v "$APPLICATION" ]; then
        APPLICATION="${HOME}/software/merit-helper/merit_helper/pipelines/run_mr.py"
    fi
    
    set -u
    "${MERIT_INTERPRETER}" "${APPLICATION}" --input "${INFILE[@]}" --output "${OUTFILE}" \
    --exposure "${EXPOSURE}" --exposure_path "${EXP_PATH}" --up "${UP}" --down "${DOWN}" \
    --pvalue "${PVAL}" --maf "${MAF}" --ensemblid "${GENE}" --logpval "${LPVAL}" \
    --ld_cut "${LDCUT}" --ld_sample "${LDSAMPLE}" --ld_seed "${LDSEED}" \
    --proximity_dist "${PROXY_DIST}" --drop_variants "${DROP_VAR}" \
    --sample_list "${SAMPLE_LIST}" --ref_pop "${REF_POP}" \
    --steiger_pvalue "${STEIGER_PVAL}" --models "${MODELS}" \
    --models-kwargs "${MODELS_KWARGS}"
 
info_msg "finished running application: ${APPLICATION}"
}

. task_setup.sh
