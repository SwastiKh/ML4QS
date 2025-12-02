#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Script: reactome_subs_single.py
# Purpose: Find pathways enriched in a subset of proteins (single input file)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import (
    test_proportions_2indep,
    confint_proportions_2indep,
)
import statsmodels.stats.multitest as smm
from project_constants import Project_paths as local_paths
from project_constants import ColumnNames


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helpers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def format_file(input_file: str) -> pd.DataFrame:
    """Load Reactome export and return a long table of (Pathway name, uniprot_id)."""
    df = pd.read_csv(input_file, sep="\t")
    cols = ["Pathway identifier", "Pathway name", "Submitted entities found"]
    df = df[cols].dropna(subset=["Submitted entities found"])
    # map files (optional; keep if you need them elsewhere)
    pathway_map_df = df.drop(columns=["Submitted entities found"]).drop_duplicates()
    protein_df = (
        df.drop(columns=["Pathway identifier"])
        .drop_duplicates()
        .rename(columns={"Submitted entities found": "uniprot_id"})
    )
    # split multi-IDs per row
    protein_df["uniprot_id"] = protein_df["uniprot_id"].str.split(";")
    protein_df = (
        protein_df.explode("uniprot_id")
        .assign(uniprot_id=lambda d: d["uniprot_id"].str.strip())
        .reset_index(drop=True)
    )

    # optional debug exports
    protein_df.to_csv(os.path.join(local_paths.results, "protein_pathways.tsv"), sep="\t", index=False)
    pathway_map_df.to_csv(os.path.join(local_paths.results, "pathway_map.tsv"), sep="\t", index=False)
    return protein_df


def proportion_difference(row: pd.Series, n_subset: int, n_other: int) -> pd.Series:
    """Attach (wald_stat, p_val) and CI for diff in proportions. Robust to zeros."""
    if n_subset <= 0 or n_other <= 0:
        row["test"] = (np.nan, np.nan)
        row["CI"] = (np.nan, np.nan)
        return row
    try:
        row["test"] = test_proportions_2indep(
            count1=int(row[ColumnNames.num_pathway_matches_in_community]),
            nobs1=int(n_subset),
            count2=int(row[ColumnNames.num_other_matches]),
            nobs2=int(n_other),
            return_results=False,
            method="score",
            compare="diff",
            correction=False,
        )
    except Exception:
        row["test"] = (np.nan, np.nan)
    try:
        row["CI"] = confint_proportions_2indep(
            count1=int(row[ColumnNames.num_pathway_matches_in_community]),
            nobs1=int(n_subset),
            count2=int(row[ColumnNames.num_other_matches]),
            nobs2=int(n_other),
            method="newcombe",
            compare="diff",
            correction=False,
        )
    except Exception:
        row["CI"] = (np.nan, np.nan)
    return row


def get_pathways_for_community(
    all_proteins_pathways: pd.DataFrame,
    proteins_of_interest: set,
    community_num,
    protein_col: str = "uniprot_id",
) -> pd.DataFrame:
    """
    Compute enrichment (difference in proportions) for pathways between:
      - proteins_of_interest
      - all OTHER proteins in all_proteins_pathways
    Returns a tidy DataFrame with stats + CI.
    """
    # subset masks
    mask_subset = all_proteins_pathways[protein_col].isin(proteins_of_interest)
    subset_df = all_proteins_pathways.loc[mask_subset].copy()
    other_df = all_proteins_pathways.loc[~mask_subset].copy()

    # sizes
    n_subset = subset_df[protein_col].nunique()
    n_other = other_df[protein_col].nunique()

    # counts for subset
    comm_counts = (
        subset_df.groupby(ColumnNames.pathway_name, as_index=False)
        .agg(
            **{
                ColumnNames.num_pathway_matches_in_community: (protein_col, "count"),
                ColumnNames.pathway_matches_in_community: (protein_col, lambda s: list(map(str, s))),
            }
        )
    )
    # proportions for subset (safe when n_subset==0)
    comm_counts[ColumnNames.percentage_matches_in_community] = (
        comm_counts[ColumnNames.num_pathway_matches_in_community] / (n_subset if n_subset else 1)
    )

    # counts for others
    other_counts = (
        other_df.groupby(ColumnNames.pathway_name, as_index=False)
        .agg(**{ColumnNames.num_other_matches: (protein_col, "count")})
    )
    other_counts["Percentage other matches"] = (
        other_counts[ColumnNames.num_other_matches] / (n_other if n_other else 1)
    )

    # merge; ensure expected columns exist even if subset side is empty
    df = other_counts.merge(comm_counts, how="left", on=ColumnNames.pathway_name)
    # fill well-typed defaults
    if ColumnNames.num_pathway_matches_in_community not in df:
        df[ColumnNames.num_pathway_matches_in_community] = 0
    df[ColumnNames.num_pathway_matches_in_community] = (
        df[ColumnNames.num_pathway_matches_in_community].fillna(0).astype(int)
    )
    if ColumnNames.pathway_matches_in_community not in df:
        df[ColumnNames.pathway_matches_in_community] = [[] for _ in range(len(df))]
    df[ColumnNames.pathway_matches_in_community] = df[ColumnNames.pathway_matches_in_community].apply(
        lambda x: x if isinstance(x, list) else []
    )
    if ColumnNames.percentage_matches_in_community not in df:
        df[ColumnNames.percentage_matches_in_community] = 0.0
    df[ColumnNames.percentage_matches_in_community] = df[
        ColumnNames.percentage_matches_in_community
    ].fillna(0.0)

    # delta (numeric) then stats
    df["delta_num"] = (
        df[ColumnNames.percentage_matches_in_community] - df["Percentage other matches"]
    )
    df = df.apply(lambda r: proportion_difference(r, n_subset, n_other), axis=1)

    # split stats, compute significance properly (CI excludes 0)
    df[["wald_statistic", "p_val"]] = df["test"].apply(
        lambda x: pd.Series(x) if isinstance(x, tuple) and len(x) == 2 else pd.Series([np.nan, np.nan])
    )
    df[["lower", "upper"]] = df["CI"].apply(
        lambda x: pd.Series(x) if isinstance(x, tuple) and len(x) == 2 else pd.Series([np.nan, np.nan])
    )
    df.drop(columns=["test", "CI"], inplace=True)

    df[ColumnNames.sig] = (df["lower"] > 0) | (df["upper"] < 0)
    df[ColumnNames.enriched] = (df["delta_num"] > 0) & df[ColumnNames.sig]

    # round numeric, then pretty delta string
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].round(3)
    df[ColumnNames.delta] = df.apply(
        lambda r: f"{r['delta_num']} ({r['lower']}; {r['upper']})", axis=1
    )

    # printable list of members in community for this pathway
    df[ColumnNames.pathway_matches_in_community] = df[ColumnNames.pathway_matches_in_community].apply(
        lambda lst: ", ".join(lst)
    )

    # meta
    df[ColumnNames.num_proteins_in_community] = n_subset
    df[ColumnNames.proteins_in_community] = ", ".join(sorted(map(str, proteins_of_interest)))

    # ordering: enriched first, better p first, then delta magnitude
    df = df.sort_values(
        by=[ColumnNames.enriched, "p_val", "delta_num"],
        ascending=[False, True, False],
        kind="mergesort",
    ).reset_index(drop=True)

    return df


def attach_labels_vectorized(df_in: pd.DataFrame, prmp: pd.DataFrame) -> pd.DataFrame:
    """
    From df with a column ColumnNames.pathway_matches_in_community (comma-separated UniProt IDs),
    create two new columns: 'Uniprot ID in community' (deduped) and 'Protein name in community'
    using prmp mapping (expects columns 'uniprot_id' and 'uniprot_display_label').
    """
    col_ids = ColumnNames.pathway_matches_in_community
    if col_ids not in df_in.columns:
        df_in["Uniprot ID in community"] = np.nan
        df_in["Protein name in community"] = np.nan
        return df_in

    tmp = df_in.copy()
    tmp["_id_list"] = tmp[col_ids].fillna("").apply(lambda s: [x for x in s.split(", ") if x] if s else [])
    expl = tmp[tmp["_id_list"].str.len() > 0].copy()
    if expl.empty:
        tmp["Uniprot ID in community"] = ""
        tmp["Protein name in community"] = ""
        return tmp.drop(columns=["_id_list"])

    expl = expl.explode("_id_list").rename(columns={"_id_list": "uniprot_id"})
    expl = expl.merge(prmp[["uniprot_id", "uniprot_display_label"]], on="uniprot_id", how="left")

    agg = expl.groupby(expl.index).agg(
        **{
            "Uniprot ID in community": ("uniprot_id", lambda s: ", ".join(sorted(set(s.dropna().astype(str))))),
            "Protein name in community": (
                "uniprot_display_label",
                lambda s: ", ".join(sorted(set(s.dropna().astype(str)))),
            ),
        }
    )
    tmp = tmp.join(agg)
    tmp[["Uniprot ID in community", "Protein name in community"]] = tmp[
        ["Uniprot ID in community", "Protein name in community"]
    ].fillna("")
    return tmp.drop(columns=["_id_list"])


def safe_fdr(df: pd.DataFrame, p_col: str = "p_val", out_col: str = "pval_adjust") -> pd.DataFrame:
    """Benjamini–Hochberg with NaN-safe handling."""
    df = df.copy()
    df[out_col] = np.nan
    mask = df[p_col].notna().values
    if mask.any():
        _, adj = smm.fdrcorrection(df.loc[mask, p_col].values, method="indep")
        df.loc[mask, out_col] = adj
    return df


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data loading
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Mappings (expects columns at least: 'uniprot_id', 'uniprot_display_label', and ideally 'ensemblid')
prmp = pd.read_csv(
    "/gpfs/work2/0/aus20644/project/ksharma/brugada/data/protein_name_mapper.tsv",
    sep="\t",
    header=0,
)

# Reactome pathways export
pid = pd.read_csv(
    "/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/reactome_all.txt",
    sep="\t",
)
pid = pid[["Pathway identifier", "Pathway name"]].drop_duplicates()

# Long table (Pathway name, uniprot_id)
df = format_file("/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/reactome_all.txt")
# ensure the protein column name
df = df.rename(columns={"Protein": "uniprot_id"})

# Datasets of interest (all in UniProt space)
psig = pd.read_csv(
    "/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/metabolite_results_n5_p2_07.tsv",
    sep="\t",
)
plis = set(psig["uniprot_id"].dropna().astype(str))

pcar = pd.read_csv(
    "/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/cardiac_results_n5_p2.04_05.tsv",
    sep="\t",
)
pcar_prots = set(pcar["uniprot_id"].dropna().astype(str))

# Outcomes table (Ensembl IDs -> map to UniProt)
out = pd.read_csv(
    "/gpfs/work2/0/aus20644/project/ksharma/brugada/results/outputs/TableX_combined_test.tsv",
    sep="\t",
    header=0,
    index_col=None,
)

# Build an Ensembl->UniProt mapper if available
ensembl_to_uniprot = {}
if "ensemblid" in prmp.columns:
    enspairs = prmp[["ensemblid", "uniprot_id"]].dropna()
    # there can be many-to-many; we keep all possible mappings
    ensembl_to_uniprot = (
        enspairs.groupby("ensemblid")["uniprot_id"]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )

# Dictionary of proteins per phenotype in UniProt space
prot_out = {}
if {"cardiac cis Outcome", "Ensembl ID"}.issubset(out.columns):
    for pheno in out["cardiac cis Outcome"].dropna().unique():
        ens = set(out.loc[out["cardiac cis Outcome"] == pheno, "Ensembl ID"].dropna().astype(str))
        if ensembl_to_uniprot:
            uni = set().union(*(ensembl_to_uniprot.get(e, set()) for e in ens))
        else:
            # fallback: assume the column already holds UniProt IDs (best-effort)
            uni = ens
        prot_out[pheno] = uni


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Per-phenotype enrichment (using prot_out)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

for pheno, uni_set in prot_out.items():
    if not uni_set:
        # still write an empty file for traceability
        empty = pd.DataFrame(columns=[ColumnNames.pathway_name])
        empty.to_csv(os.path.join(local_paths.results, f"{pheno}_enrichment.tsv"), sep="\t", index=False)
        continue

    tmp = get_pathways_for_community(df, uni_set, community_num=pheno, protein_col="uniprot_id")
    # FDR (safe)
    tmp = safe_fdr(tmp, p_col="p_val", out_col="pval_adjust")
    # add identifiers and labels
    tmp = attach_labels_vectorized(tmp, prmp)
    # add pathway identifier
    tmp = tmp.merge(pid, on="Pathway name", how="left")
    # order by adjusted p-value
    tmp = tmp.sort_values(["pval_adjust", "delta_num"], ascending=[True, False])

    # export
    tmp.to_csv(os.path.join(local_paths.results, f"{pheno}_enrichment.tsv"), sep="\t", na_rep="-", index=False)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Aggregate comparisons: metabolic vs all, cardiac vs all, prioritised(outcomes) vs all
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

enr = get_pathways_for_community(df, plis, community_num=0, protein_col="uniprot_id")
enr = safe_fdr(enr, p_col="p_val", out_col="pval_adjust")
enr = attach_labels_vectorized(enr, prmp).merge(pid, on="Pathway name", how="left")
enr = enr.sort_values(["pval_adjust", "delta_num"], ascending=[True, False])

car = get_pathways_for_community(df, pcar_prots, community_num=5, protein_col="uniprot_id")
car = safe_fdr(car, p_col="p_val", out_col="pval_adjust")
car = attach_labels_vectorized(car, prmp).merge(pid, on="Pathway name", how="left")
car = car.sort_values(["pval_adjust", "delta_num"], ascending=[True, False])

# prioritised = all outcome proteins mapped to UniProt
pri_set = set().union(*(v for v in prot_out.values())) if prot_out else set()
pri = get_pathways_for_community(df, pri_set, community_num=6, protein_col="uniprot_id") if pri_set else pd.DataFrame()
if not pri.empty:
    pri = safe_fdr(pri, p_col="p_val", out_col="pval_adjust")
    pri = attach_labels_vectorized(pri, prmp).merge(pid, on="Pathway name", how="left")
    pri = pri.sort_values(["pval_adjust", "delta_num"], ascending=[True, False])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Save outputs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# enr.to_csv(os.path.join(local_paths.results, "metabolite_protein_enrichment_.tsv"), sep="\t", index=False)
car.to_csv(os.path.join(local_paths.results, "cardiac_protein_enrichment_.tsv"), sep="\t", index=False)
if not pri.empty:
    pri.to_csv(os.path.join(local_paths.results, "prioritised_protein_enrichment_.tsv"), sep="\t", index=False)
