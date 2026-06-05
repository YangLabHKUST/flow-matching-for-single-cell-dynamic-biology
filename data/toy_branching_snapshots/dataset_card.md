# Toy Branching Snapshots Dataset

This directory stores reusable toy data assets shared across the tutorial chapters.
The directory is organized by dataset, not by chapter; chapter notebooks may copy
selected tables to `outputs/chXX/` as run artifacts.

Files:

- `observed_2d_snapshots.csv`: destructive 2D branching population snapshots used
  by Chapter 1 and as the control condition source for Chapter 2.
- `hidden_same_cell_paths_for_fig01_02.csv`: hypothetical same-cell paths used only
  as a visual aid for Chapter 1 Figure 1.2. These are not observed trajectories
  and are not training supervision.
- `observed_2d_schema.csv`: schema for the observed 2D snapshot table.
- `observed_2d_counts_by_time.csv`: per-timepoint count diagnostics for the
  observed 2D snapshots.
- `conditioned_snapshot_table.csv`: two-condition snapshot metadata table used by
  Chapter 2. The control condition is derived from `observed_2d_snapshots.csv`;
  the perturbed condition is generated deterministically with a higher rare-fate
  fraction.
- `branching_toy_pseudocounts.h5ad`: toy-derived pseudo-count AnnData object for
  Chapter 2 preprocessing and sampler examples. It is not real scRNA-seq data.
