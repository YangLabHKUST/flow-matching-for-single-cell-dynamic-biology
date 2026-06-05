# Flow Matching for Dynamic Biology

Teaching code for a paper-facing tutorial on flow matching for time-resolved single-cell snapshots.

This tree now uses the v2 paper workflow at the repository root. The migrated
notebooks and generated artifacts have replaced the older notebook/result set.

## Layout

    notebooks/   v2 paper-facing notebooks
    scripts/     v2 runners and notebook builders
    src/         shared modules imported by notebooks and scripts
    figures/     generated paper figures
    outputs/     generated run summaries, executed notebooks, and caches
    tables/      generated paper tables
    configs/     YAML configs retained for reusable examples
    data/        reusable datasets, organized by dataset rather than chapter
    tests/       lightweight migration and artifact sanity tests

## Data

Reusable data stays under `data/`. The current v2 workflow uses the EB
time-course assets, sci-Plex A549 assets, LINCS compound metadata, and selected
toy assets through the shared `src/` loaders.

## Design Rules

- Keep training loops explicit and readable.
- Avoid production abstractions, experiment managers, and deep inheritance.
- Keep paper claims tied to generated artifacts in `figures/`, `tables/`, and
  `outputs/`.

## Quick Smoke Test

    python -m pytest -q

The base conda environment used for the current v2 artifacts is
`/home/xmabs/anaconda3/bin/python`.
