---
name: notebook-tutorialization
description: Use when research Jupyter notebooks need tutorial-style restructuring, especially when cells are too large, figures are only saved not displayed, generic helper code is duplicated, or markdown was added without making the notebook teachable.
---

# Notebook Tutorialization

## Overview

Turn a research notebook into a tutorial notebook that teaches the experiment flow, runs from a fresh kernel, displays the outputs it creates, and preserves exact paper artifacts. Tutorialization is not markdown-only editing: code organization, visible results, reusable helper boundaries, and verification must change when needed.

Use a STitch3D-style narrative shape: biological/scientific question -> setup -> preprocessing/data audit -> model or diagnostic construction -> results tables -> visualization -> artifact/claim audit.

## When To Use

Use this for:

- Splitting one large paper notebook into focused tutorial notebooks.
- Refactoring an experimental notebook so a reader can run it from top to bottom.
- Auditing a notebook that claims to be tutorial-style but still has huge cells, hidden figures, or duplicated boilerplate.
- Adding artifact manifests, run configs, and executed-copy checks.
- Retiring old combined notebooks without losing paper figures, tables, or cached full-run outputs.

Do not use this for ordinary script refactors or for changing scientific claims without re-running the relevant experiments.

## Non-Negotiable Principle

Do not accept "we added explanatory markdown" as tutorialization. A tutorial notebook must let a reader learn by running small, meaningful steps and seeing outputs inline.

Keep the core experiment visible in the notebook:

- model/variant definitions
- pair/topology/split definitions
- training/evaluation calls
- metric row construction
- diagnostic row construction
- claim-table construction calls

Move only generic or presentation-oriented code into `src`:

- path/project-root resolution
- global seed/config helpers
- CSV/JSON serialization
- figure construction and save/display helpers
- table preview helpers
- artifact manifest helpers
- final audit/display-value checks

Do not hide the experiment behind one high-level `run_everything(...)` call unless the user explicitly asks for a production pipeline rather than a tutorial.

## Core Workflow

1. Map the old notebook before editing.
   - Extract markdown headings, experiment sections, code cells, and artifact writes.
   - Compare current and backup versions when available: cell counts, code/markdown character counts, long code cells, image outputs, artifact names.
   - Identify setup cells, shared utility cells, data-loading cells, experiment cells, and final summary/manifest cells.
   - Record which figures, tables, JSON files, caches, and executed notebooks belong to each section.
   - Measure tutorial debt:
     - code cells over 80-100 lines
     - setup cells over 80 lines
     - utility cells that define many generic helpers
     - saved figures without immediate display
     - markdown-only changes where code is unchanged

2. Define tutorial boundaries.
   - Each new notebook should have a single tutorial topic and no cross-topic experiment blocks.
   - Keep common sections consistent: title, short purpose, setup, data loading/preprocessing, experiment design, execution, metrics, visualization, artifact/claim audit.
   - If a later section depends on variables defined in an earlier old section, move the minimal helper or alias into setup/shared utilities.
   - Prefer many small cells over a few giant cells. As a target, keep most code cells under 60 lines and avoid any cell over 90 lines unless it is a clearly justified core experiment block.

3. Refactor helper boundaries before rewriting the notebook.
   - Extract repeated generic helpers to `src/<topic>_tutorial.py` or an existing suitable module.
   - Leave experiment-specific orchestration in the notebook.
   - If multiple notebooks duplicate the same setup/helper block, create one shared helper module rather than copying the block again.
   - Add or update tests for the helper module before relying on it.

4. Rewrite as a step-by-step tutorial.
   - Start with a concrete question and what the reader should watch for.
   - Introduce data and design choices before running models.
   - Show small tables for variants, splits, topologies, or artifact plans.
   - Split long experiment cells into:
     1. define design/config
     2. run core computation
     3. inspect intermediate results
     4. save artifacts
     5. display generated figures/tables
   - Use markdown to explain why the next cell exists, not to summarize a hidden monolith.

5. Display outputs inline.
   - Every newly saved PNG/PDF figure should be shown in the notebook soon after saving, usually with `IPython.display.Image` or a local display helper.
   - Display key summary tables with selected columns, not only `print(path)`.
   - Executed notebooks should contain `image/png` outputs for important generated figures.

6. Preserve paper artifacts explicitly.
   - Build `expected_figures` and `expected_tables` lists for each notebook.
   - Include all正文 figures/tables and relevant supplementary diagnostics for that tutorial.
   - Do not rely on "the file exists somewhere"; the final manifest cell must check it.
   - Keep official artifact names stable unless the user explicitly asks to rename them.

7. Make the notebook independently executable.
   - Run from a fresh kernel using `jupyter nbconvert --to notebook --execute`.
   - Save executed copies under `outputs/<chapter>/...executed.ipynb`.
   - Check every code cell has `execution_count` and no `output_type == "error"`.
   - If execution fails, fix root causes in the notebook rather than assuming previous notebook state.
   - Avoid smoke-mode execution if it overwrites official full-run artifacts with smoke outputs. In that case, use structural checks, read-only cache-loading checks, or execute in a copied output directory.

8. Handle heavy cached artifacts deliberately.
   - Keep full-run caches and full paper artifacts unless the user explicitly asks to delete them.
   - For expensive diagnostics, default to verifying and loading existing full artifacts.
   - Provide an explicit recompute flag, for example `CH04_RECOMPUTE_EXP8B=1`, when recomputation is optional and costly.
   - If a required full artifact is missing, fail loudly with `FileNotFoundError`.

9. Retire old notebooks or generators safely.
   - Search active code with `rg` before moving old notebooks or manifests.
   - Update tests and scripts so they target new split notebooks or standalone scripts.
   - Move old combined notebooks and old combined `artifact_manifest.csv` / `run_config.json` to `archive/` only after active references are gone.
   - If a notebook is now maintained directly, replace old generator scripts with an explicit retired entry point that fails loudly instead of silently overwriting the notebook.

## Tutorial Quality Gates

Before calling the result tutorial-style, check all gates:

- Code changed as needed; not markdown-only unless the original already had tutorial-sized cells and inline outputs.
- Most code cells are under 60 lines; no unexplained giant cells remain.
- Generic helpers are in `src`, not copied through multiple notebooks.
- Core scientific flow remains visible in the notebook.
- Saved figures are displayed inline.
- Key tables are displayed with readable columns.
- Final audit checks required artifacts, non-empty files, and claim/display invariants when applicable.
- Executed copy has no errors and contains expected image outputs.

## Validation Checklist

Run these checks before reporting completion:

- Notebook structure summary:
  - markdown/code cell counts
  - max code-cell length
  - code-cell compile check
  - count of `image/png` outputs in executed copy
- `ls -lh notebooks/<split>.ipynb outputs/<chapter>/*executed.ipynb`
- Python executed-copy check:
  - all code cells have non-null `execution_count`
  - no output has `output_type == "error"`
- Manifest check:
  - split manifest CSV exists and is non-empty
  - no artifact row has `exists=False`
  - no non-`run_config` row has `bytes=0`
- Active-reference check:
  - `rg -n "<old_notebook>|artifact_manifest.csv|run_config.json" notebooks scripts tests src`
- Full test suite:
  - use the project Python, for example `/home/xmabs/anaconda3/bin/python -m pytest -q`

## Tests To Add Or Update

Add tests for process invariants, not just string presence:

- Split notebooks cover all expected experiment headings with no duplicated experiment heading across splits.
- Artifact names are present in the final manifest cell, not merely somewhere in the notebook text.
- A notebook that calls a helper such as `train_or_load_model` defines it locally.
- Aliases such as `X0_eb` and `X0p_eb` are defined before first use in the same notebook.
- Tutorial-style notebooks have enough code cells to be stepwise and no unexplained giant cells.
- Important generated figures are displayed inline in the executed notebook.
- Generic tutorial helper APIs exist in `src` when multiple notebooks would otherwise duplicate them.
- Retired generator scripts do not call `nbf.write` or overwrite directly maintained notebooks.
- Tests should not keep active dependencies on retired combined notebooks.

## Common Failure Modes

- Markdown-only "tutorialization" leaves code unchanged. Fix by splitting cells, extracting generic helpers, and displaying outputs.

- A notebook saves figures but only prints paths. Fix by displaying each important saved figure inline.

- A tutorial hides the core experiment behind a pipeline call. Fix by keeping the design, training/evaluation, metric rows, and diagnostics visible in notebook cells.

- Common helpers are copied into every notebook. Fix by moving generic helpers to `src` and importing them.

- A split notebook works interactively because a previous notebook defined variables, but fails from a fresh kernel.
  Fix by moving minimal helpers and aliases into the split notebook.

- A paper figure is generated in an experiment cell but omitted from `expected_figures`.
  Fix the manifest cell and re-execute the notebook.

- A heavy diagnostic kills the kernel during tutorial verification.
  Prefer loading verified full artifacts by default and gate recomputation behind an explicit environment flag.

- A test still points at the retired combined notebook.
  Retarget the test to the new split notebooks or standalone preparation script.

## Reporting Format

Report concisely:

- which notebooks were created or modified
- cell-count and max-code-cell-length summary
- executed-copy paths
- inline figure-output count or reason execution was not run
- manifest status for each split
- whether active references to old notebooks remain
- full pytest result
- which old files are safe to archive
