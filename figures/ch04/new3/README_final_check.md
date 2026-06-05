# Chapter 4 Final Small Figure Package

Output directory: `figures/ch04/new3`

Claim boundary:
- Raw observed EB counts are sampling-depth proxies, not calibrated biological abundance.
- Equal-depth subsampling changes the mass convention while preserving state-bin composition diagnostics.
- Raw-count growth proxies and WFR-FM growth readouts depend on the input mass convention.
- Stochastic bridge width is a separate synthetic normalized path-family assumption.

Generated figure stems:
- `fig4_11a_raw_observed_counts`: Raw destructive snapshot counts are sampling-depth proxies, not calibrated biological abundance.
- `fig4_11b_equal_depth_composition`: Equal-depth subsampling changes the mass convention while preserving state-bin composition diagnostics.
- `fig4_11c_sampling_depth_bootstrap_sensitivity`: Raw-count growth proxies are sensitive to equal-depth bootstrap intervals under the sampling-depth diagnostic.
- `fig4_11d_wfrfm_raw_minus_equal_growth_heatmap`: WFR-FM growth readout changes under raw-depth minus equal-depth mass convention.
- `fig4_11e_wfrfm_mass_convention_agreement_summary`: WFR-FM rank agreement is high while signs and calibration remain convention-dependent.
- `fig4_11f_stochastic_bridge_demo`: Stochastic bridge width is a separate synthetic normalized path-family assumption from EB mass convention.

Source tables/json checked:
- `outputs/ch04/table4_6_eb_downsampling_diagnostics.csv`
- `outputs/ch04/table4_6b_eb_bridge_sampling_diagnostics.csv`
- `outputs/ch04/table4_6c_wfrfm_growth_by_bin_full.csv`
- `outputs/ch04/table4_6d_wfrfm_sampling_sensitivity_full.csv`
- `outputs/ch04/wfrfm_sampling_sensitivity_summary_full.json`
- `outputs/ch04/cache/exp10_stochastic_bridge_manifest.json`

QA checks:
- Required PNG/PDF/SVG files exist and are nonzero.
- Output filenames do not imply a composite figure.
- This README and `final_polish_manifest.csv` were written.
