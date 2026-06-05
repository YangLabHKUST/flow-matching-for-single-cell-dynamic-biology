# Final Check for Chapter 4.2 Small Figures

Scope: final paper-ready formatting pass for `figures/ch04/new2`. Data, selected examples, and numeric values were not changed.

| Figure | Formats complete | Layout QA | Claim | Metric/display-space note |
|---|---:|---|---|---|
| `fig4_2_toy_pca30_representative_pairs` | yes | Pass: no clipping/overlap detected; legends/notes placed outside data or in clear white-backed regions. | Toy PCA-30 endpoint links show one representation-specific coupling over the same cells. | Metric: standardized PCA-30; display: toy 2D state. |
| `fig4_2_toy_program4_representative_pairs` | yes | Pass: no clipping/overlap detected; legends/notes placed outside data or in clear white-backed regions. | Program-4 changes the highest-mass target links for the matched toy source cells. | Metric: standardized Program-4; display: toy 2D state. |
| `fig4_2_toy_representation_coupling_summary` | yes | Pass: no clipping/overlap detected; legends/notes placed outside data or in clear white-backed regions. | Toy representation changes the coupling matrix, support size, and entropy. | Metrics: coupling L1/top-k over matched cells; support/entropy from each coupling. |
| `fig4_2_eb_pc20_coupling_representative_pairs` | yes | Pass: no clipping/overlap detected; legends/notes placed outside data or in clear white-backed regions. | EB PC-20 coupling examples are defined in standardized PC-20 and shown only in PHATE. | Metric: standardized PC-20; display: PHATE 2D only. |
| `fig4_2_eb_phate_diagnostic_coupling_representative_pairs` | yes | Pass: no clipping/overlap detected; legends/notes placed outside data or in clear white-backed regions. | PHATE-induced links are a contrastive diagnostic, not EB training geometry. | Metric: PHATE 2D diagnostic; display: PHATE 2D only; not training geometry. |
| `fig4_2_eb_pc_vs_phate_distance_summary` | yes | Pass: no clipping/overlap detected; legends/notes placed outside data or in clear white-backed regions. | EB coupling distances differ between standardized PC-20 training space and PHATE display space. | Metrics: standardized PC-20 and PHATE display distances, explicitly separated. |
| `fig4_2_state_space_model_readout_summary` | yes | Pass: no clipping/overlap detected; legends/notes placed outside data or in clear white-backed regions. | Toy model readouts are summarized separately from representation-native metrics. | Metrics: native MMD scoped within representation; shared readout uses toy Program-4 scores. |
| `fig4_3_toy_single_pair_chord_vs_graph_path` | yes | Pass: no clipping/overlap detected; legends/notes placed outside data or in clear white-backed regions. | Same toy endpoints yield different intermediate states under chord versus graph path families. | Metric/display: toy 2D state; fixed endpoint pair. |
| `fig4_3_eb_chord_vs_graph_matched_examples` | yes | Pass: no clipping/overlap detected; legends/notes placed outside data or in clear white-backed regions. | Matched EB endpoints have distinct PC-20 chord and graph diagnostic paths when displayed in PHATE. | Metric/path: standardized PC-20; display: PHATE 2D only. |
| `fig4_3_eb_density_radius_delta` | yes | Pass: no clipping/overlap detected; legends/notes placed outside data or in clear white-backed regions. | Straight PC-20 chords are farther from observed support in density-radius percentile. | Metric: standardized PC-20 support diagnostic; no PHATE metric. |
| `fig4_3_eb_knn_radius_delta` | yes | Pass: no clipping/overlap detected; legends/notes placed outside data or in clear white-backed regions. | Straight PC-20 chords have larger support distance by kNN radius. | Metric: standardized PC-20 kNN radius; no PHATE metric. |
| `fig4_3_eb_off_manifold_positive_fraction` | yes | Pass: no clipping/overlap detected; legends/notes placed outside data or in clear white-backed regions. | The straight-minus-graph support-distance effect is positive for a majority of EB examples. | Metric: standardized PC-20 straight-minus-graph deltas; no PHATE metric. |

File sizes (bytes):

```json
{
  "fig4_2_eb_pc20_coupling_representative_pairs": {
    "pdf": 48705,
    "png": 877927,
    "svg": 66632
  },
  "fig4_2_eb_pc_vs_phate_distance_summary": {
    "pdf": 13182,
    "png": 350071,
    "svg": 17893
  },
  "fig4_2_eb_phate_diagnostic_coupling_representative_pairs": {
    "pdf": 48449,
    "png": 888663,
    "svg": 66645
  },
  "fig4_2_state_space_model_readout_summary": {
    "pdf": 14518,
    "png": 389468,
    "svg": 26610
  },
  "fig4_2_toy_pca30_representative_pairs": {
    "pdf": 19387,
    "png": 414446,
    "svg": 26839
  },
  "fig4_2_toy_program4_representative_pairs": {
    "pdf": 19459,
    "png": 417101,
    "svg": 26841
  },
  "fig4_2_toy_representation_coupling_summary": {
    "pdf": 18691,
    "png": 302771,
    "svg": 15333
  },
  "fig4_3_eb_chord_vs_graph_matched_examples": {
    "pdf": 88593,
    "png": 1449504,
    "svg": 176657
  },
  "fig4_3_eb_density_radius_delta": {
    "pdf": 20752,
    "png": 364088,
    "svg": 26134
  },
  "fig4_3_eb_knn_radius_delta": {
    "pdf": 20158,
    "png": 342927,
    "svg": 25545
  },
  "fig4_3_eb_off_manifold_positive_fraction": {
    "pdf": 13714,
    "png": 231193,
    "svg": 13702
  },
  "fig4_3_toy_single_pair_chord_vs_graph_path": {
    "pdf": 28732,
    "png": 499659,
    "svg": 47743
  }
}
```
