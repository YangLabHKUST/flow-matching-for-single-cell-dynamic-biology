# TrajectoryNet EB Snapshot Embedding

Source copied from `../baselines/trajectorynet/data/eb_velocity_v5.npz`.

Underlying public dataset: PHATE Embryoid Body time-course, Mendeley Data DOI
`10.17632/v6n743h5ng.1`.

This asset is used here only as real time-indexed single-cell snapshot
embeddings for Chapter 3.

- `phate` is used for 2D visualization.
- `pcs[:, :20]` is used for OT cost computation.
- `sample_labels` are used as time labels.
- `pcs_delta` and `delta_embedding` are not used as supervision.
- No paired cell trajectories are assumed.
