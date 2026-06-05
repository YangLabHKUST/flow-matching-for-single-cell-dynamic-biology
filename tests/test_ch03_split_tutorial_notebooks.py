from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVED_MONOLITH_PATH = (
    PROJECT_ROOT
    / "archive"
    / "notebooks_retired_20260604_ch03_split"
    / "03_flow_matching_from_scratch.ipynb"
)


SPLIT_SPECS = {
    "03_1_toy_flow_matching_from_scratch.ipynb": {
        "min_code_cells": 18,
        "required_headings": [
            "2D Toy Sanity Check",
            "Conditional Velocity Versus Marginal Velocity",
            "CFM Object Hierarchy",
        ],
        "required_artifacts": [
            "fig_toy_loss.png",
            "fig_toy_evolution.png",
            "fig03_02_conditional_vs_marginal_toy.png",
            "fig03_03_cfm_object_hierarchy_toy.png",
            "artifact_manifest_03_1_toy_flow_matching_from_scratch.csv",
            "run_config_03_1_toy_flow_matching_from_scratch.json",
        ],
        "forbidden_headings": [
            "Load EB Data",
            "CNF-Endpoint Baseline",
            "Time Sampling Strategy Ablation",
        ],
    },
    "03_2_eb20d_main_flow_matching.ipynb": {
        "min_code_cells": 20,
        "required_headings": [
            "Data audit",
            "Train/validation split",
            "Endpoint pairing",
            "Model config",
            "Training loop",
            "Loss table",
            "Loss figure",
            "Model cache",
            "Endpoint pair visualization",
            "Sampling and population evolution",
            "Euler step sensitivity",
        ],
        "required_artifacts": [
            "ch03_eb_timepoint_counts.csv",
            "ch03_eb20d_train_val_split.csv",
            "ch03_eb20d_training_log.csv",
            "ch03_euler_step_sensitivity.csv",
            "figB1_eb20d_train_val_loss.png",
            "fig03_04_eb_endpoint_pairs_phate.png",
            "fig03_08_eb_population_evolution_phate.png",
            "fig03_09_euler_step_sensitivity_phate.png",
            "ch03_eb20d_velocity_mlp_seed42.pt",
            "ch03_eb20d_main_config_seed42.json",
            "artifact_manifest_03_2_eb20d_main_flow_matching.csv",
            "run_config_03_2_eb20d_main_flow_matching.json",
        ],
        "forbidden_headings": [
            "2D Toy Sanity Check",
            "CNF-Endpoint Baseline",
            "Time Sampling Strategy Ablation",
            "Network Capacity Ablation",
        ],
    },
    "03_3_eb20d_baselines_ablations_and_claim_audit.ipynb": {
        "min_code_cells": 24,
        "required_headings": [
            "Solver Comparison Lite",
            "CNF-Endpoint Baseline",
            "Time Sampling Strategy Ablation",
            "Network Capacity Ablation",
            "Trajectory Straightness in 20D",
            "Chapter 3 Artifact Index",
            "Validation Notes and Summary",
        ],
        "required_artifacts": [
            "fig03_10_nfe_vs_endpoint_error.png",
            "figE1_cfm_vs_cnf_endpoint_samples_phate.png",
            "figE1_cfm_vs_cnf_endpoint_training_cost.png",
            "figE2_capacity_endpoint_mmd.png",
            "figE2_capacity_val_mse.png",
            "figE3_time_sampling_distributions.png",
            "figE3_time_sampling_endpoint_mmd.png",
            "figE3_time_sampling_final_bar.png",
            "figE3_time_sampling_val_mse.png",
            "figE5_endpoint_distance_vs_straightness.png",
            "figE5_representative_trajectories_phate.png",
            "figE5_straightness_hist.png",
            "table03_01_solver_diagnostics.csv",
            "tableE1_cfm_vs_cnf_endpoint.csv",
            "tableE3_time_sampling_ablation.csv",
            "tableT2_training_hyperparams_capacity.csv",
            "tableE5_trajectory_straightness.csv",
            "paper_table03_01_solver_diagnostics.csv",
            "paper_table03_01_solver_diagnostics.md",
            "paper_table03_01_solver_diagnostics.tex",
            "paper_tableE1_cfm_vs_cnf_endpoint.csv",
            "paper_tableE1_cfm_vs_cnf_endpoint.md",
            "paper_tableE1_cfm_vs_cnf_endpoint.tex",
            "paper_tableE3_time_sampling_ablation.csv",
            "paper_tableE3_time_sampling_ablation.md",
            "paper_tableE3_time_sampling_ablation.tex",
            "paper_tableT2_training_hyperparams_capacity.csv",
            "paper_tableT2_training_hyperparams_capacity.md",
            "paper_tableT2_training_hyperparams_capacity.tex",
            "paper_tableE5_trajectory_straightness_summary.csv",
            "paper_tableE5_trajectory_straightness_summary.md",
            "paper_tableE5_trajectory_straightness_summary.tex",
            "ch03_artifact_index.csv",
            "ch03_flow_matching_from_scratch_run_summary.json",
            "artifact_manifest_03_3_eb20d_baselines_ablations_and_claim_audit.csv",
        ],
        "forbidden_headings": [
            "2D Toy Sanity Check",
            "Train EB 20D VelocityMLP",
        ],
    },
}

LEGACY_CH03_STATIC_FIGURES = {
    "fig03_01_training_vs_sampling_compute.png",
    "fig03_05_velocity_mlp_architecture.png",
    "fig03_06_minimal_cfm_training_loop.png",
    "fig03_11_cfm_extension_map.png",
}


def _payload(filename: str) -> dict:
    return json.loads((PROJECT_ROOT / "notebooks" / filename).read_text())


def _payload_from_path(path: Path) -> dict:
    return json.loads(path.read_text())


def _sources(filename: str, cell_type: str | None = None) -> list[str]:
    payload = _payload(filename)
    return [
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell_type is None or cell.get("cell_type") == cell_type
    ]


def test_ch03_split_notebooks_exist_and_are_tutorial_sized():
    for filename, spec in SPLIT_SPECS.items():
        payload = _payload(filename)
        code_sources = _sources(filename, "code")
        markdown_text = "\n".join(_sources(filename, "markdown"))
        code_lengths = [len(source.splitlines()) for source in code_sources]

        assert payload["nbformat"] >= 4
        assert len(code_sources) >= spec["min_code_cells"], filename
        assert max(code_lengths) <= 90, (filename, max(code_lengths))

        for heading in spec["required_headings"]:
            assert heading in markdown_text, (filename, heading)
        for heading in spec["forbidden_headings"]:
            assert heading not in markdown_text, (filename, heading)
        for index, source in enumerate(code_sources, 1):
            compile(source, f"{filename}:code-cell-{index}", "exec")


def test_ch03_split_notebooks_cover_artifacts_without_overlap():
    artifact_owners: dict[str, str] = {}

    for filename, spec in SPLIT_SPECS.items():
        code_text = "\n".join(_sources(filename, "code"))
        markdown_text = "\n".join(_sources(filename, "markdown"))
        notebook_text = code_text + "\n" + markdown_text

        assert "from IPython.display import Image, display" in code_text, filename
        assert "expected_figures" in code_text, filename
        assert "expected_tables" in code_text or "expected_outputs" in code_text, filename
        assert "raise FileNotFoundError" in code_text, filename
        assert "display_table(" in code_text, filename

        display_markers = [
            "display(Image(",
            "display_saved_figure(",
            "display_saved_figures(",
            "display_png(",
        ]
        assert any(marker in code_text for marker in display_markers), filename

        for artifact in spec["required_artifacts"]:
            assert artifact in code_text, (filename, artifact)
            previous_owner = artifact_owners.setdefault(artifact, filename)
            assert previous_owner == filename, (artifact, previous_owner, filename)

    expected_all = {artifact for spec in SPLIT_SPECS.values() for artifact in spec["required_artifacts"]}
    assert set(artifact_owners) == expected_all


def test_ch03_split_notebooks_cover_legacy_monolith_artifact_contract():
    archived_payload = _payload_from_path(ARCHIVED_MONOLITH_PATH)
    archived_text = "\n".join("".join(cell.get("source", [])) for cell in archived_payload["cells"])
    split_text = "\n".join(
        "\n".join(_sources(filename))
        for filename in SPLIT_SPECS
    )

    expected_legacy_artifacts = {
        artifact
        for spec in SPLIT_SPECS.values()
        for artifact in spec["required_artifacts"]
        if not artifact.startswith("artifact_manifest_") and not artifact.startswith("run_config_")
    }
    expected_legacy_artifacts.update(LEGACY_CH03_STATIC_FIGURES)

    for artifact in expected_legacy_artifacts:
        assert artifact in archived_text or artifact in split_text, artifact
        assert artifact in split_text, artifact


def test_ch03_manifest_contracts_are_explicit_not_summary_only():
    for filename, spec in SPLIT_SPECS.items():
        code_text = "\n".join(_sources(filename, "code"))
        manifest_start = code_text.rfind("expected_figures")
        manifest_region = code_text[manifest_start:] if manifest_start >= 0 else ""
        assert "expected_figures" in manifest_region, filename
        assert "expected_tables" in manifest_region or "expected_outputs" in manifest_region, filename
        assert "check_required_artifacts" in code_text, filename

        for artifact in spec["required_artifacts"]:
            if artifact.startswith("artifact_manifest_"):
                continue
            assert artifact in manifest_region or artifact in code_text, (filename, artifact)


def test_ch03_shared_tutorial_helpers_are_generic(tmp_path):
    from src import ch03_tutorial as tutorial

    for name in [
        "json_ready",
        "save_json",
        "save_csv",
        "display_saved_figure",
        "display_saved_figures",
        "check_required_artifacts",
    ]:
        assert hasattr(tutorial, name), name

    json_path = tutorial.save_json(tmp_path / "nested" / "payload.json", {"x": 1})
    csv_path = tutorial.save_csv(tmp_path / "nested" / "table.csv", [{"a": 1}, {"a": 2}])

    assert json_path.exists()
    assert csv_path.exists()

    manifest = tutorial.check_required_artifacts(
        expected_figures=[json_path],
        expected_tables=[csv_path],
    )
    assert set(manifest["kind"]) == {"figure", "table"}
    assert manifest["exists"].all()


def test_old_ch03_monolith_is_retired_index_only():
    active_payload = _payload("03_flow_matching_from_scratch.ipynb")
    active_text = "\n".join("".join(cell.get("source", [])) for cell in active_payload["cells"])
    active_code_sources = [
        "".join(cell.get("source", []))
        for cell in active_payload["cells"]
        if cell.get("cell_type") == "code"
    ]

    assert "retired" in active_text.lower()
    for filename in SPLIT_SPECS:
        assert filename in active_text
    assert len(active_code_sources) <= 1
    assert "Train EB 20D VelocityMLP" not in active_text
    assert "CNF-Endpoint Baseline" not in active_text
    assert "save_figure(" not in active_text
    assert "train_velocity_model" not in active_text

    archived_payload = _payload_from_path(ARCHIVED_MONOLITH_PATH)
    archived_text = "\n".join("".join(cell.get("source", [])) for cell in archived_payload["cells"])
    archived_code_sources = [
        "".join(cell.get("source", []))
        for cell in archived_payload["cells"]
        if cell.get("cell_type") == "code"
    ]
    assert len(archived_code_sources) >= 10
    assert "Train EB 20D VelocityMLP" in archived_text
    assert "CNF-Endpoint Baseline" in archived_text
