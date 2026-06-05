from __future__ import annotations

import json

import pandas as pd
import pytest


METHOD_LABELS = {
    "baseline": "Baseline method",
    "model_a": "Model A",
    "model_b": "Model B",
}


def test_metric_table_for_split_orders_methods_and_reports_missing_labels():
    from src.ch05_sciplex_tutorial import metric_table_for_split, metric_value_table

    summary = pd.DataFrame(
        [
            {"split_name": "heldout", "method": "model_b", "program_readout_mmd": 0.2, "program_readout_sliced_w2": 1.2},
            {"split_name": "heldout", "method": "model_a", "program_readout_mmd": 0.1, "program_readout_sliced_w2": 1.1},
            {"split_name": "other", "method": "baseline", "program_readout_mmd": 0.9, "program_readout_sliced_w2": 1.9},
        ]
    )

    frame, missing = metric_table_for_split(summary, "heldout", ["baseline", "model_a", "model_b"], METHOD_LABELS)

    assert frame["method"].tolist() == ["model_a", "model_b"]
    assert frame["method_label"].tolist() == ["Model A", "Model B"]
    assert missing == ["Baseline method"]
    assert metric_value_table(frame).to_dict(orient="list") == {
        "method": ["Model A", "Model B"],
        "MMD": [0.1, 0.2],
        "Sliced W2": [1.1, 1.2],
    }


def test_make_metric_display_table_from_summary_validates_expected_values(tmp_path):
    from src.ch05_sciplex_tutorial import make_metric_display_table_from_summary

    source = tmp_path / "summary.json"
    rows = [
        {"split_name": "heldout", "method": "model_a", "program_readout_mmd": 0.12344, "program_readout_sliced_w2": 1.2344},
        {"split_name": "heldout", "method": "model_b", "program_readout_mmd": 0.56789, "program_readout_sliced_w2": 2.3456},
    ]
    source.write_text(json.dumps({"key_metrics": {"sciplex_summary": rows}}), encoding="utf-8")

    expected = pd.DataFrame(
        [
            {"method": "model_a", "MMD": 0.1234, "Sliced W2": 1.234},
            {"method": "model_b", "MMD": 0.5679, "Sliced W2": 2.346},
        ]
    )
    frame = make_metric_display_table_from_summary(
        source,
        "heldout",
        ["model_a", "model_b"],
        expected,
        METHOD_LABELS,
        project_root=tmp_path,
    )

    assert frame["method_label"].tolist() == ["Model A", "Model B"]
    assert frame["metric_display_source"].tolist() == ["summary.json", "summary.json"]

    with pytest.raises(ValueError, match="Display metrics do not match"):
        make_metric_display_table_from_summary(
            source,
            "heldout",
            ["model_a", "model_b"],
            expected.assign(MMD=[0.0, 0.0]),
            METHOD_LABELS,
            project_root=tmp_path,
        )


def test_wrapped_labels_are_stable():
    from src.ch05_sciplex_tutorial import short_compound_label, wrapped_method_label

    assert wrapped_method_label("model_a", METHOD_LABELS, width=20) == "Model A"
    assert "\n" in wrapped_method_label("very_long_method_name_without_label", METHOD_LABELS, width=10)
    assert short_compound_label("Compound Name (high dose)", width=16) == "Compound Name\n(high dose)"
