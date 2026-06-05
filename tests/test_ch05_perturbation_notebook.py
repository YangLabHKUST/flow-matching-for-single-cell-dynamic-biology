from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "05_2_perturbation_response_sciplex.ipynb"


def _notebook_text() -> str:
    payload = json.loads(NOTEBOOK_PATH.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def _setup_cell() -> str:
    payload = json.loads(NOTEBOOK_PATH.read_text())
    return "".join(payload["cells"][2].get("source", []))


def test_ch05_2_notebook_is_split_b_c_perturbation_only():
    assert NOTEBOOK_PATH.exists()
    text = _notebook_text()

    assert "Section 5.2 Perturbation response prediction with sci-Plex" in text
    assert "Split B held-out highest dose" in text
    assert "Split C held-out compound" in text
    assert "pd.concat([split_b_metrics, split_c_metrics], ignore_index=True)" in text

    assert "Split A random sanity" not in text
    assert "split_a_metrics" not in text
    assert "load_eb_ch05" not in text
    assert "run_eb_" not in text
    assert "EB_PATH" not in text
    assert "tab_5_1_" not in text


def test_ch05_2_notebook_declares_only_perturbation_artifacts():
    text = _notebook_text()

    for required in [
        "tab_5_2_sciplex_splits.csv",
        "sciplex_metrics_by_group.csv",
        "sciplex_metrics_summary.csv",
        "fig_5_3_sciplex_heldout_compound_summary.png",
        "fig_5_3_sciplex_heldout_compound.png",
        "run_summary_perturbation_sciplex.json",
    ]:
        assert required in text

    for removed in [
        "fig_5_1_eb_pairwise_vs_shared.png",
        "fig_5_1_skip_pair_ablation.png",
        "fig_5_1_main_suite.png",
        "tab_5_1_main_suite.csv",
    ]:
        assert removed not in text


def test_ch05_2_defaults_to_full_section52_reproduction_config():
    setup = _setup_cell()

    assert 'DEFAULT_SEED = int(os.environ.get("CH05_SEED", "42"))' in setup
    assert 'QUICK_MODE = os.environ.get("CH05_QUICK", "0") == "1"' in setup
    assert 'TRAINING_STEPS = int(os.environ.get("CH05_TRAINING_STEPS", "6000"))' in setup
    assert 'BATCH_SIZE = int(os.environ.get("CH05_BATCH_SIZE", "256"))' in setup
    assert 'NFE = int(os.environ.get("CH05_NFE", "32"))' in setup
    assert 'SCIPLEX_DOWNLOAD_IN_CH05 = os.environ.get("CH05_SCIPLEX_DOWNLOAD_IN_CH05", "0") == "1"' in setup
    assert 'SCIPLEX_SYNTHETIC_IF_MISSING = os.environ.get("CH05_ALLOW_SYNTHETIC_SCIPLEX", "0") == "1"' in setup
    assert 'MAX_EVAL_GROUPS = None if MAX_EVAL_GROUPS == "" else int(MAX_EVAL_GROUPS)' in setup

    assert '"1500" if QUICK_MODE else "6000"' not in setup
    assert '"128" if QUICK_MODE else "256"' not in setup
    assert '"16" if QUICK_MODE else "32"' not in setup


def test_ch05_2_setup_controls_random_seeds_and_cuda_determinism():
    setup = _setup_cell()

    for required in [
        "import random",
        "random.seed(DEFAULT_SEED)",
        "np.random.seed(DEFAULT_SEED)",
        "torch.manual_seed(DEFAULT_SEED)",
        "torch.cuda.manual_seed_all(DEFAULT_SEED)",
        "torch.backends.cudnn.benchmark = False",
        "torch.backends.cudnn.deterministic = True",
        "torch.use_deterministic_algorithms(True, warn_only=True)",
    ]:
        assert required in setup
