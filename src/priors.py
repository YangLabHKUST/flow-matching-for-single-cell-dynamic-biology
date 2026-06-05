from __future__ import annotations

import numpy as np


def marker_monotonicity_score(values):
    values = np.asarray(values, dtype=float)
    diffs = np.diff(values)
    return float((diffs >= 0).mean()) if len(diffs) else 1.0


def forbidden_transition_rate(source_labels, target_labels, forbidden: set[tuple[str, str]]):
    pairs = zip(map(str, source_labels), map(str, target_labels))
    pairs = list(pairs)
    if not pairs:
        return 0.0
    return sum(pair in forbidden for pair in pairs) / len(pairs)
