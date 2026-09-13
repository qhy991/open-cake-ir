"""Text presentation of the task audit's descriptive performance projection."""
from __future__ import annotations

import json


def primary_summary(performance) -> str:
    """Keep metric, status and missing coverage visible without inventing a score."""
    rows = [{key: row.get(key) for key in ("run_id", "candidate_id", "role", "primary_score")}
            for row in performance.get("rows", ())]
    return "Task performance: " + json.dumps({
        "policy": performance.get("policy"), "rows": rows,
        "missing": performance.get("missing", []),
        "ranking_scope": performance.get("ranking_scope"),
        "threshold_status": performance.get("threshold_status"),
    }, sort_keys=True, allow_nan=False)


