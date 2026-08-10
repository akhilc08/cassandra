"""Oracle evaluation pipeline — quality gates, hallucination detection, calibration.

Submodule imports are LAZY. `import oracle.evaluation.pnl` used to drag in every
module here, and `calibration` pulls aiosqlite, so the backtest and forward test
could not run anywhere the full platform was not installed. That failed in CI while
passing locally, because the dev venv happened to have aiosqlite.

`from oracle.evaluation import TradeGate, ...` still works; the import cost is just
deferred to first attribute access (PEP 562).
"""

from __future__ import annotations

from typing import Any

_LAZY_EXPORTS = {
    "CalibrationData": "oracle.evaluation.calibration",
    "CalibrationMonitor": "oracle.evaluation.calibration",
    "GateResult": "oracle.evaluation.gates",
    "TradeGate": "oracle.evaluation.gates",
    "HallucinationDetector": "oracle.evaluation.hallucination",
    "HallucinationResult": "oracle.evaluation.hallucination",
    "EvaluationJudge": "oracle.evaluation.judge",
    "EvaluationResult": "oracle.evaluation.judge",
    "PostMortem": "oracle.evaluation.post_mortem",
    "PostMortemGenerator": "oracle.evaluation.post_mortem",
    "PostResolutionEvaluator": "oracle.evaluation.post_resolution",
    "ResolutionResult": "oracle.evaluation.post_resolution",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value
