from __future__ import annotations

import numpy as np

from revprint.page_interactions import (
    _body_registration_mask,
    _registration_decision,
    _registration_signal,
)


def test_registration_decision_rejects_nan_error() -> None:
    applied, conf, reason = _registration_decision(
        float("nan"),
        np.array([0.0, 0.0], dtype=np.float64),
        1200,
        800,
        0.05,
    )
    assert applied is False
    assert conf == 0.0
    assert reason == "invalid_registration_error"


def test_body_registration_mask_reports_usable_coverage() -> None:
    ink = np.zeros((240, 200), dtype=np.float32)
    ink[60:180, 70:130] = 0.7
    _mask, coverage = _body_registration_mask(ink)
    assert coverage > 0.01


def test_registration_signal_has_texture_response() -> None:
    gray = np.full((120, 120), 240, dtype=np.uint8)
    gray[30:90, 45:75] = 150
    sig = _registration_signal(gray)
    assert float(sig.mean()) > 0.01
    assert float(sig[60, 60]) > float(sig[5, 5])
