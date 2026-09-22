import numpy as np

from anomaly.monitoring import DriftMonitor


def test_window_and_shift_warning(drift_reference, caplog):
    monitor = DriftMonitor(drift_reference, window_size=64)
    incoming = np.random.default_rng(20).normal(size=(64, 16)) + 20
    for feature in incoming[:-1]:
        assert monitor.update(feature)["checked"] is False
    result = monitor.update(incoming[-1])
    assert result["checked"] and result["last_check"]["drift_detected"]
    assert result["pending_images"] == 0
    assert "feature drift detected" in caplog.text


def test_unchanged_reference_does_not_warn(drift_reference):
    monitor = DriftMonitor(drift_reference, window_size=len(drift_reference))
    for feature in drift_reference:
        result = monitor.update(feature)
    assert result["last_check"]["drift_detected"] is False
