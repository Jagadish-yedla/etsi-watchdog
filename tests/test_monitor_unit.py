# tests/test_monitor_unit.py
import pytest
import numpy as np
from datetime import datetime, timedelta

from concept_drift.monitoring.etsi_monitor import ETSIConceptDriftMonitor, DriftSeverity, DriftAlert

@pytest.fixture
def monitor():
    cfg = {
        'drift_detection': {'detection_methods': ['ddm'], 'window_size': 100},
        'alerts': {'thresholds': {'performance_decline': 0.05}},
        'metrics_history_size': 1000,
        'general': {'service_name': 'test_drift', 'log_level': 'WARNING'}
    }
    m = ETSIConceptDriftMonitor(cfg)
    return m

def test_register_and_set_baseline(monitor):
    monitor.register_model('m1', {'name': 'm1'})
    monitor.set_baseline('m1', {'accuracy': 0.9})
    assert 'm1' in monitor.monitored_models
    assert monitor.model_baselines['m1']['accuracy'] == 0.9

def test_evaluate_no_drift(monitor):
    monitor.register_model('m2', {'name': 'm2'})
    # perfect predictions -> accuracy 1.0
    y_true = np.array([0,1,0,1])
    y_pred = np.array([0,1,0,1])
    res = monitor.evaluate_predictions('m2', y_true, y_pred)
    assert res['model_id'] == 'm2'
    assert 'current_metrics' in res
    assert res['current_metrics']['accuracy'] == pytest.approx(1.0, rel=1e-6)
    # no performance alerts expected
    assert isinstance(res['alerts'], list)

def test_evaluate_triggers_performance_alert(monitor):
    monitor.register_model('m3', {'name': 'm3'})
    # set a high baseline, then feed low-accuracy predictions
    monitor.set_baseline('m3', {'accuracy': 0.98})
    y_true = np.array([0,1,0,1,0,1,0,1])
    # always predict 0 -> accuracy 50%
    y_pred = np.zeros_like(y_true)
    res = monitor.evaluate_predictions('m3', y_true, y_pred)
    # Expect at least one alert for performance degradation
    assert any(getattr(a, 'drift_type', '') == 'performance' for a in res['alerts'])

def test_generate_alerts_persists(monkeypatch, monitor):
    monitor.register_model('m4', {'name': 'm4'})
    # fake SessionLocal to capture adds
    added = []
    class DummySession:
        def add(self, o): added.append(o)
        def commit(self): pass
        def rollback(self): pass
        def close(self): pass

    monkeypatch.setattr('concept_drift.monitoring.etsi_monitor.SessionLocal', lambda: DummySession())

    drift_results = {'drift_detected': True, 'detectors_triggered': ['ddm'], 'confidence': 0.8}
    baseline_comparison = {'has_baseline': True, 'changes': {'accuracy': {'degraded': True, 'percent_change': -20}}}
    current_metrics = {'accuracy': 0.5}

    alerts = monitor._generate_alerts('m4', drift_results, baseline_comparison, current_metrics)

    assert len(alerts) >= 1
    # our DummySession should have seen the DB model objects added
    assert len(added) >= 1
