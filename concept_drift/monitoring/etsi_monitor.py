# concept_drift/monitoring/etsi_monitor.py
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import yaml
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from prometheus_client import Counter, Gauge, Histogram
import time
import os

# Import the drift detection classes
from concept_drift.detection.drift_detectors import ConceptDriftDetector
from concept_drift.monitoring.db import SessionLocal
from concept_drift.monitoring.models import Alert as AlertModel, Evaluation as EvalModel


# -------------------------
# Prometheus metrics (module-level)
# -------------------------
EVALUATIONS_TOTAL = Counter(
    'drift_evaluations_total',
    'Total number of evaluation batches processed',
    ['model_id']
)

EVALUATION_LATENCY_SECONDS = Histogram(
    'drift_evaluation_latency_seconds',
    'Latency for evaluating a batch of predictions (seconds)',
    ['model_id']
)

DRIFT_DETECTIONS_TOTAL = Counter(
    'drift_detections_total',
    'Number of drift detections (per detector, per model)',
    ['model_id', 'detector']
)

ALERTS_TOTAL = Counter(
    'drift_alerts_total',
    'Number of alerts generated',
    ['model_id', 'severity', 'alert_type']
)

MODEL_HEALTH_SCORE = Gauge(
    'model_health_score',
    'Model health score (0-100)',
    ['model_id']
)


class DriftSeverity(Enum):
    """Enumeration for drift severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    timestamp: datetime
    drift_type: str
    severity: DriftSeverity
    detection_method: str
    confidence_score: float
    affected_model: str
    metrics: Dict[str, Any]
    description: str
    recommended_actions: List[str]


class ETSIConceptDriftMonitor:
    def __init__(self, config: Optional[Union[str, Dict[str, Any]]] = None):
        """
        config: either a dict of settings OR a path (str/Path) to drift_monitor_config.yaml.
                If None, we'll try to load ../config/drift_monitor_config.yaml relative to this file.
        """
        # Accept either dict or path string/Path/None
        if isinstance(config, dict):
            self.config = config
        else:
            # config is expected to be a path (string/Path) or None
            if config:
                cfg_path = Path(config)
            else:
                # default path: project_root/config/drift_monitor_config.yaml
                cfg_path = Path(__file__).parents[2] / 'config' / 'drift_monitor_config.yaml'

            if not cfg_path.exists():
                raise FileNotFoundError(f"Config file not found at: {cfg_path}")

            with open(cfg_path, 'r') as f:
                self.config = yaml.safe_load(f)

        # now proceed to read detection settings etc...
        det_cfg = self.config.get('drift_detection', {})
        methods = det_cfg.get('detection_methods', ['ddm', 'eddm', 'adwin'])
        window = det_cfg.get('window_size', 1000)

        self.concept_drift_detector = ConceptDriftDetector(
            detection_methods=methods,
            window_size=window
        )

        # Alert thresholds
        alerts_cfg = self.config.get('alerts', {}).get('thresholds', {})
        self.alert_thresholds = {
            'performance_decline': alerts_cfg.get('performance_decline', 0.05),
            'error_rate_increase': alerts_cfg.get('error_rate_increase', 0.1),
            'confidence_drop': alerts_cfg.get('confidence_drop', 0.15)
        }

        # Logging
        self.logger = self._setup_logging()

        # Internal state
        self.alerts: List[DriftAlert] = []
        self.monitored_models: Dict[str, Dict] = {}
        self.model_baselines: Dict[str, Dict[str, float]] = {}
        self.metrics_history: Dict[str, Dict[str, List[float]]] = {}

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(self.config.get('general', {}).get('service_name', 'etsi_concept_drift'))
        level = self.config.get('general', {}).get('log_level', 'INFO')
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        if not logger.handlers:
            h = logging.StreamHandler()
            fmt = self.config.get('logging', {}).get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            h.setFormatter(logging.Formatter(fmt))
            logger.addHandler(h)
        return logger

    # -------------------------
    # Helper to convert numpy types -> native Python for JSON/DB
    # -------------------------
    def _to_serializable(self, obj: Any) -> Any:
        """Recursively convert numpy types & arrays into Python native types."""
        if obj is None:
            return None
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (list, tuple)):
            return [self._to_serializable(i) for i in obj]
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        # fallback for datetimes
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    def register_model(self, model_id: str, model_info: Dict[str, Any]):
        """
        Register a model for drift monitoring
        """
        self.monitored_models[model_id] = {
            'info': model_info,
            'registered_at': datetime.now(),
            'last_evaluation': None,
            'drift_history': [],
            'performance_baseline': None
        }

        # Initialize metrics history
        self.metrics_history[model_id] = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'avg_confidence': [],
            'prediction_latency': []
        }

        self.logger.info(f"Registered model {model_id} for drift monitoring")

    def set_baseline(self, model_id: str, baseline_metrics: Dict[str, float]):
        if model_id not in self.monitored_models:
            raise ValueError(f"Model {model_id} not registered")

        self.model_baselines[model_id] = baseline_metrics
        self.monitored_models[model_id]['performance_baseline'] = baseline_metrics

        self.logger.info(f"Set baseline for model {model_id}: {baseline_metrics}")

    def evaluate_predictions(self,
                             model_id: str,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             prediction_confidence: Optional[np.ndarray] = None,
                             feature_data: Optional[np.ndarray] = None,
                             metadata: Optional[Dict] = None) -> Dict[str, Any]:
        if model_id not in self.monitored_models:
            raise ValueError(f"Model {model_id} not registered")

        evaluation_timestamp = datetime.now()

        # Start timer for latency metric
        start_time = time.time()

        # Detect concept drift
        drift_results = self.concept_drift_detector.detect_drift(
            y_true, y_pred, prediction_confidence
        )

        # Calculate additional metrics
        current_metrics = self._calculate_metrics(y_true, y_pred, prediction_confidence)

        # Persist the evaluation record (make metrics serializable)
        db = None
        try:
            db = SessionLocal()
            serial_metrics = self._to_serializable(current_metrics)
            db.add(EvalModel(
                timestamp=evaluation_timestamp,
                model_id=model_id,
                accuracy=str(serial_metrics.get('accuracy')) if serial_metrics.get('accuracy') is not None else None,
                metrics=serial_metrics
            ))
            db.commit()
        except Exception as exc:
            self.logger.exception("Failed to persist evaluation to DB: %s", exc)
            if db is not None:
                try:
                    db.rollback()
                except Exception:
                    pass
        finally:
            if db is not None:
                db.close()

        # Store metrics in history
        self._update_metrics_history(model_id, current_metrics)

        # Compare with baseline
        baseline_comparison = self._compare_with_baseline(model_id, current_metrics)

        # Generate alerts if necessary
        alerts = self._generate_alerts(model_id, drift_results, baseline_comparison, current_metrics)

        # Update model information
        self.monitored_models[model_id]['last_evaluation'] = evaluation_timestamp
        self.monitored_models[model_id]['drift_history'].append({
            'timestamp': evaluation_timestamp,
            'drift_detected': bool(drift_results.get('drift_detected', False)),
            'metrics': current_metrics,
            'alerts': len(alerts)
        })

        # Prepare comprehensive results
        evaluation_results = {
            'model_id': model_id,
            'timestamp': evaluation_timestamp,
            'drift_detection': drift_results,
            'current_metrics': current_metrics,
            'baseline_comparison': baseline_comparison,
            'alerts': alerts,
            'detector_status': self.concept_drift_detector.get_detector_status(),
            'recommendations': self._generate_recommendations(drift_results, baseline_comparison)
        }

        # Record Prometheus metrics (best-effort)
        try:
            elapsed = time.time() - start_time
            model_label = str(model_id)
            EVALUATIONS_TOTAL.labels(model_id=model_label).inc()
            EVALUATION_LATENCY_SECONDS.labels(model_id=model_label).observe(elapsed)

            if drift_results.get('drift_detected'):
                detectors = drift_results.get('detectors_triggered', [])
                if isinstance(detectors, (list, tuple)):
                    for d in detectors:
                        try:
                            DRIFT_DETECTIONS_TOTAL.labels(model_id=model_label, detector=str(d)).inc()
                        except Exception:
                            pass
                else:
                    try:
                        DRIFT_DETECTIONS_TOTAL.labels(model_id=model_label, detector=str(detectors)).inc()
                    except Exception:
                        pass

            # Update health gauge using generated drift report (small overhead)
            try:
                report = self.generate_drift_report(model_id)
                health_score = float(report.get('model_health_score', 0.0))
                MODEL_HEALTH_SCORE.labels(model_id=model_label).set(health_score)
            except Exception:
                # ignore errors from reporting here
                pass

        except Exception:
            # metrics instrumentation should not break evaluation flow
            self.logger.debug("Prometheus metrics update failed", exc_info=True)

        # Log results
        self._log_evaluation_results(evaluation_results)

        return evaluation_results

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                           confidence: Optional[np.ndarray] = None) -> Dict[str, float]:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }

        if confidence is not None:
            metrics['avg_confidence'] = float(np.mean(confidence))
            metrics['min_confidence'] = float(np.min(confidence))
            metrics['confidence_std'] = float(np.std(confidence))

        return metrics

    def _update_metrics_history(self, model_id: str, metrics: Dict[str, float]):
        for metric_name, value in metrics.items():
            if metric_name in self.metrics_history.get(model_id, {}):
                self.metrics_history[model_id][metric_name].append(float(value))
                max_history = self.config.get('metrics_history_size', 1000)
                if len(self.metrics_history[model_id][metric_name]) > max_history:
                    self.metrics_history[model_id][metric_name] = \
                        self.metrics_history[model_id][metric_name][-max_history:]

    def _compare_with_baseline(self, model_id: str, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        if model_id not in self.model_baselines:
            return {'has_baseline': False}

        baseline = self.model_baselines[model_id]
        comparison = {'has_baseline': True, 'changes': {}}

        for metric, current_value in current_metrics.items():
            if metric in baseline:
                baseline_value = baseline[metric]
                change = current_value - baseline_value
                percent_change = (change / baseline_value) * 100 if baseline_value != 0 else 0

                comparison['changes'][metric] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'absolute_change': change,
                    'percent_change': percent_change,
                    'degraded': change < -self.alert_thresholds.get(f'{metric}_decline', 0.05)
                }

        return comparison

    def _generate_alerts(self, model_id: str, drift_results: Dict,
                         baseline_comparison: Dict, current_metrics: Dict) -> List[DriftAlert]:
        alerts: List[DriftAlert] = []

        # --- Concept drift alerts ---
        if drift_results.get('drift_detected'):
            severity = self._determine_drift_severity(drift_results, baseline_comparison)
            detectors = drift_results.get('detectors_triggered', [])
            detection_method_str = ', '.join(map(str, detectors)) if isinstance(detectors, (list, tuple)) else str(detectors)
            alert_obj = DriftAlert(
                timestamp=datetime.now(),
                drift_type='concept',
                severity=severity,
                detection_method=detection_method_str,
                confidence_score=float(drift_results.get('confidence', 0.9)) if drift_results.get('confidence') is not None else 0.9,
                affected_model=model_id,
                metrics=current_metrics.copy() if isinstance(current_metrics, dict) else {'metrics': current_metrics},
                description=f"Concept drift detected by {detection_method_str}",
                recommended_actions=self._get_drift_recommendations(severity)
            )
            alerts.append(alert_obj)

        # --- Performance degradation alerts ---
        if baseline_comparison.get('has_baseline', False):
            for metric, change_info in baseline_comparison.get('changes', {}).items():
                if change_info.get('degraded', False):
                    perf_alert = DriftAlert(
                        timestamp=datetime.now(),
                        drift_type='performance',
                        severity=DriftSeverity.MEDIUM,
                        detection_method='baseline_comparison',
                        confidence_score=float(abs(change_info.get('percent_change', 0)) / 100),
                        affected_model=model_id,
                        metrics={metric: change_info},
                        description=f"{metric} degraded by {change_info.get('percent_change', 0):.2f}%",
                        recommended_actions=['retrain_model', 'investigate_data_quality']
                    )
                    alerts.append(perf_alert)

        if not alerts:
            return alerts

        # Store alerts in memory
        self.alerts.extend(alerts)

        # Persist alerts to DB
        db = None
        try:
            db = SessionLocal()
            for a in alerts:
                # Convert metrics & recommended_actions to Python-native structures safe for JSON/DB
                metrics_val = self._to_serializable(a.metrics) if a.metrics is not None else None
                rec_actions_val = list(a.recommended_actions) if a.recommended_actions is not None else []

                db_alert = AlertModel(
                    timestamp=a.timestamp,
                    model_id=a.affected_model,
                    drift_type=a.drift_type,
                    severity=a.severity.value if hasattr(a.severity, 'value') else str(a.severity),
                    detection_method=str(a.detection_method),
                    confidence_score=float(a.confidence_score) if a.confidence_score is not None else None,
                    description=a.description,
                    metrics=metrics_val,
                    recommended_actions=rec_actions_val
                )
                db.add(db_alert)

                # Export Prometheus counter for this alert (best-effort)
                try:
                    ALERTS_TOTAL.labels(model_id=a.affected_model,
                                       severity=(a.severity.value if hasattr(a.severity, 'value') else str(a.severity)),
                                       alert_type=a.drift_type).inc()
                except Exception:
                    pass

            db.commit()
        except Exception as exc:
            self.logger.exception("Failed to persist alerts to DB: %s", exc)
            if db is not None:
                try:
                    db.rollback()
                except Exception:
                    pass
        finally:
            if db is not None:
                db.close()

        return alerts

    def _determine_drift_severity(self, drift_results: Dict, baseline_comparison: Dict) -> DriftSeverity:
        n_detectors = len(drift_results.get('detectors_triggered', []))
        performance_decline = 0
        if baseline_comparison.get('has_baseline', False):
            accuracy_change = baseline_comparison['changes'].get('accuracy', {})
            performance_decline = abs(accuracy_change.get('percent_change', 0))

        if n_detectors >= 3 or performance_decline > 20:
            return DriftSeverity.CRITICAL
        elif n_detectors >= 2 or performance_decline > 10:
            return DriftSeverity.HIGH
        elif n_detectors >= 1 or performance_decline > 5:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.LOW

    def _get_drift_recommendations(self, severity: DriftSeverity) -> List[str]:
        recommendations = {
            DriftSeverity.LOW: [
                'monitor_closely',
                'collect_more_data'
            ],
            DriftSeverity.MEDIUM: [
                'investigate_data_changes',
                'consider_model_update',
                'increase_monitoring_frequency'
            ],
            DriftSeverity.HIGH: [
                'retrain_model',
                'investigate_data_sources',
                'implement_online_learning',
                'notify_stakeholders'
            ],
            DriftSeverity.CRITICAL: [
                'immediate_model_retrain',
                'switch_to_backup_model',
                'investigate_data_pipeline',
                'alert_operations_team',
                'consider_model_rollback'
            ]
        }
        return recommendations[severity]

    def _generate_recommendations(self, drift_results: Dict, baseline_comparison: Dict) -> List[str]:
        recommendations = []

        if drift_results.get('drift_detected'):
            recommendations.extend([
                'Model retraining recommended due to concept drift',
                'Investigate changes in data distribution',
                'Consider implementing adaptive learning mechanisms'
            ])

        if drift_results.get('performance_decline', 0) > 0.05:
            recommendations.append('Performance decline detected - review model performance')

        if baseline_comparison.get('has_baseline', False):
            degraded_metrics = [
                metric for metric, info in baseline_comparison['changes'].items()
                if info.get('degraded', False)
            ]
            if degraded_metrics:
                recommendations.append(f'Metrics showing degradation: {", ".join(degraded_metrics)}')

        return recommendations

    def _log_evaluation_results(self, results: Dict[str, Any]):
        model_id = results['model_id']
        drift_detected = bool(results['drift_detection'].get('drift_detected', False))
        n_alerts = len(results.get('alerts', []))
        log_message = f"Model {model_id} evaluation - Drift: {drift_detected}, Alerts: {n_alerts}"
        if drift_detected:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        if model_id not in self.monitored_models:
            raise ValueError(f"Model {model_id} not registered")

        model_info = self.monitored_models[model_id]
        recent_alerts = [alert for alert in self.alerts
                         if alert.affected_model == model_id and
                         (datetime.now() - alert.timestamp).days <= 7]

        return {
            'model_id': model_id,
            'registration_info': model_info['info'],
            'last_evaluation': model_info['last_evaluation'],
            'drift_history': model_info['drift_history'][-10:],  # Last 10 evaluations
            'recent_alerts': [asdict(alert) for alert in recent_alerts],
            'current_detector_status': self.concept_drift_detector.get_detector_status(),
            'metrics_summary': self._get_metrics_summary(model_id)
        }

    def _get_metrics_summary(self, model_id: str) -> Dict[str, Any]:
        if model_id not in self.metrics_history:
            return {}
        summary = {}
        for metric, values in self.metrics_history[model_id].items():
            if values:
                summary[metric] = {
                    'current': values[-1],
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'trend': 'improving' if len(values) > 10 and np.mean(values[-5:]) > np.mean(values[-10:-5]) else 'stable'
                }
        return summary

    def export_alerts(self, format_type: str = 'json',
                      time_range: Optional[Tuple[datetime, datetime]] = None) -> str:
        filtered_alerts = self.alerts
        if time_range:
            start_time, end_time = time_range
            filtered_alerts = [
                alert for alert in filtered_alerts
                if start_time <= alert.timestamp <= end_time
            ]
        if format_type.lower() == 'json':
            return json.dumps([asdict(alert) for alert in filtered_alerts],
                              default=str, indent=2)
        elif format_type.lower() == 'csv':
            if not filtered_alerts:
                return "No alerts to export"
            import csv
            import io
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=asdict(filtered_alerts[0]).keys())
            writer.writeheader()
            for alert in filtered_alerts:
                row = asdict(alert)
                row['recommended_actions'] = '; '.join(row['recommended_actions'])
                row['metrics'] = json.dumps(row['metrics'])
                writer.writerow(row)
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def reset_model_monitoring(self, model_id: str):
        if model_id not in self.monitored_models:
            raise ValueError(f"Model {model_id} not registered")
        self.concept_drift_detector.reset_detectors()
        self.monitored_models[model_id]['drift_history'] = []
        self.monitored_models[model_id]['last_evaluation'] = None
        for metric in self.metrics_history[model_id]:
            self.metrics_history[model_id][metric] = []
        self.alerts = [alert for alert in self.alerts if alert.affected_model != model_id]
        self.logger.info(f"Reset monitoring state for model {model_id}")

    def generate_drift_report(self, model_id: str,
                              time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        if model_id not in self.monitored_models:
            raise ValueError(f"Model {model_id} not registered")
        model_info = self.monitored_models[model_id]
        drift_history = model_info['drift_history']
        if time_range:
            start_time, end_time = time_range
            drift_history = [
                entry for entry in drift_history
                if start_time <= entry['timestamp'] <= end_time
            ]
        total_evaluations = len(drift_history)
        drift_detections = sum(1 for entry in drift_history if entry.get('drift_detected'))
        drift_rate = drift_detections / total_evaluations if total_evaluations > 0 else 0
        model_alerts = [alert for alert in self.alerts if alert.affected_model == model_id]
        if time_range:
            start_time, end_time = time_range
            model_alerts = [
                alert for alert in model_alerts
                if start_time <= alert.timestamp <= end_time
            ]
        alert_severity_counts = {}
        for alert in model_alerts:
            severity = alert.severity.value
            alert_severity_counts[severity] = alert_severity_counts.get(severity, 0) + 1
        performance_trend = self._analyze_performance_trend(model_id, time_range)
        report_recommendations = []
        if drift_rate > 0.3:
            report_recommendations.append("High drift rate detected - consider implementing continuous learning")
        if performance_trend.get('declining_metrics'):
            report_recommendations.append(f"Declining performance in: {', '.join(performance_trend['declining_metrics'])}")
        report = {
            'model_id': model_id,
            'report_generated_at': datetime.now(),
            'analysis_period': {
                'start': time_range[0] if time_range else model_info['registered_at'],
                'end': time_range[1] if time_range else datetime.now(),
                'total_evaluations': total_evaluations
            },
            'drift_analysis': {
                'drift_detections': drift_detections,
                'drift_rate': drift_rate,
                'most_triggered_detector': self._get_most_triggered_detector(model_id),
                'drift_frequency': self._calculate_drift_frequency(drift_history)
            },
            'alert_summary': {
                'total_alerts': len(model_alerts),
                'severity_breakdown': alert_severity_counts,
                'most_common_alert_type': self._get_most_common_alert_type(model_alerts)
            },
            'performance_analysis': performance_trend,
            'recommendations': report_recommendations,
            'model_health_score': self._calculate_model_health_score(model_id, drift_rate, performance_trend)
        }
        return report

    def _analyze_performance_trend(self, model_id: str,
                                    time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        metrics_history = self.metrics_history[model_id]
        trend_analysis = {}
        for metric, values in metrics_history.items():
            if len(values) < 10:
                continue
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            trend_analysis[metric] = {
                'slope': float(slope),
                'trend': 'improving' if slope > 0.001 else 'declining' if slope < -0.001 else 'stable',
                'recent_average': float(np.mean(values[-10:])),
                'historical_average': float(np.mean(values[:-10])) if len(values) > 10 else float(np.mean(values))
            }
        declining_metrics = [metric for metric, analysis in trend_analysis.items() if analysis['trend'] == 'declining']
        return {
            'metrics_trends': trend_analysis,
            'declining_metrics': declining_metrics,
            'overall_trend': 'declining' if len(declining_metrics) > len(trend_analysis) / 2 else 'stable'
        }

    def _get_most_triggered_detector(self, model_id: str) -> str:
        detector_status = self.concept_drift_detector.get_detector_status()
        max_detections = 0
        most_triggered = "None"
        for detector_name, status in detector_status.items():
            if status.get('n_detections', 0) > max_detections:
                max_detections = status.get('n_detections', 0)
                most_triggered = detector_name
        return most_triggered

    def _calculate_drift_frequency(self, drift_history: List[Dict]) -> Dict[str, Any]:
        if len(drift_history) < 2:
            return {'insufficient_data': True}
        drift_events = [entry for entry in drift_history if entry.get('drift_detected')]
        if len(drift_events) < 2:
            return {'frequency': 'rare', 'average_interval_days': None}
        intervals = []
        for i in range(1, len(drift_events)):
            interval = (drift_events[i]['timestamp'] - drift_events[i-1]['timestamp']).days
            intervals.append(interval)
        avg_interval = float(np.mean(intervals)) if intervals else None
        frequency_label = 'frequent' if avg_interval is not None and avg_interval < 7 else 'moderate' if avg_interval is not None and avg_interval < 30 else 'rare'
        return {
            'frequency': frequency_label,
            'average_interval_days': avg_interval,
            'total_drift_events': len(drift_events)
        }

    def _get_most_common_alert_type(self, alerts: List[DriftAlert]) -> str:
        if not alerts:
            return "None"
        type_counts = {}
        for alert in alerts:
            alert_type = alert.drift_type
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
        return max(type_counts, key=type_counts.get)

    def _calculate_model_health_score(self, model_id: str, drift_rate: float,
                                    performance_trend: Dict[str, Any]) -> float:
        base_score = 100.0
        drift_penalty = drift_rate * 30
        declining_metrics = len(performance_trend.get('declining_metrics', []))
        total_metrics = len(performance_trend.get('metrics_trends', {}))
        performance_penalty = (declining_metrics / max(total_metrics, 1)) * 25
        recent_alerts = [
            alert for alert in self.alerts
            if alert.affected_model == model_id and
            (datetime.now() - alert.timestamp).days <= 7
        ]
        alert_penalty = min(len(recent_alerts) * 5, 20)
        health_score = max(base_score - drift_penalty - performance_penalty - alert_penalty, 0)
        return round(health_score, 1)


# Configuration example for ETSI Watchdog integration
ETSI_DRIFT_CONFIG = {
    'detection_methods': ['ddm', 'eddm', 'adwin'],
    'window_size': 1000,
    'metrics_history_size': 5000,
    'alert_thresholds': {
        'performance_decline': 0.05,
        'error_rate_increase': 0.1,
        'confidence_drop': 0.15
    },
    'monitoring_frequency': 'real_time',  # or 'batch', 'scheduled'
    'alert_destinations': ['email', 'slack', 'webhook'],
    'retention_days': 90
}

# Example usage
if __name__ == "__main__":
    # Initialize the monitor
    drift_monitor = ETSIConceptDriftMonitor(ETSI_DRIFT_CONFIG)
    
    # Register a model
    model_info = {
        'name': 'fraud_detection_v1',
        'type': 'classification',
        'framework': 'sklearn',
        'features': ['transaction_amount', 'merchant_category', 'time_of_day'],
        'target': 'is_fraud',
        'training_date': '2024-01-15',
        'version': '1.0'
    }
    
    drift_monitor.register_model('fraud_model_001', model_info)
    
    # Set baseline performance
    baseline_metrics = {
        'accuracy': 0.92,
        'precision': 0.89,
        'recall': 0.94,
        'f1_score': 0.91
    }
    
    drift_monitor.set_baseline('fraud_model_001', baseline_metrics)
    
    # Simulate evaluation with concept drift
    np.random.seed(42)
    
    # Generate synthetic data
    y_true = np.random.choice([0, 1], 500, p=[0.95, 0.05])  # Imbalanced fraud data
    y_pred = np.where(np.random.random(500) < 0.88, y_true, 1 - y_true)  # Some errors
    confidence = np.random.uniform(0.6, 0.99, 500)
    
    # Evaluate predictions
    results = drift_monitor.evaluate_predictions(
        'fraud_model_001', 
        y_true, 
        y_pred, 
        confidence
    )
    
    print("Evaluation Results:")
    print(f"Drift Detected: {results['drift_detection']['drift_detected']}")
    print(f"Alerts Generated: {len(results['alerts'])}")
    print(f"Current Accuracy: {results['current_metrics']['accuracy']:.3f}")
    
    # Generate a drift report
    report = drift_monitor.generate_drift_report('fraud_model_001')
    print(f"\nModel Health Score: {report['model_health_score']}")
    print(f"Drift Rate: {report['drift_analysis']['drift_rate']:.3f}")