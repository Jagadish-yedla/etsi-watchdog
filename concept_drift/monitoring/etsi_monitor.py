import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum

# Import the drift detection classes (assuming they're in a separate module)
from concept_drift_detection import ConceptDriftDetector
import os
import yaml

def load_config():
    path = os.getenv("DRIFT_CONFIG_PATH", "concept_drift/config/drift_monitor_config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)

class DriftSeverity(Enum):
    """Enumeration for drift severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DriftAlert:
    """Data class for drift alerts"""
    timestamp: datetime
    drift_type: str  # 'concept', 'label', 'feature'
    severity: DriftSeverity
    detection_method: str
    confidence_score: float
    affected_model: str
    metrics: Dict[str, Any]
    description: str
    recommended_actions: List[str]

class ETSIConceptDriftMonitor:
    """
    Main class for integrating concept drift detection into ETSI Watchdog
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize drift detectors
        self.concept_drift_detector = ConceptDriftDetector(
            detection_methods=config.get('detection_methods', ['ddm', 'eddm', 'adwin']),
            window_size=config.get('window_size', 1000)
        )
        
        # Alert system
        self.alerts = []
        self.alert_thresholds = config.get('alert_thresholds', {
            'performance_decline': 0.05,
            'error_rate_increase': 0.1,
            'confidence_drop': 0.15
        })
        
        # Model registry
        self.monitored_models = {}
        self.model_baselines = {}
        
        # Metrics tracking
        self.metrics_history = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the drift monitor"""
        logger = logging.getLogger('etsi_concept_drift')
        logger.setLevel(logging.INFO)
        
        # Create handler if not exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def register_model(self, model_id: str, model_info: Dict[str, Any]):
        """
        Register a model for drift monitoring
        
        Args:
            model_id: Unique identifier for the model
            model_info: Dictionary containing model metadata
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
            'confidence_scores': [],
            'prediction_latency': []
        }
        
        self.logger.info(f"Registered model {model_id} for drift monitoring")
    
    def set_baseline(self, model_id: str, baseline_metrics: Dict[str, float]):
        """
        Set performance baseline for a model
        
        Args:
            model_id: Model identifier
            baseline_metrics: Dictionary of baseline performance metrics
        """
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
        """
        Evaluate model predictions and detect concept drift
        
        Args:
            model_id: Model identifier
            y_true: True labels
            y_pred: Predicted labels
            prediction_confidence: Confidence scores for predictions
            feature_data: Input features (for feature drift detection)
            metadata: Additional metadata
            
        Returns:
            Dictionary containing evaluation results and drift detection outcomes
        """
        if model_id not in self.monitored_models:
            raise ValueError(f"Model {model_id} not registered")
        
        evaluation_timestamp = datetime.now()
        
        # Detect concept drift
        drift_results = self.concept_drift_detector.detect_drift(
            y_true, y_pred, prediction_confidence
        )
        
        # Calculate additional metrics
        current_metrics = self._calculate_metrics(y_true, y_pred, prediction_confidence)
        
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
            'drift_detected': drift_results['drift_detected'],
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
        
        # Log results
        self._log_evaluation_results(evaluation_results)
        
        return evaluation_results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          confidence: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate performance metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        if confidence is not None:
            metrics['avg_confidence'] = np.mean(confidence)
            metrics['min_confidence'] = np.min(confidence)
            metrics['confidence_std'] = np.std(confidence)
        
        return metrics
    
    def _update_metrics_history(self, model_id: str, metrics: Dict[str, float]):
        """Update metrics history for a model"""
        for metric_name, value in metrics.items():
            if metric_name in self.metrics_history[model_id]:
                self.metrics_history[model_id][metric_name].append(value)
                
                # Keep only recent history (configurable window)
                max_history = self.config.get('metrics_history_size', 1000)
                if len(self.metrics_history[model_id][metric_name]) > max_history:
                    self.metrics_history[model_id][metric_name] = \
                        self.metrics_history[model_id][metric_name][-max_history:]
    
    def _compare_with_baseline(self, model_id: str, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Compare current metrics with baseline"""
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
        """Generate alerts based on drift detection and performance changes"""
        alerts = []
        
        # Concept drift alerts
        if drift_results['drift_detected']:
            severity = self._determine_drift_severity(drift_results, baseline_comparison)
            
            alert = DriftAlert(
                timestamp=datetime.now(),
                drift_type='concept',
                severity=severity,
                detection_method=', '.join(drift_results['detectors_triggered']),
                confidence_score=0.9,  # Could be calculated based on multiple detectors
                affected_model=model_id,
                metrics=current_metrics,
                description=f"Concept drift detected by {drift_results['detectors_triggered']}",
                recommended_actions=self._get_drift_recommendations(severity)
            )
            alerts.append(alert)
        
        # Performance degradation alerts
        if baseline_comparison.get('has_baseline', False):
            for metric, change_info in baseline_comparison['changes'].items():
                if change_info['degraded']:
                    alert = DriftAlert(
                        timestamp=datetime.now(),
                        drift_type='performance',
                        severity=DriftSeverity.MEDIUM,
                        detection_method='baseline_comparison',
                        confidence_score=abs(change_info['percent_change']) / 100,
                        affected_model=model_id,
                        metrics={metric: change_info},
                        description=f"{metric} degraded by {change_info['percent_change']:.2f}%",
                        recommended_actions=['retrain_model', 'investigate_data_quality']
                    )
                    alerts.append(alert)
        
        # Store alerts
        self.alerts.extend(alerts)
        
        return alerts
    
    def _determine_drift_severity(self, drift_results: Dict, baseline_comparison: Dict) -> DriftSeverity:
        """Determine the severity of detected drift"""
        # Number of detectors triggered
        n_detectors = len(drift_results['detectors_triggered'])
        
        # Performance decline
        performance_decline = 0
        if baseline_comparison.get('has_baseline', False):
            accuracy_change = baseline_comparison['changes'].get('accuracy', {})
            performance_decline = abs(accuracy_change.get('percent_change', 0))
        
        # Determine severity
        if n_detectors >= 3 or performance_decline > 20:
            return DriftSeverity.CRITICAL
        elif n_detectors >= 2 or performance_decline > 10:
            return DriftSeverity.HIGH
        elif n_detectors >= 1 or performance_decline > 5:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.LOW
    
    def _get_drift_recommendations(self, severity: DriftSeverity) -> List[str]:
        """Get recommendations based on drift severity"""
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
        """Generate general recommendations based on evaluation results"""
        recommendations = []
        
        if drift_results['drift_detected']:
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
        """Log evaluation results"""
        model_id = results['model_id']
        drift_detected = results['drift_detection']['drift_detected']
        n_alerts = len(results['alerts'])
        
        log_message = f"Model {model_id} evaluation - Drift: {drift_detected}, Alerts: {n_alerts}"
        
        if drift_detected:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive status for a model"""
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
        """Get summary statistics for model metrics"""
        if model_id not in self.metrics_history:
            return {}
        
        summary = {}
        for metric, values in self.metrics_history[model_id].items():
            if values:
                summary[metric] = {
                    'current': values[-1],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'trend': 'improving' if len(values) > 10 and np.mean(values[-5:]) > np.mean(values[-10:-5]) else 'stable'
                }
        return summary
    
    def export_alerts(self, format_type: str = 'json', 
                     time_range: Optional[Tuple[datetime, datetime]] = None) -> str:
        """
        Export alerts in specified format
        
        Args:
            format_type: Export format ('json', 'csv')
            time_range: Optional time range filter (start, end)
            
        Returns:
            Serialized alerts data
        """
        filtered_alerts = self.alerts
        
        if time_range:
            start_time, end_time = time_range
            filtered_alerts = [
                alert for alert in self.alerts
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
                # Convert complex fields to strings
                row['recommended_actions'] = '; '.join(row['recommended_actions'])
                row['metrics'] = json.dumps(row['metrics'])
                writer.writerow(row)
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def reset_model_monitoring(self, model_id: str):
        """Reset monitoring state for a specific model"""
        if model_id not in self.monitored_models:
            raise ValueError(f"Model {model_id} not registered")
        
        # Reset drift detectors
        self.concept_drift_detector.reset_detectors()
        
        # Clear model history
        self.monitored_models[model_id]['drift_history'] = []
        self.monitored_models[model_id]['last_evaluation'] = None
        
        # Clear metrics history
        for metric in self.metrics_history[model_id]:
            self.metrics_history[model_id][metric] = []
        
        # Remove model-specific alerts
        self.alerts = [alert for alert in self.alerts if alert.affected_model != model_id]
        
        self.logger.info(f"Reset monitoring state for model {model_id}")
    
    def generate_drift_report(self, model_id: str, 
                            time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive drift report for a model
        
        Args:
            model_id: Model identifier
            time_range: Optional time range for the report
            
        Returns:
            Comprehensive drift analysis report
        """
        if model_id not in self.monitored_models:
            raise ValueError(f"Model {model_id} not registered")
        
        model_info = self.monitored_models[model_id]
        
        # Filter drift history by time range
        drift_history = model_info['drift_history']
        if time_range:
            start_time, end_time = time_range
            drift_history = [
                entry for entry in drift_history
                if start_time <= entry['timestamp'] <= end_time
            ]
        
        # Calculate drift statistics
        total_evaluations = len(drift_history)
        drift_detections = sum(1 for entry in drift_history if entry['drift_detected'])
        drift_rate = drift_detections / total_evaluations if total_evaluations > 0 else 0
        
        # Alert statistics
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
        
        # Performance trends
        performance_trend = self._analyze_performance_trend(model_id, time_range)
        
        # Generate recommendations
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
        """Analyze performance trends for a model"""
        metrics_history = self.metrics_history[model_id]
        trend_analysis = {}
        
        for metric, values in metrics_history.items():
            if len(values) < 10:  # Need sufficient data for trend analysis
                continue
            
            # Simple trend analysis using linear regression slope
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            
            trend_analysis[metric] = {
                'slope': slope,
                'trend': 'improving' if slope > 0.001 else 'declining' if slope < -0.001 else 'stable',
                'recent_average': np.mean(values[-10:]),
                'historical_average': np.mean(values[:-10]) if len(values) > 10 else np.mean(values)
            }
        
        # Identify declining metrics
        declining_metrics = [
            metric for metric, analysis in trend_analysis.items()
            if analysis['trend'] == 'declining'
        ]
        
        return {
            'metrics_trends': trend_analysis,
            'declining_metrics': declining_metrics,
            'overall_trend': 'declining' if len(declining_metrics) > len(trend_analysis) / 2 else 'stable'
        }
    
    def _get_most_triggered_detector(self, model_id: str) -> str:
        """Get the detector that has been triggered most often"""
        detector_counts = {}
        model_history = self.monitored_models[model_id]['drift_history']
        
        # This would need to be tracked more granularly in practice
        # For now, return the detector with most total detections
        detector_status = self.concept_drift_detector.get_detector_status()
        max_detections = 0
        most_triggered = "None"
        
        for detector_name, status in detector_status.items():
            if status['n_detections'] > max_detections:
                max_detections = status['n_detections']
                most_triggered = detector_name
        
        return most_triggered
    
    def _calculate_drift_frequency(self, drift_history: List[Dict]) -> Dict[str, Any]:
        """Calculate drift frequency statistics"""
        if len(drift_history) < 2:
            return {'insufficient_data': True}
        
        drift_events = [entry for entry in drift_history if entry['drift_detected']]
        
        if len(drift_events) < 2:
            return {'frequency': 'rare', 'average_interval_days': None}
        
        # Calculate average time between drift events
        intervals = []
        for i in range(1, len(drift_events)):
            interval = (drift_events[i]['timestamp'] - drift_events[i-1]['timestamp']).days
            intervals.append(interval)
        
        avg_interval = np.mean(intervals)
        
        frequency_label = 'frequent' if avg_interval < 7 else 'moderate' if avg_interval < 30 else 'rare'
        
        return {
            'frequency': frequency_label,
            'average_interval_days': avg_interval,
            'total_drift_events': len(drift_events)
        }
    
    def _get_most_common_alert_type(self, alerts: List[DriftAlert]) -> str:
        """Get the most common type of alert"""
        if not alerts:
            return "None"
        
        type_counts = {}
        for alert in alerts:
            alert_type = alert.drift_type
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
        
        return max(type_counts, key=type_counts.get)
    
    def _calculate_model_health_score(self, model_id: str, drift_rate: float, 
                                    performance_trend: Dict[str, Any]) -> float:
        """Calculate an overall model health score (0-100)"""
        base_score = 100.0
        
        # Penalize for high drift rate
        drift_penalty = drift_rate * 30  # Up to 30 points penalty
        
        # Penalize for declining performance
        declining_metrics = len(performance_trend.get('declining_metrics', []))
        total_metrics = len(performance_trend.get('metrics_trends', {}))
        performance_penalty = (declining_metrics / max(total_metrics, 1)) * 25  # Up to 25 points
        
        # Penalize for recent alerts
        recent_alerts = [
            alert for alert in self.alerts
            if alert.affected_model == model_id and 
            (datetime.now() - alert.timestamp).days <= 7
        ]
        alert_penalty = min(len(recent_alerts) * 5, 20)  # Up to 20 points
        
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