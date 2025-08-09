from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional
import logging

# Import our drift monitoring classes
# from etsi_integration import ETSIConceptDriftMonitor, ETSI_DRIFT_CONFIG
from concept_drift.monitoring.etsi_monitor import ETSIConceptDriftMonitor, ETSI_DRIFT_CONFIG
from concept_drift.monitoring.db import SessionLocal
from concept_drift.monitoring.models import Alert as AlertModel, Evaluation as EvalModel
# ---- helpers to convert numpy / datetime -> native py types for JSON ----
import numpy as _np
from datetime import datetime as _datetime

def to_native(obj):
    """Recursively convert numpy types, datetimes, and other non-jsonables to native python types."""
    # numpy scalar (np.bool_, np.int64, np.float64, etc.)
    if isinstance(obj, _np.generic):
        return obj.item()
    # numpy array -> list
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    # datetime -> isoformat string
    if isinstance(obj, _datetime):
        return obj.isoformat()
    # dataclass -> dict (if accidentally passed)
    try:
        from dataclasses import is_dataclass, asdict
        if is_dataclass(obj):
            return to_native(asdict(obj))
    except Exception:
        pass
    # dict
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    # list/tuple/set
    if isinstance(obj, (list, tuple, set)):
        t = [to_native(v) for v in obj]
        return type(obj)(t) if isinstance(obj, tuple) else t
    # fallback: native python types (str, int, bool, float, None) or other serializables
    return obj
# -----------------------------------------------------------------------

#app = Flask(__name__)
#CORS(app)
import os
from flask import Flask

# compute absolute path to the dashboard folder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DASHBOARD_DIR = os.path.join(BASE_DIR, 'dashboard')

app = Flask(
    __name__,
    template_folder=os.path.join(DASHBOARD_DIR),
    static_folder=os.path.join(DASHBOARD_DIR, 'static')
)
# Global drift monitor instance
drift_monitor = None

def initialize_drift_monitor():
    """Initialize the global drift monitor"""
    global drift_monitor
    config = {
        'detection_methods': ['ddm', 'eddm', 'adwin'],
        'window_size': 1000,
        'metrics_history_size': 5000,
        'alert_thresholds': {
            'performance_decline': 0.05,
            'error_rate_increase': 0.1,
            'confidence_drop': 0.15
        }
    }
    drift_monitor = ETSIConceptDriftMonitor(config)

# Initialize the global drift_monitor at import time
initialize_drift_monitor()

# API Endpoints

@app.route('/api/models', methods=['POST'])
def register_model():
    """Register a new model for drift monitoring"""
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        model_info = data.get('model_info', {})
        
        if not model_id:
            return jsonify({'error': 'model_id is required'}), 400
        
        drift_monitor.register_model(model_id, model_info)
        
        return jsonify({
            'status': 'success',
            'message': f'Model {model_id} registered successfully',
            'model_id': model_id
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<model_id>/baseline', methods=['POST'])
def set_model_baseline(model_id):
    """Set baseline performance metrics for a model"""
    try:
        data = request.get_json()
        baseline_metrics = data.get('baseline_metrics', {})
        
        if not baseline_metrics:
            return jsonify({'error': 'baseline_metrics is required'}), 400
        
        drift_monitor.set_baseline(model_id, baseline_metrics)
        
        return jsonify({
            'status': 'success',
            'message': f'Baseline set for model {model_id}',
            'baseline_metrics': baseline_metrics
        }), 200
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<model_id>/evaluate', methods=['POST'])
def evaluate_model_predictions(model_id):
    """Evaluate model predictions and detect drift"""
    try:
        data = request.get_json()
        
        # Extract prediction data
        y_true = np.array(data.get('y_true', []))
        y_pred = np.array(data.get('y_pred', []))
        confidence = data.get('confidence')
        metadata = data.get('metadata', {})
        
        if len(y_true) == 0 or len(y_pred) == 0:
            return jsonify({'error': 'y_true and y_pred are required'}), 400
        
        if len(y_true) != len(y_pred):
            return jsonify({'error': 'y_true and y_pred must have same length'}), 400
        
        confidence_array = np.array(confidence) if confidence else None
        
        # Evaluate predictions
        results = drift_monitor.evaluate_predictions(
            model_id, y_true, y_pred, confidence_array, metadata=metadata
        )
        
        # Convert datetime objects to strings for JSON serialization
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        # Process results for JSON response
        # Build response_data (use to_native to ensure JSON serializability)
        response_data = {
            'model_id': results['model_id'],
            'timestamp': to_native(results['timestamp']),
            'drift_detected': to_native(results['drift_detection'].get('drift_detected', False)),
            'warning_detected': to_native(results['drift_detection'].get('warning_detected', False)),
            'detectors_triggered': to_native(results['drift_detection'].get('detectors_triggered', [])),
            'current_metrics': to_native(results.get('current_metrics', {})),
            'baseline_comparison': to_native(results.get('baseline_comparison', {})),
            'alerts_count': to_native(len(results.get('alerts', []))),
            'recommendations': to_native(results.get('recommendations', [])),
            'detector_status': to_native(results.get('detector_status', {}))
        }

        # Return converted response
        return jsonify({
            'status': 'success',
            'evaluation_results': response_data
        }), 200

        
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<model_id>/status', methods=['GET'])
def get_model_status(model_id):
    """Get comprehensive status for a model"""
    try:
        status = drift_monitor.get_model_status(model_id)
        
        # Serialize datetime objects
        def serialize_dates(obj):
            if isinstance(obj, dict):
                return {k: serialize_dates(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_dates(item) for item in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        serialized_status = serialize_dates(status)
        
        return jsonify({
            'status': 'success',
            'model_status': serialized_status
        }), 200
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<model_id>/report', methods=['GET'])
def generate_model_report(model_id):
    """Generate comprehensive drift report for a model"""
    try:
        # Parse optional time range parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        time_range = None
        if start_date and end_date:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            time_range = (start_dt, end_dt)
        
        report = drift_monitor.generate_drift_report(model_id, time_range)
        
        # Serialize datetime objects
        def serialize_dates(obj):
            if isinstance(obj, dict):
                return {k: serialize_dates(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_dates(item) for item in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        serialized_report = serialize_dates(report)
        
        return jsonify({
            'status': 'success',
            'drift_report': serialized_report
        }), 200
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get alerts with optional filtering"""
    try:
        # Parse query parameters
        model_id = request.args.get('model_id')
        severity = request.args.get('severity')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = int(request.args.get('limit', 100))
        
        # Get all alerts
        all_alerts = drift_monitor.alerts
        
        # Apply filters
        filtered_alerts = all_alerts
        
        if model_id:
            filtered_alerts = [a for a in filtered_alerts if a.affected_model == model_id]
        
        if severity:
            filtered_alerts = [a for a in filtered_alerts if a.severity.value == severity]
        
        if start_date and end_date:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            filtered_alerts = [a for a in filtered_alerts if start_dt <= a.timestamp <= end_dt]
        
        # Limit results
        filtered_alerts = filtered_alerts[-limit:]
        
        # Serialize alerts
        serialized_alerts = []
        for alert in filtered_alerts:
            alert_dict = {
                'timestamp': alert.timestamp.isoformat(),
                'drift_type': alert.drift_type,
                'severity': alert.severity.value,
                'detection_method': alert.detection_method,
                'confidence_score': alert.confidence_score,
                'affected_model': alert.affected_model,
                'metrics': alert.metrics,
                'description': alert.description,
                'recommended_actions': alert.recommended_actions
            }
            serialized_alerts.append(alert_dict)
        
        return jsonify({
            'status': 'success',
            'alerts': serialized_alerts,
            'total_count': len(filtered_alerts)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def list_models():
    """List all registered models"""
    try:
        models_info = []
        for model_id, model_data in drift_monitor.monitored_models.items():
            model_info = {
                'model_id': model_id,
                'info': model_data['info'],
                'registered_at': model_data['registered_at'].isoformat(),
                'last_evaluation': model_data['last_evaluation'].isoformat() if model_data['last_evaluation'] else None,
                'total_evaluations': len(model_data['drift_history']),
                'drift_detections': sum(1 for h in model_data['drift_history'] if h['drift_detected'])
            }
            models_info.append(model_info)
        
        return jsonify({
            'status': 'success',
            'models': models_info,
            'total_count': len(models_info)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<model_id>/reset', methods=['POST'])
def reset_model_monitoring(model_id):
    """Reset monitoring state for a specific model"""
    try:
        drift_monitor.reset_model_monitoring(model_id)
        
        return jsonify({
            'status': 'success',
            'message': f'Monitoring reset for model {model_id}'
        }), 200
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Dashboard Routes

@app.route('/')
def dashboard():
    """Main monitoring dashboard"""
    return render_template('dashboard.html')

@app.route('/dashboard/model/<model_id>')
def model_dashboard(model_id):
    """Model-specific dashboard"""
    return render_template('model_dashboard.html', model_id=model_id)

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'etsi-concept-drift-monitor',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }), 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize the drift monitor
    initialize_drift_monitor()
    
    # Create templates directory and files
    import os
    os.makedirs('templates', exist_ok=True)
    
    with open('templates/dashboard.html', 'w') as f:
        f.write(dashboard_html)
    
    with open('templates/model_dashboard.html', 'w') as f:
        f.write(model_dashboard_html)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)