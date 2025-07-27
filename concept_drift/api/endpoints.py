from flask import Flask,Blueprint, request, jsonify, render_template
from flask_cors import CORS
import json
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional
import logging

# Import our drift monitoring classes
from etsi_integration import ETSIConceptDriftMonitor

drift_bp = Blueprint('drift', __name__)
monitor = ETSIConceptDriftMonitor()

app = Flask(__name__)
CORS(app)

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
        response_data = {
            'model_id': results['model_id'],
            'timestamp': results['timestamp'].isoformat(),
            'drift_detected': results['drift_detection']['drift_detected'],
            'warning_detected': results['drift_detection']['warning_detected'],
            'detectors_triggered': results['drift_detection']['detectors_triggered'],
            'current_metrics': results['current_metrics'],
            'baseline_comparison': results['baseline_comparison'],
            'alerts_count': len(results['alerts']),
            'recommendations': results['recommendations'],
            'detector_status': results['detector_status']
        }
        
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

# Dashboard HTML Templates (Basic versions)

dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETSI Concept Drift Monitor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .metric { text-align: center; padding: 15px; }
        .metric-value { font-size: 2em; font-weight: bold; color: #333; }
        .metric-label { color: #666; margin-top: 5px; }
        .alert-high { border-left: 4px solid #ff4444; }
        .alert-medium { border-left: 4px solid #ffaa00; }
        .alert-low { border-left: 4px solid #44ff44; }
        .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        .status-healthy { background-color: #44ff44; }
        .status-warning { background-color: #ffaa00; }
        .status-critical { background-color: #ff4444; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; }
        .btn { padding: 8px 16px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .btn:hover { background-color: #0056b3; }
        .refresh-btn { float: right; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 ETSI Concept Drift Monitor</h1>
        <p>Real-time monitoring of machine learning model drift and performance</p>
        <button class="btn refresh-btn" onclick="refreshDashboard()">Refresh</button>
    </div>
    
    <div class="metrics">
        <div class="card metric">
            <div class="metric-value" id="total-models">-</div>
            <div class="metric-label">Total Models</div>
        </div>
        <div class="card metric">
            <div class="metric-value" id="active-alerts">-</div>
            <div class="metric-label">Active Alerts</div>
        </div>
        <div class="card metric">
            <div class="metric-value" id="drift-detections">-</div>
            <div class="metric-label">Drift Detections (24h)</div>
        </div>
        <div class="card metric">
            <div class="metric-value" id="avg-health-score">-</div>
            <div class="metric-label">Avg Health Score</div>
        </div>
    </div>
    
    <div class="card">
        <h2>📊 Registered Models</h2>
        <table id="models-table">
            <thead>
                <tr>
                    <th>Status</th>
                    <th>Model ID</th>
                    <th>Last Evaluation</th>
                    <th>Total Evaluations</th>
                    <th>Drift Detections</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody id="models-tbody">
                <!-- Models will be populated here -->
            </tbody>
        </table>
    </div>
    
    <div class="card">
        <h2>🚨 Recent Alerts</h2>
        <div id="alerts-container">
            <!-- Alerts will be populated here -->
        </div>
    </div>

    <script>
        async function loadDashboardData() {
            try {
                // Load models
                const modelsResponse = await fetch('/api/models');
                const modelsData = await modelsResponse.json();
                
                if (modelsData.status === 'success') {
                    updateModelsTable(modelsData.models);
                    document.getElementById('total-models').textContent = modelsData.total_count;
                }
                
                // Load alerts
                const alertsResponse = await fetch('/api/alerts?limit=10');
                const alertsData = await alertsResponse.json();
                
                if (alertsData.status === 'success') {
                    updateAlertsContainer(alertsData.alerts);
                    document.getElementById('active-alerts').textContent = alertsData.total_count;
                }
                
                // Calculate drift detections in last 24h
                const last24h = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
                const recentAlertsResponse = await fetch(`/api/alerts?start_date=${last24h}`);
                const recentAlertsData = await recentAlertsResponse.json();
                
                if (recentAlertsData.status === 'success') {
                    const driftAlerts = recentAlertsData.alerts.filter(a => a.drift_type === 'concept');
                    document.getElementById('drift-detections').textContent = driftAlerts.length;
                }
                
            } catch (error) {
                console.error('Error loading dashboard data:', error);
            }
        }
        
        function updateModelsTable(models) {
            const tbody = document.getElementById('models-tbody');
            tbody.innerHTML = '';
            
            models.forEach(model => {
                const row = document.createElement('tr');
                
                const driftRate = model.total_evaluations > 0 ? 
                    (model.drift_detections / model.total_evaluations * 100).toFixed(1) : '0';
                
                const statusClass = model.drift_detections > model.total_evaluations * 0.3 ? 'status-critical' :
                                  model.drift_detections > 0 ? 'status-warning' : 'status-healthy';
                
                row.innerHTML = `
                    <td><span class="status-indicator ${statusClass}"></span></td>
                    <td><a href="/dashboard/model/${model.model_id}">${model.model_id}</a></td>
                    <td>${model.last_evaluation ? new Date(model.last_evaluation).toLocaleString() : 'Never'}</td>
                    <td>${model.total_evaluations}</td>
                    <td>${model.drift_detections} (${driftRate}%)</td>
                    <td>
                        <button class="btn" onclick="viewModelDetails('${model.model_id}')">View</button>
                        <button class="btn" onclick="resetModel('${model.model_id}')" style="background-color: #dc3545;">Reset</button>
                    </td>
                `;
                
                tbody.appendChild(row);
            });
        }
        
        function updateAlertsContainer(alerts) {
            const container = document.getElementById('alerts-container');
            container.innerHTML = '';
            
            if (alerts.length === 0) {
                container.innerHTML = '<p>No recent alerts</p>';
                return;
            }
            
            alerts.forEach(alert => {
                const alertDiv = document.createElement('div');
                alertDiv.className = `card alert-${alert.severity}`;
                
                alertDiv.innerHTML = `
                    <h4>${alert.drift_type.toUpperCase()} - ${alert.severity.toUpperCase()}</h4>
                    <p><strong>Model:</strong> ${alert.affected_model}</p>
                    <p><strong>Time:</strong> ${new Date(alert.timestamp).toLocaleString()}</p>
                    <p><strong>Description:</strong> ${alert.description}</p>
                    <p><strong>Detection Method:</strong> ${alert.detection_method}</p>
                    <p><strong>Recommended Actions:</strong> ${alert.recommended_actions.join(', ')}</p>
                `;
                
                container.appendChild(alertDiv);
            });
        }
        
        function viewModelDetails(modelId) {
            window.location.href = `/dashboard/model/${modelId}`;
        }
        
        async function resetModel(modelId) {
            if (!confirm(`Are you sure you want to reset monitoring for ${modelId}?`)) {
                return;
            }
            
            try {
                const response = await fetch(`/api/models/${modelId}/reset`, {
                    method: 'POST'
                });
                
                if (response.ok) {
                    alert('Model monitoring reset successfully');
                    refreshDashboard();
                } else {
                    alert('Error resetting model monitoring');
                }
            } catch (error) {
                console.error('Error resetting model:', error);
                alert('Error resetting model monitoring');
            }
        }
        
        function refreshDashboard() {
            loadDashboardData();
        }
        
        // Load data on page load
        document.addEventListener('DOMContentLoaded', loadDashboardData);
        
        // Auto-refresh every 30 seconds
        setInterval(loadDashboardData, 30000);
    </script>
</body>
</html>
"""

model_dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Dashboard - ETSI Drift Monitor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .metric { text-align: center; padding: 15px; }
        .metric-value { font-size: 1.8em; font-weight: bold; color: #333; }
        .metric-label { color: #666; margin-top: 5px; font-size: 0.9em; }
        .health-score { font-size: 3em; }
        .health-excellent { color: #28a745; }
        .health-good { color: #ffc107; }
        .health-poor { color: #dc3545; }
        .btn { padding: 8px 16px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
        .btn:hover { background-color: #0056b3; }
        .btn-danger { background-color: #dc3545; }
        .btn-danger:hover { background-color: #c82333; }
        .chart-container { height: 300px; margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; }
        .detector-status { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; }
        .detector-active { background-color: #dc3545; color: white; }
        .detector-warning { background-color: #ffc107; color: black; }
        .detector-normal { background-color: #28a745; color: white; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="header">
        <h1>📊 Model Dashboard: <span id="model-id-header">Loading...</span></h1>
        <button class="btn" onclick="goBack()">← Back to Dashboard</button>
        <button class="btn" onclick="refreshData()">Refresh</button>
        <button class="btn btn-danger" onclick="resetModelData()">Reset Model</button>
    </div>
    
    <div class="metrics">
        <div class="card metric">
            <div class="metric-value health-score" id="health-score">-</div>
            <div class="metric-label">Health Score</div>
        </div>
        <div class="card metric">
            <div class="metric-value" id="drift-rate">-</div>
            <div class="metric-label">Drift Rate</div>
        </div>
        <div class="card metric">
            <div class="metric-value" id="total-evaluations">-</div>
            <div class="metric-label">Total Evaluations</div>
        </div>
        <div class="card metric">
            <div class="metric-value" id="recent-alerts">-</div>
            <div class="metric-label">Recent Alerts (7d)</div>
        </div>
    </div>
    
    <div class="card">
        <h3>🎯 Current Performance Metrics</h3>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value" id="current-accuracy">-</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="current-precision">-</div>
                <div class="metric-label">Precision</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="current-recall">-</div>
                <div class="metric-label">Recall</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="current-f1">-</div>
                <div class="metric-label">F1 Score</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h3>🔍 Drift Detector Status</h3>
        <table id="detector-status-table">
            <thead>
                <tr>
                    <th>Detector</th>
                    <th>Status</th>
                    <th>Total Detections</th>
                    <th>Last Detection</th>
                </tr>
            </thead>
            <tbody id="detector-tbody">
                <!-- Detector status will be populated here -->
            </tbody>
        </table>
    </div>
    
    <div class="card">
        <h3>📈 Performance Trend</h3>
        <div class="chart-container">
            <canvas id="performance-chart"></canvas>
        </div>
    </div>
    
    <div class="card">
        <h3>🚨 Recent Alerts</h3>
        <div id="model-alerts-container">
            <!-- Model-specific alerts will be populated here -->
        </div>
    </div>

    <script>
        const modelId = window.location.pathname.split('/').pop();
        let performanceChart = null;
        
        async function loadModelData() {
            try {
                document.getElementById('model-id-header').textContent = modelId;
                
                // Load model status
                const statusResponse = await fetch(`/api/models/${modelId}/status`);
                const statusData = await statusResponse.json();
                
                if (statusData.status === 'success') {
                    updateModelMetrics(statusData.model_status);
                }
                
                // Load model report
                const reportResponse = await fetch(`/api/models/${modelId}/report`);
                const reportData = await reportResponse.json();
                
                if (reportData.status === 'success') {
                    updateDriftMetrics(reportData.drift_report);
                }
                
                // Load model alerts
                const alertsResponse = await fetch(`/api/alerts?model_id=${modelId}&limit=10`);
                const alertsData = await alertsResponse.json();
                
                if (alertsData.status === 'success') {
                    updateModelAlerts(alertsData.alerts);
                }
                
            } catch (error) {
                console.error('Error loading model data:', error);
                alert('Error loading model data');
            }
        }
        
        function updateModelMetrics(modelStatus) {
            // Update detector status table
            const tbody = document.getElementById('detector-tbody');
            tbody.innerHTML = '';
            
            if (modelStatus.current_detector_status) {
                Object.entries(modelStatus.current_detector_status).forEach(([detector, status]) => {
                    const row = document.createElement('tr');
                    
                    let statusClass = 'detector-normal';
                    let statusText = 'Normal';
                    
                    if (status.drift_detected) {
                        statusClass = 'detector-active';
                        statusText = 'Drift Detected';
                    } else if (status.warning_detected) {
                        statusClass = 'detector-warning';
                        statusText = 'Warning';
                    }
                    
                    row.innerHTML = `
                        <td>${detector}</td>
                        <td><span class="detector-status ${statusClass}">${statusText}</span></td>
                        <td>${status.n_detections}</td>
                        <td>-</td>
                    `;
                    
                    tbody.appendChild(row);
                });
            }
            
            // Update current performance metrics
            if (modelStatus.metrics_summary) {
                const summary = modelStatus.metrics_summary;
                
                if (summary.accuracy) {
                    document.getElementById('current-accuracy').textContent = summary.accuracy.current.toFixed(3);
                }
                if (summary.precision) {
                    document.getElementById('current-precision').textContent = summary.precision.current.toFixed(3);
                }
                if (summary.recall) {
                    document.getElementById('current-recall').textContent = summary.recall.current.toFixed(3);
                }
                if (summary.f1_score) {
                    document.getElementById('current-f1').textContent = summary.f1_score.current.toFixed(3);
                }
            }
            
            // Update evaluation count
            document.getElementById('total-evaluations').textContent = modelStatus.drift_history ? modelStatus.drift_history.length : 0;
        }
        
        function updateDriftMetrics(driftReport) {
            // Update health score
            const healthScore = driftReport.model_health_score;
            const healthElement = document.getElementById('health-score');
            healthElement.textContent = healthScore.toFixed(1);
            
            // Set health score color
            healthElement.className = 'metric-value health-score ';
            if (healthScore >= 80) {
                healthElement.className += 'health-excellent';
            } else if (healthScore >= 60) {
                healthElement.className += 'health-good';
            } else {
                healthElement.className += 'health-poor';
            }
            
            // Update drift rate
            const driftRate = (driftReport.drift_analysis.drift_rate * 100).toFixed(1);
            document.getElementById('drift-rate').textContent = `${driftRate}%`;
            
            // Update recent alerts count
            const recentAlerts = driftReport.alert_summary.total_alerts;
            document.getElementById('recent-alerts').textContent = recentAlerts;
        }
        
        function updateModelAlerts(alerts) {
            const container = document.getElementById('model-alerts-container');
            container.innerHTML = '';
            
            if (alerts.length === 0) {
                container.innerHTML = '<p>No recent alerts for this model</p>';
                return;
            }
            
            alerts.forEach(alert => {
                const alertDiv = document.createElement('div');
                alertDiv.className = `card alert-${alert.severity}`;
                alertDiv.style.borderLeft = `4px solid ${getSeverityColor(alert.severity)}`;
                
                alertDiv.innerHTML = `
                    <h4>${alert.drift_type.toUpperCase()} - ${alert.severity.toUpperCase()}</h4>
                    <p><strong>Time:</strong> ${new Date(alert.timestamp).toLocaleString()}</p>
                    <p><strong>Description:</strong> ${alert.description}</p>
                    <p><strong>Detection Method:</strong> ${alert.detection_method}</p>
                    <p><strong>Confidence:</strong> ${(alert.confidence_score * 100).toFixed(1)}%</p>
                    <p><strong>Recommended Actions:</strong> ${alert.recommended_actions.join(', ')}</p>
                `;
                
                container.appendChild(alertDiv);
            });
        }
        
        function getSeverityColor(severity) {
            const colors = {
                'low': '#28a745',
                'medium': '#ffc107',
                'high': '#fd7e14',
                'critical': '#dc3545'
            };
            return colors[severity] || '#6c757d';
        }
        
        function goBack() {
            window.location.href = '/';
        }
        
        function refreshData() {
            loadModelData();
        }
        
        async function resetModelData() {
            if (!confirm(`Are you sure you want to reset all monitoring data for ${modelId}?`)) {
                return;
            }
            
            try {
                const response = await fetch(`/api/models/${modelId}/reset`, {
                    method: 'POST'
                });
                
                if (response.ok) {
                    alert('Model monitoring data reset successfully');
                    refreshData();
                } else {
                    alert('Error resetting model data');
                }
            } catch (error) {
                console.error('Error resetting model:', error);
                alert('Error resetting model data');
            }
        }
        
        // Load data on page load
        document.addEventListener('DOMContentLoaded', loadModelData);
        
        // Auto-refresh every 30 seconds
        setInterval(loadModelData, 30000);
    </script>
</body>
</html>
"""

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