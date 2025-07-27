import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, log_loss
from collections import deque
import warnings
from abc import ABC, abstractmethod

class BaseDriftDetector(ABC):
    """Base class for all drift detection methods"""
    
    def __init__(self, name: str):
        self.name = name
        self.drift_detected = False
        self.warning_detected = False
        self.n_detections = 0
        
    @abstractmethod
    def add_element(self, prediction_correct: bool) -> bool:
        """Add new element and return True if drift detected"""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset detector state"""
        pass

class DDM(BaseDriftDetector):
    """Drift Detection Method - monitors error rate and standard deviation"""
    
    def __init__(self, min_num_instances: int = 30, warning_level: float = 2.0, 
                 out_control_level: float = 3.0):
        super().__init__("DDM")
        self.min_num_instances = min_num_instances
        self.warning_level = warning_level
        self.out_control_level = out_control_level
        
        self.reset()
    
    def reset(self):
        """Reset all statistics"""
        self.n_min = float('inf')
        self.s_min = float('inf')
        self.n = 1
        self.s = 0.0
        self.drift_detected = False
        self.warning_detected = False
    
    def add_element(self, prediction_correct: bool) -> bool:
        """
        Add new prediction result and check for drift
        
        Args:
            prediction_correct: True if prediction was correct, False otherwise
            
        Returns:
            True if concept drift detected
        """
        if prediction_correct:
            self.s += 0
        else:
            self.s += 1
            
        self.n += 1
        
        if self.n < self.min_num_instances:
            return False
            
        # Calculate error rate and standard deviation
        p = self.s / self.n
        std = np.sqrt(p * (1 - p) / self.n)
        
        # Update minimum values
        if p + std < self.n_min + self.s_min:
            self.n_min = p
            self.s_min = std
            
        # Check for warnings and drift
        if p + std > self.n_min + self.s_min + self.warning_level * self.s_min:
            self.warning_detected = True
            
        if p + std > self.n_min + self.s_min + self.out_control_level * self.s_min:
            self.drift_detected = True
            self.n_detections += 1
            return True
            
        return False

class EDDM(BaseDriftDetector):
    """Early Drift Detection Method - more sensitive to gradual changes"""
    
    def __init__(self, alpha: float = 0.95, beta: float = 0.9, min_num_instances: int = 30):
        super().__init__("EDDM")
        self.alpha = alpha
        self.beta = beta
        self.min_num_instances = min_num_instances
        self.reset()
    
    def reset(self):
        """Reset all statistics"""
        self.n = 0
        self.distances = deque(maxlen=self.min_num_instances * 2)
        self.mean_max = 0.0
        self.std_max = 0.0
        self.drift_detected = False
        self.warning_detected = False
        self.last_error_pos = None
    
    def add_element(self, prediction_correct: bool) -> bool:
        """Add new prediction result and check for drift"""
        self.n += 1
        
        if not prediction_correct:
            if self.last_error_pos is not None:
                distance = self.n - self.last_error_pos
                self.distances.append(distance)
            self.last_error_pos = self.n
        
        if len(self.distances) < self.min_num_instances:
            return False
            
        # Calculate statistics
        mean_dist = np.mean(self.distances)
        std_dist = np.std(self.distances)
        
        # Update maximum values
        if mean_dist + 2 * std_dist > self.mean_max + 2 * self.std_max:
            self.mean_max = mean_dist
            self.std_max = std_dist
            
        # Check for drift
        if (mean_dist + 2 * std_dist) / (self.mean_max + 2 * self.std_max) < self.alpha:
            self.warning_detected = True
            
        if (mean_dist + 2 * std_dist) / (self.mean_max + 2 * self.std_max) < self.beta:
            self.drift_detected = True
            self.n_detections += 1
            return True
            
        return False

class ADWIN(BaseDriftDetector):
    """Adaptive Windowing for change detection"""
    
    def __init__(self, delta: float = 0.002):
        super().__init__("ADWIN")
        self.delta = delta
        self.reset()
    
    def reset(self):
        """Reset all statistics"""
        self.window = deque()
        self.total = 0.0
        self.variance = 0.0
        self.width = 0
        self.drift_detected = False
        self.warning_detected = False
    
    def add_element(self, prediction_correct: bool) -> bool:
        """Add new element and check for drift"""
        value = 1.0 if prediction_correct else 0.0
        
        # Add to window
        self.window.append(value)
        self.total += value
        self.width += 1
        
        # Update variance
        if self.width > 1:
            mean = self.total / self.width
            self.variance = sum((x - mean) ** 2 for x in self.window) / (self.width - 1)
        
        # Check for drift by testing different cut points
        drift_detected = False
        for i in range(1, self.width):
            # Split window into two parts
            w0 = i
            w1 = self.width - i
            
            if w0 < 5 or w1 < 5:  # Minimum window size
                continue
                
            sum0 = sum(list(self.window)[:i])
            sum1 = sum(list(self.window)[i:])
            
            mean0 = sum0 / w0
            mean1 = sum1 / w1
            
            # Calculate bound for change detection
            bound = np.sqrt((2 * np.log(2 / self.delta)) * ((1/w0) + (1/w1)) / 2)
            
            if abs(mean0 - mean1) > bound:
                # Remove old data
                for _ in range(i):
                    removed = self.window.popleft()
                    self.total -= removed
                    self.width -= 1
                
                drift_detected = True
                self.drift_detected = True
                self.n_detections += 1
                break
                
        return drift_detected

class ConceptDriftDetector:
    """Main concept drift detection system"""
    
    def __init__(self, detection_methods=None, window_size: int = 1000):
        if detection_methods is None:
            detection_methods = ['ddm', 'eddm', 'adwin']
            
        self.detectors = {}
        self.window_size = window_size
        self.prediction_history = deque(maxlen=window_size)
        self.performance_history = deque(maxlen=window_size)
        
        # Initialize detectors
        for method in detection_methods:
            if method.lower() == 'ddm':
                self.detectors['DDM'] = DDM()
            elif method.lower() == 'eddm':
                self.detectors['EDDM'] = EDDM()
            elif method.lower() == 'adwin':
                self.detectors['ADWIN'] = ADWIN()
    
    def detect_drift(self, y_true, y_pred, model_confidence=None):
        """
        Detect concept drift using multiple methods
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_confidence: Confidence scores (optional)
            
        Returns:
            Dictionary with drift detection results
        """
        results = {
            'drift_detected': False,
            'warning_detected': False,
            'detectors_triggered': [],
            'performance_metrics': {}
        }
        
        # Calculate performance metrics
        accuracy = accuracy_score(y_true, y_pred)
        results['performance_metrics']['accuracy'] = accuracy
        
        # Store in history
        self.performance_history.append(accuracy)
        
        # Check each prediction
        for true_label, pred_label in zip(y_true, y_pred):
            prediction_correct = true_label == pred_label
            self.prediction_history.append(prediction_correct)
            
            # Test each detector
            for name, detector in self.detectors.items():
                drift_detected = detector.add_element(prediction_correct)
                
                if drift_detected:
                    results['drift_detected'] = True
                    results['detectors_triggered'].append(name)
                    
                if detector.warning_detected:
                    results['warning_detected'] = True
        
        # Additional drift indicators
        results.update(self._calculate_drift_indicators())
        
        return results
    
    def _calculate_drift_indicators(self):
        """Calculate additional drift indicators"""
        indicators = {}
        
        if len(self.performance_history) < 2:
            return indicators
            
        # Performance trend
        recent_performance = np.mean(list(self.performance_history)[-10:])
        historical_performance = np.mean(list(self.performance_history)[:-10]) if len(self.performance_history) > 10 else recent_performance
        
        indicators['performance_decline'] = historical_performance - recent_performance
        indicators['performance_trend'] = 'declining' if recent_performance < historical_performance else 'stable'
        
        # Prediction stability
        if len(self.prediction_history) >= 50:
            recent_errors = [not x for x in list(self.prediction_history)[-25:]]
            historical_errors = [not x for x in list(self.prediction_history)[-50:-25]]
            
            indicators['error_rate_change'] = np.mean(recent_errors) - np.mean(historical_errors)
        
        return indicators
    
    def reset_detectors(self):
        """Reset all detectors"""
        for detector in self.detectors.values():
            detector.reset()
        
        self.prediction_history.clear()
        self.performance_history.clear()
    
    def get_detector_status(self):
        """Get status of all detectors"""
        status = {}
        for name, detector in self.detectors.items():
            status[name] = {
                'drift_detected': detector.drift_detected,
                'warning_detected': detector.warning_detected,
                'n_detections': detector.n_detections
            }
        return status

# Example usage and testing
if __name__ == "__main__":
    # Create detector
    drift_detector = ConceptDriftDetector(['ddm', 'eddm', 'adwin'])
    
    # Simulate data with concept drift
    np.random.seed(42)
    
    # Stable period
    n_stable = 500
    y_true_stable = np.random.choice([0, 1], n_stable, p=[0.7, 0.3])
    y_pred_stable = np.where(np.random.random(n_stable) < 0.9, y_true_stable, 1 - y_true_stable)
    
    # Drift period - concept changes
    n_drift = 500
    y_true_drift = np.random.choice([0, 1], n_drift, p=[0.3, 0.7])  # Distribution flipped
    y_pred_drift = np.where(np.random.random(n_drift) < 0.6, y_true_drift, 1 - y_true_drift)  # Lower accuracy
    
    # Test stable period
    print("Testing stable period...")
    stable_results = drift_detector.detect_drift(y_true_stable, y_pred_stable)
    print(f"Stable period results: {stable_results}")
    
    # Test drift period
    print("\nTesting drift period...")
    drift_results = drift_detector.detect_drift(y_true_drift, y_pred_drift)
    print(f"Drift period results: {drift_results}")
    
    # Get detector status
    print(f"\nDetector status: {drift_detector.get_detector_status()}")