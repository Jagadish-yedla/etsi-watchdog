# Test script
from concept_drift.detection.drift_detectors import ConceptDriftDetector
import numpy as np

# Create test data
detector = ConceptDriftDetector(['ddm', 'eddm'])
y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])

results = detector.detect_drift(y_true, y_pred)
print(f"Drift detected: {results['drift_detected']}")