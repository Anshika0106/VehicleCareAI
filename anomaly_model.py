"""
Anomaly Detection Model
Uses Isolation Forest for unsupervised anomaly detection on vehicle telemetry.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from vehicle_sim import VehicleSimulator


class AnomalyDetector:
    """Isolation Forest-based anomaly detector for vehicle telemetry."""
    
    def __init__(self, contamination: float = 0.01):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies (0.0 to 0.5)
            Lower values reduce false positives but may miss subtle anomalies
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.anomaly_threshold = -0.3  # Only flag as anomaly if score is below this threshold
        self.is_trained = False
        self.feature_names = [
            "engine_rpm",
            "engine_temp_c",
            "vibration_level_g",
            "throttle_pos_pct",
            "battery_voltage_v"
        ]
    
    def _extract_features(self, reading: dict) -> np.ndarray:
        """
        Extract feature vector from sensor reading.
        
        Args:
            reading: Dictionary containing sensor data
            
        Returns:
            NumPy array of feature values
        """
        sensors = reading["sensors"]
        features = np.array([
            sensors["engine_rpm"],
            sensors["engine_temp_c"],
            sensors["vibration_level_g"],
            sensors["throttle_pos_pct"],
            sensors["battery_voltage_v"]
        ]).reshape(1, -1)
        
        return features
    
    def train_initial_model(self, n_samples: int = 1000):
        """
        Train the Isolation Forest model on normal vehicle readings.
        Uses more samples to better capture normal variation patterns.
        
        Args:
            n_samples: Number of normal readings to generate for training
        """
        simulator = VehicleSimulator()
        
        # Ensure no faults are active during training
        simulator.inject_fault(None)
        
        # Generate normal readings (no faults) with more variation
        training_data = []
        for _ in range(n_samples):
            reading = simulator.generate_reading()
            features = self._extract_features(reading)
            training_data.append(features[0])
        
        # Convert to numpy array
        X_train = np.array(training_data)
        
        # Train the model
        self.model.fit(X_train)
        self.is_trained = True
        
        print(f"Model trained on {n_samples} normal readings.")
    
    def detect_anomaly(self, reading: dict) -> int:
        """
        Detect if a reading is anomalous.
        Uses both the model prediction and score threshold to reduce false positives.
        
        Args:
            reading: Dictionary containing sensor data
            
        Returns:
            -1 if anomaly detected, 1 if normal
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before detection. Call train_initial_model() first.")
        
        features = self._extract_features(reading)
        prediction = self.model.predict(features)
        score = self.model.score_samples(features)[0]
        
        # Only flag as anomaly if both prediction says anomaly AND score is below threshold
        # This reduces false positives from normal variation
        if prediction[0] == -1 and score < self.anomaly_threshold:
            return -1
        else:
            return 1
    
    def get_anomaly_score(self, reading: dict) -> float:
        """
        Get the anomaly score for a reading (lower = more anomalous).
        
        Args:
            reading: Dictionary containing sensor data
            
        Returns:
            Anomaly score (negative values indicate anomalies)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring. Call train_initial_model() first.")
        
        features = self._extract_features(reading)
        score = self.model.score_samples(features)
        
        return float(score[0])

