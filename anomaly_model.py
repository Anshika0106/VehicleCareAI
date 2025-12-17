"""
Anomaly Detection Model
Uses LSTM for time series prediction and XGBoost for classification.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from vehicle_sim import VehicleSimulator


class AnomalyDetector:
    """Hybrid LSTM + XGBoost anomaly detector for vehicle telemetry."""
    
    def __init__(self, sequence_length: int = 10, contamination: float = 0.01):
        """
        Initialize the anomaly detector.
        
        Args:
            sequence_length: Number of previous readings to use for LSTM prediction
            contamination: Expected proportion of anomalies (0.0 to 0.5)
        """
        self.sequence_length = sequence_length
        self.contamination = contamination
        
        # LSTM model for time series prediction
        self.lstm_model: Optional[Sequential] = None
        
        # XGBoost model for classification
        self.xgb_model: Optional[XGBClassifier] = None
        
        # Scaler for normalizing features
        self.scaler = MinMaxScaler()
        
        # History buffer for maintaining sequences
        self.reading_history: List[Dict] = []
        
        self.is_trained = False
        self.feature_names = [
            "engine_rpm",
            "engine_temp_c",
            "vibration_level_g",
            "throttle_pos_pct",
            "battery_voltage_v"
        ]
    
    def sync_history(self, readings: List[Dict]):
        """
        Sync the detector's reading history with external readings.
        This ensures the LSTM has access to recent readings for predictions.
        
        Args:
            readings: List of reading dictionaries to sync from
        """
        # Update history with the most recent readings
        # Keep at least sequence_length readings if available
        if readings:
            self.reading_history = readings[-50:]  # Keep last 50 readings
    
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
        ])
        
        return features
    
    def _create_sequences(self, data: np.ndarray, seq_length: int) -> tuple:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Array of shape (n_samples, n_features)
            seq_length: Length of input sequences
            
        Returns:
            X: Input sequences of shape (n_sequences, seq_length, n_features)
            y: Target values of shape (n_sequences, n_features)
        """
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        
        return np.array(X), np.array(y)
    
    def _build_lstm_model(self, input_shape: tuple) -> Sequential:
        """
        Build LSTM model for time series prediction.
        
        Args:
            input_shape: Shape of input sequences (seq_length, n_features)
            
        Returns:
            Compiled LSTM model
        """
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(5, activation='linear')  # 5 sensor features
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _extract_classification_features(self, reading: dict, 
                                        lstm_prediction: np.ndarray,
                                        lstm_error: np.ndarray) -> np.ndarray:
        """
        Extract features for XGBoost classification.
        Includes raw sensor values, LSTM prediction errors, and statistical features.
        
        Args:
            reading: Current sensor reading
            lstm_prediction: LSTM predicted values
            lstm_error: Prediction errors (actual - predicted)
            
        Returns:
            Feature vector for classification
        """
        sensors = reading["sensors"]
        raw_features = np.array([
            sensors["engine_rpm"],
            sensors["engine_temp_c"],
            sensors["vibration_level_g"],
            sensors["throttle_pos_pct"],
            sensors["battery_voltage_v"]
        ])
        
        # Threshold-based anomaly indicators (critical for detecting faults)
        threshold_features = np.array([
            1.0 if sensors["engine_temp_c"] > 105 else 0.0,  # Warning threshold
            1.0 if sensors["engine_temp_c"] > 120 else 0.0,  # Critical threshold
            1.0 if sensors["vibration_level_g"] > 0.4 else 0.0,  # Warning threshold
            1.0 if sensors["vibration_level_g"] > 1.0 else 0.0,  # Critical threshold
            1.0 if sensors["battery_voltage_v"] < 13.5 else 0.0,  # Low voltage
            1.0 if sensors["battery_voltage_v"] > 14.5 else 0.0,  # High voltage
            1.0 if sensors["engine_rpm"] > 3000 else 0.0,  # High RPM
            1.0 if sensors["engine_rpm"] < 800 else 0.0,  # Low RPM
            sensors["engine_temp_c"] - 105 if sensors["engine_temp_c"] > 105 else 0.0,  # Excess temp
            sensors["vibration_level_g"] - 0.4 if sensors["vibration_level_g"] > 0.4 else 0.0,  # Excess vibration
        ])
        
        # Combine raw features, LSTM predictions, errors, and threshold features
        features = np.concatenate([
            raw_features,
            lstm_prediction,
            lstm_error,
            np.abs(lstm_error),  # Absolute errors
            np.square(lstm_error),  # Squared errors
            threshold_features  # Threshold violation indicators
        ])
        
        # Add statistical features from recent history
        if len(self.reading_history) >= 5:
            recent_features = np.array([
                self._extract_features(r) for r in self.reading_history[-5:]
            ])
            
            # Statistical features
            mean_features = np.mean(recent_features, axis=0)
            std_features = np.std(recent_features, axis=0)
            max_features = np.max(recent_features, axis=0)
            min_features = np.min(recent_features, axis=0)
            
            # Relative changes
            relative_changes = raw_features - mean_features
            
            features = np.concatenate([
                features,
                mean_features,
                std_features,
                max_features,
                min_features,
                relative_changes
            ])
        else:
            # Pad with zeros if not enough history
            features = np.concatenate([
                features,
                np.zeros(25)  # 5 features * 5 stats
            ])
        
        return features
    
    def train_initial_model(self, n_samples: int = 1000):
        """
        Train both LSTM and XGBoost models on normal vehicle readings.
        
        Args:
            n_samples: Number of normal readings to generate for training
        """
        simulator = VehicleSimulator()
        
        # Ensure no faults are active during training
        simulator.inject_fault(None)
        
        # Generate normal readings (no faults)
        print(f"Generating {n_samples} normal readings for training...")
        training_readings = []
        for _ in range(n_samples):
            reading = simulator.generate_reading()
            training_readings.append(reading)
        
        # Extract features
        training_data = np.array([
            self._extract_features(r) for r in training_readings
        ])
        
        # Normalize features
        training_data_scaled = self.scaler.fit_transform(training_data)
        
        # Train LSTM model
        print("Training LSTM model...")
        X_seq, y_seq = self._create_sequences(training_data_scaled, self.sequence_length)
        
        if len(X_seq) > 0:
            self.lstm_model = self._build_lstm_model((self.sequence_length, 5))
            self.lstm_model.fit(
                X_seq, y_seq,
                epochs=50,
                batch_size=32,
                verbose=0,
                validation_split=0.2
            )
            print("LSTM model trained.")
        else:
            raise ValueError("Not enough data to create sequences. Need at least sequence_length + 1 samples.")
        
        # Generate features for XGBoost training
        print("Generating features for XGBoost training...")
        X_train_xgb = []
        y_train_xgb = []
        
        # Use readings after sequence_length to have LSTM predictions
        for i in range(self.sequence_length, len(training_readings)):
            # Get sequence for LSTM prediction
            sequence = training_data_scaled[i - self.sequence_length:i]
            sequence = sequence.reshape(1, self.sequence_length, 5)
            
            # Get LSTM prediction
            lstm_pred = self.lstm_model.predict(sequence, verbose=0)[0]
            
            # Get actual values
            actual = training_data_scaled[i]
            
            # Calculate errors
            lstm_error = actual - lstm_pred
            
            # Extract classification features
            # Need to temporarily store history
            temp_history = self.reading_history.copy()
            self.reading_history = training_readings[max(0, i-5):i]
            
            features = self._extract_classification_features(
                training_readings[i],
                lstm_pred,
                lstm_error
            )
            
            X_train_xgb.append(features)
            y_train_xgb.append(0)  # 0 = normal
            
            # Restore history
            self.reading_history = temp_history
        
        # Generate anomaly samples for training XGBoost
        # Use a fixed number per fault type to ensure good coverage
        print("Generating anomaly samples for XGBoost training...")
        anomaly_types = ["overheat", "vibration", "battery_failure", 
                        "throttle_malfunction", "engine_misfire", 
                        "fuel_system", "cooling_system"]
        n_anomalies_per_type = max(50, int(n_samples * 0.1))  # At least 50 samples per type, or 10% of n_samples
        
        for fault_type in anomaly_types:
            simulator.inject_fault(fault_type)
            for _ in range(n_anomalies_per_type):
                reading = simulator.generate_reading()
                features_raw = self._extract_features(reading)
                features_scaled = self.scaler.transform(features_raw.reshape(1, -1))[0]
                
                # Use recent normal history for sequence if available
                if len(training_data_scaled) >= self.sequence_length:
                    # Use last normal sequence
                    sequence = training_data_scaled[-self.sequence_length:].reshape(1, self.sequence_length, 5)
                    lstm_pred = self.lstm_model.predict(sequence, verbose=0)[0]
                    lstm_error = features_scaled - lstm_pred
                    
                    # Use recent normal readings for history
                    self.reading_history = training_readings[-5:]
                    features = self._extract_classification_features(reading, lstm_pred, lstm_error)
                    X_train_xgb.append(features)
                    y_train_xgb.append(1)  # 1 = anomaly
        
        # Reset to normal
        simulator.inject_fault(None)
        
        # Train XGBoost model
        print("Training XGBoost model...")
        X_train_xgb = np.array(X_train_xgb)
        y_train_xgb = np.array(y_train_xgb)
        
        self.xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=len(y_train_xgb[y_train_xgb == 0]) / len(y_train_xgb[y_train_xgb == 1]) if len(y_train_xgb[y_train_xgb == 1]) > 0 else 1.0
        )
        
        self.xgb_model.fit(X_train_xgb, y_train_xgb)
        print("XGBoost model trained.")
        
        self.is_trained = True
        print(f"Models trained successfully on {n_samples} normal readings.")
    
    def detect_anomaly(self, reading: dict) -> int:
        """
        Detect if a reading is anomalous using LSTM + XGBoost.
        
        Args:
            reading: Dictionary containing sensor data
            
        Returns:
            -1 if anomaly detected, 1 if normal
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before detection. Call train_initial_model() first.")
        
        # Extract features
        features_raw = self._extract_features(reading)
        features_scaled = self.scaler.transform(features_raw.reshape(1, -1))[0]
        
        # Get LSTM prediction
        if len(self.reading_history) >= self.sequence_length:
            # Use recent history for sequence
            recent_features = np.array([
                self._extract_features(r) for r in self.reading_history[-self.sequence_length:]
            ])
            recent_features_scaled = self.scaler.transform(recent_features)
            sequence = recent_features_scaled.reshape(1, self.sequence_length, 5)
            
            lstm_pred = self.lstm_model.predict(sequence, verbose=0)[0]
        else:
            # Not enough history, use current reading as prediction (fallback)
            lstm_pred = features_scaled
        
        # Calculate prediction error
        lstm_error = features_scaled - lstm_pred
        
        # Extract classification features
        features = self._extract_classification_features(reading, lstm_pred, lstm_error)
        features = features.reshape(1, -1)
        
        # XGBoost prediction
        prediction = self.xgb_model.predict(features)[0]
        
        # Update history (will be synced back to session state)
        # Only add if not already the last reading (avoid duplicates)
        if not self.reading_history or self.reading_history[-1] != reading:
            self.reading_history.append(reading)
            # Keep only recent history (last 50 readings)
            if len(self.reading_history) > 50:
                self.reading_history = self.reading_history[-50:]
        
        # Return -1 for anomaly (class 1), 1 for normal (class 0)
        return -1 if prediction == 1 else 1
    
    def get_anomaly_score(self, reading: dict) -> float:
        """
        Get the anomaly score for a reading (higher = more anomalous).
        
        Args:
            reading: Dictionary containing sensor data
            
        Returns:
            Anomaly score (probability of being an anomaly)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring. Call train_initial_model() first.")
        
        # Extract features
        features_raw = self._extract_features(reading)
        features_scaled = self.scaler.transform(features_raw.reshape(1, -1))[0]
        
        # Get LSTM prediction
        if len(self.reading_history) >= self.sequence_length:
            recent_features = np.array([
                self._extract_features(r) for r in self.reading_history[-self.sequence_length:]
            ])
            recent_features_scaled = self.scaler.transform(recent_features)
            sequence = recent_features_scaled.reshape(1, self.sequence_length, 5)
            
            lstm_pred = self.lstm_model.predict(sequence, verbose=0)[0]
        else:
            lstm_pred = features_scaled
        
        # Calculate prediction error
        lstm_error = features_scaled - lstm_pred
        
        # Extract classification features
        features = self._extract_classification_features(reading, lstm_pred, lstm_error)
        features = features.reshape(1, -1)
        
        # XGBoost probability prediction
        anomaly_prob = self.xgb_model.predict_proba(features)[0][1]
        
        # Update history only if not already added (detect_anomaly may have added it)
        if not self.reading_history or self.reading_history[-1] != reading:
            self.reading_history.append(reading)
            if len(self.reading_history) > 50:
                self.reading_history = self.reading_history[-50:]
        
        # Return probability as score (convert to negative for consistency with old interface)
        # Higher probability = more anomalous, so we return negative of probability
        # This way, lower scores still indicate anomalies
        return -anomaly_prob
