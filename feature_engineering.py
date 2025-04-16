import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

class FeatureEngineer:
    """
    Prepares raw metrics for machine learning models
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scalers = {
            'node': StandardScaler(),
            'pod': StandardScaler()
        }
        self.is_fitted = {
            'node': False,
            'pod': False
        }
    
    def prepare_node_features(self, node_df, fit_scaler=False):
        """
        Prepare node metrics for model input
        
        Args:
            node_df: DataFrame with raw node metrics
            fit_scaler: Whether to fit the scaler (for training) or use existing scaler (for prediction)
        
        Returns:
            DataFrame with processed features
        """
        if node_df.empty:
            self.logger.warning("Empty node metrics dataframe provided")
            return pd.DataFrame()
        
        try:
            # Select relevant columns
            feature_cols = [
                'cpu_usage_percent', 'memory_usage_percent', 'disk_usage_percent', 
                'network_load_mbps', 'pod_count', 'api_server_latency_ms', 
                'etcd_latency_ms', 'network_errors', 'network_drops', 'response_time_ms'
            ]
            
            # Ensure all required columns exist, if not, add with default values
            for col in feature_cols:
                if col not in node_df.columns:
                    self.logger.warning(f"Missing column {col} in node metrics, adding with default values")
                    node_df[col] = 0
            
            X = node_df[feature_cols].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Add engineered features
            X['memory_disk_ratio'] = X['memory_usage_percent'] / (X['disk_usage_percent'] + 1)
            X['cpu_memory_ratio'] = X['cpu_usage_percent'] / (X['memory_usage_percent'] + 1)
            X['network_error_rate'] = X['network_errors'] / (X['network_load_mbps'] + 0.1)
            
            # Scale features
            if fit_scaler:
                X_scaled = self.scalers['node'].fit_transform(X)
                self.is_fitted['node'] = True
            elif self.is_fitted['node']:
                X_scaled = self.scalers['node'].transform(X)
            else:
                self.logger.warning("Scaler not fitted yet, will fit now")
                X_scaled = self.scalers['node'].fit_transform(X)
                self.is_fitted['node'] = True
            
            # Convert back to DataFrame
            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            # Add identifier columns back
            for col in ['node_name', 'timestamp', 'node_condition']:
                if col in node_df.columns:
                    X_scaled_df[col] = node_df[col]
            
            return X_scaled_df
        
        except Exception as e:
            self.logger.error(f"Error preparing node features: {e}")
            return pd.DataFrame()
    
    def prepare_pod_features(self, pod_df, fit_scaler=False):
        """
        Prepare pod metrics for model input
        
        Args:
            pod_df: DataFrame with raw pod metrics
            fit_scaler: Whether to fit the scaler (for training) or use existing scaler (for prediction)
        
        Returns:
            DataFrame with processed features
        """
        if pod_df.empty:
            self.logger.warning("Empty pod metrics dataframe provided")
            return pd.DataFrame()
        
        try:
            # Select relevant columns
            feature_cols = [
                'pod_cpu_usage_percent', 'pod_memory_usage_percent', 
                'pod_network_mbps', 'restart_count'
            ]
            
            # Ensure all required columns exist, if not, add with default values
            for col in feature_cols:
                if col not in pod_df.columns:
                    self.logger.warning(f"Missing column {col} in pod metrics, adding with default values")
                    pod_df[col] = 0
            
            X = pod_df[feature_cols].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Add engineered features
            X['cpu_memory_ratio'] = X['pod_cpu_usage_percent'] / (X['pod_memory_usage_percent'] + 1)
            X['relative_restart_severity'] = np.log1p(X['restart_count'])
            
            # Add status encodings
            if 'pod_status' in pod_df.columns:
                status_dummies = pd.get_dummies(pod_df['pod_status'], prefix='status')
                X = pd.concat([X, status_dummies], axis=1)
            
            # Scale features
            if fit_scaler:
                X_scaled = self.scalers['pod'].fit_transform(X)
                self.is_fitted['pod'] = True
            elif self.is_fitted['pod']:
                X_scaled = self.scalers['pod'].transform(X)
            else:
                self.logger.warning("Scaler not fitted yet, will fit now")
                X_scaled = self.scalers['pod'].fit_transform(X)
                self.is_fitted['pod'] = True
            
            # Convert back to DataFrame
            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            # Add identifier columns back
            for col in ['pod_name', 'pod_namespace', 'node_name', 'timestamp', 'pod_status', 'container_status']:
                if col in pod_df.columns:
                    X_scaled_df[col] = pod_df[col]
            
            return X_scaled_df
        
        except Exception as e:
            self.logger.error(f"Error preparing pod features: {e}")
            return pd.DataFrame()
models/resource_predictor.py - For predicting resource exhaustion
Copyimport numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
import logging

class ResourcePredictor:
    """
    Predicts resource usage (CPU, memory, disk, network) for nodes and pods
    """
    def __init__(self, model_dir='./models'):
        self.logger = logging.getLogger(__name__)
        self.model_dir = model_dir
        self.models = {
            'node_cpu': None,
            'node_memory': None,
            'node_disk': None,
            'node_network': None,
            'pod_cpu': None,
            'pod_memory': None,
            'pod_network': None
        }
        self.thresholds = {
            'node_cpu': 85,
            'node_memory': 85,
            'node_disk': 85,
            'node_network': 80,
            'pod_cpu': 90,
            'pod_memory': 90,
            'pod_network': 85
        }
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def train_node_models(self, X, y_cpu, y_memory, y_disk, y_network):
        """
        Train models to predict node resource usage
        
        Args:
            X: Feature DataFrame
            y_cpu: CPU usage target values
            y_memory: Memory usage target values
            y_disk: Disk usage target values
            y_network: Network usage target values
        """
        try:
            # Train CPU usage model
            self.models['node_cpu'] = RandomForestRegressor(n_estimators=100, random_state=42)
            self.models['node_cpu'].fit(X, y_cpu)
            self.logger.info("Trained node CPU usage model")
            
            # Train memory usage model
            self.models['node_memory'] = RandomForestRegressor(n_estimators=100, random_state=42)
            self.models['node_memory'].fit(X, y_memory)
            self.logger.info("Trained node memory usage model")
            
            # Train disk usage model
            self.models['node_disk'] = RandomForestRegressor(n_estimators=100, random_state=42)
            self.models['node_disk'].fit(X, y_disk)
            self.logger.info("Trained node disk usage model")
            
            # Train network usage model
            self.models['node_network'] = RandomForestRegressor(n_estimators=100, random_state=42)
            self.models['node_network'].fit(X, y_network)
            self.logger.info("Trained node network usage model")
            
            # Save models
            self.save_models()
        
        except Exception as e:
            self.logger.error(f"Error training node resource models: {e}")
    
    def train_pod_models(self, X, y_cpu, y_memory, y_network):
        """
        Train models to predict pod resource usage
        
        Args:
            X: Feature DataFrame
            y_cpu: CPU usage target values
            y_memory: Memory usage target values
            y_network: Network usage target values
        """
        try:
            # Train CPU usage model
            self.models['pod_cpu'] = RandomForestRegressor(n_estimators=100, random_state=42)
            self.models['pod_cpu'].fit(X, y_cpu)
            self.logger.info("Trained pod CPU usage model")
            
            # Train memory usage model
            self.models['pod_memory'] = RandomForestRegressor(n_estimators=100, random_state=42)
            self.models['pod_memory'].fit(X, y_memory)
            self.logger.info("Trained pod memory usage model")
            
            # Train network usage model
            self.models['pod_network'] = RandomForestRegressor(n_estimators=100, random_state=42)
            self.models['pod_network'].fit(X, y_network)
            self.logger.info("Trained pod network usage model")
            
            # Save models
            self.save_models()
        
        except Exception as e:
            self.logger.error(f"Error training pod resource models: {e}")
    
    def predict_node_resources(self, X):
        """
        Predict resource usage for nodes
        
        Args:
            X: Feature DataFrame
        
        Returns:
            DataFrame with predicted resource usage
        """
        try:
            # Check if models are loaded
            if not all([self.models[model] for model in ['node_cpu', 'node_memory', 'node_disk', 'node_network']]):
                self.load_models()
            
            # Make predictions
            predicted_cpu = self.models['node_cpu'].predict(X)
            predicted_memory = self.models['node_memory'].predict(X)
            predicted_disk = self.models['node_disk'].predict(X)
            predicted_network = self.models['node_network'].predict(X)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'predicted_cpu_usage_percent': predicted_cpu,
                'predicted_memory_usage_percent': predicted_memory,
                'predicted_disk_usage_percent': predicted_disk,
                'predicted_network_load_mbps': predicted_network
            })
            
            # Add warnings based on thresholds
            results['cpu_warning'] = results['predicted_cpu_usage_percent'] > self.thresholds['node_cpu']
            results['memory_warning'] = results['predicted_memory_usage_percent'] > self.thresholds['node_memory']
            results['disk_warning'] = results['predicted_disk_usage_percent'] > self.thresholds['node_disk']
            results['network_warning'] = results['predicted_network_load_mbps'] > self.thresholds['node_network']
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error predicting node resources: {e}")
            return pd.DataFrame()
    
    def predict_pod_resources(self, X):
        """
        Predict resource usage for pods
        
        Args:
            X: Feature DataFrame
        
        Returns:
            DataFrame with predicted resource usage
        """
        try:
            # Check if models are loaded
            if not all([self.models[model] for model in ['pod_cpu', 'pod_memory', 'pod_network']]):
                self.load_models()
            
            # Make predictions
            predicted_cpu = self.models['pod_cpu'].predict(X)
            predicted_memory = self.models['pod_memory'].predict(X)
            predicted_network = self.models['pod_network'].predict(X)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'predicted_pod_cpu_usage_percent': predicted_cpu,
                'predicted_pod_memory_usage_percent': predicted_memory,
                'predicted_pod_network_mbps': predicted_network
            })
            
            # Add warnings based on thresholds
            results['cpu_warning'] = results['predicted_pod_cpu_usage_percent'] > self.thresholds['pod_cpu']
            results['memory_warning'] = results['predicted_pod_memory_usage_percent'] > self.thresholds['pod_memory']
            results['network_warning'] = results['predicted_pod_network_mbps'] > self.thresholds['pod_network']
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error predicting pod resources: {e}")
            return pd.DataFrame()
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            for model_name, model in self.models.items():
                if model is not None:
                    with open(os.path.join(self.model_dir, f"{model_name}.pkl"), 'wb') as f:
                        pickle.dump(model, f)
            self.logger.info("Saved all trained models to disk")
        
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            for model_name in self.models.keys():
                model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    self.logger.info(f"Loaded model {model_name}")
                else:
                    self.logger.warning(f"Model file {model_path} not found")
        
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")