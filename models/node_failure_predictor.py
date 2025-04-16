import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import logging

class NodeFailurePredictor:
    """
    Predicts node failures based on node metrics and conditions
    """
    def __init__(self, model_dir='./models'):
        self.logger = logging.getLogger(__name__)
        self.model_dir = model_dir
        self.model = None
        self.threshold = 0.65  # Probability threshold for failure prediction
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def train(self, X, y):
        """
        Train the node failure prediction model
        
        Args:
            X: Feature DataFrame
            y: Target values (1 for failure, 0 for no failure)
        """
        try:
            # Train failure prediction model
            self.model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
            self.model.fit(X, y)
            self.logger.info("Trained node failure prediction model")
            
            # Save model
            self.save_model()
            
            # Evaluate model performance
            train_score = self.model.score(X, y)
            self.logger.info(f"Node failure model training accuracy: {train_score:.4f}")
            
            return train_score
        
        except Exception as e:
            self.logger.error(f"Error training node failure model: {e}")
            return 0
    
    def predict(self, X):
        """
        Predict node failures
        
        Args:
            X: Feature DataFrame
        
        Returns:
            DataFrame with failure predictions and probabilities
        """
        try:
            # Check if model is loaded
            if self.model is None:
                self.load_model()
                
                # If still None after trying to load, return empty DataFrame
                if self.model is None:
                    self.logger.error("No node failure model available for prediction")
                    return pd.DataFrame()
            
            # Make predictions
            failure_proba = self.model.predict_proba(X)[:, 1]
            failure_pred = (failure_proba >= self.threshold).astype(int)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'failure_probability': failure_proba,
                'failure_predicted': failure_pred
            })
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error predicting node failures: {e}")
            return pd.DataFrame()
    
    def save_model(self):
        """Save trained model to disk"""
        try:
            if self.model is not None:
                with open(os.path.join(self.model_dir, "node_failure_model.pkl"), 'wb') as f:
                    pickle.dump(self.model, f)
                self.logger.info("Saved node failure model to disk")
        
        except Exception as e:
            self.logger.error(f"Error saving node failure model: {e}")
    
    def load_model(self):
        """Load trained model from disk"""
        try:
            model_path = os.path.join(self.model_dir, "node_failure_model.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.logger.info("Loaded node failure model")
            else:
                self.logger.warning(f"Node failure model file {model_path} not found")
        
        except Exception as e:
            self.logger.error(f"Error loading node failure model: {e}")