import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import logging

class PodCrashPredictor:
    """
    Predicts pod crashes based on pod metrics and status
    """
    def __init__(self, model_dir='./models'):
        self.logger = logging.getLogger(__name__)
        self.model_dir = model_dir
        self.model = None
        self.threshold = 0.6  # Probability threshold for crash prediction
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def train(self, X, y):
        """
        Train the pod crash prediction model
        
        Args:
            X: Feature DataFrame
            y: Target values (1 for crash, 0 for no crash)
        """
        try:
            # Train crash prediction model
            self.model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
            self.model.fit(X, y)
            self.logger.info("Trained pod crash prediction model")
            
            # Save model
            self.save_model()
            
            # Evaluate model performance
            train_score = self.model.score(X, y)
            self.logger.info(f"Pod crash model training accuracy: {train_score:.4f}")
            
            return train_score
        
        except Exception as e:
            self.logger.error(f"Error training pod crash model: {e}")
            return 0
    
    def predict(self, X):
        """
        Predict pod crashes
        
        Args:
            X: Feature DataFrame
        
        Returns:
            DataFrame with crash predictions and probabilities
        """
        try:
            # Check if model is loaded
            if self.model is None:
                self.load_model()
                
                # If still None after trying to load, return empty DataFrame
                if self.model is None:
                    self.logger.error("No pod crash model available for prediction")
                    return pd.DataFrame()
            
            # Make predictions
            crash_proba = self.model.predict_proba(X)[:, 1]
            crash_pred = (crash_proba >= self.threshold).astype(int)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'crash_probability': crash_proba,
                'crash_predicted': crash_pred
            })
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error predicting pod crashes: {e}")
            return pd.DataFrame()
    
    def save_model(self):
        """Save trained model to disk"""
        try:
            if self.model is not None:
                with open(os.path.join(self.model_dir, "pod_crash_model.pkl"), 'wb') as f:
                    pickle.dump(self.model, f)
                self.logger.info("Saved pod crash model to disk")
        
        except Exception as e:
            self.logger.error(f"Error saving pod crash model: {e}")
    
    def load_model(self):
        """Load trained model from disk"""
        try:
            model_path = os.path.join(self.model_dir, "pod_crash_model.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.logger.info("Loaded pod crash model")
            else:
                self.logger.warning(f"Pod crash model file {model_path} not found")
        
        except Exception as e:
            self.logger.error(f"Error loading pod crash model: {e}")