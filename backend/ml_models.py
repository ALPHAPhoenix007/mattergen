"""
Machine Learning models for property prediction
Uses scikit-learn for interpretable, hackathon-friendly models
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Dict, Tuple, Optional, List
import joblib
from backend.features import extract_features_from_smiles, get_feature_names
from utils.config import ML_CONFIG


class PropertyPredictor:
    """
    ML model for predicting material properties from molecular structure
    """
    
    def __init__(self, property_name: str = 'band_gap_ev'):
        self.property_name = property_name
        self.model = None
        self.feature_mean = None
        self.feature_std = None
        self.feature_names = get_feature_names()
        self.trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray, model_type: str = 'random_forest'):
        """
        Train the prediction model
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            model_type: Type of model ('random_forest', 'gradient_boosting', 'ridge')
        """
        # Normalize features
        self.feature_mean = np.mean(X, axis=0)
        self.feature_std = np.std(X, axis=0)
        self.feature_std = np.where(self.feature_std == 0, 1, self.feature_std)
        
        X_normalized = (X - self.feature_mean) / self.feature_std
        
        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=ML_CONFIG['random_state'],
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=ML_CONFIG['random_state']
            )
        else:  # ridge
            self.model = Ridge(alpha=1.0, random_state=ML_CONFIG['random_state'])
        
        # Train
        self.model.fit(X_normalized, y)
        self.trained = True
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, X_normalized, y, 
            cv=ML_CONFIG['cv_folds'], 
            scoring='r2'
        )
        
        return {
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted property values
        """
        if not self.trained or self.model is None:
            raise ValueError("Model not trained yet")
        
        # Normalize using training statistics
        X_normalized = (X - self.feature_mean) / self.feature_std
        
        predictions = self.model.predict(X_normalized)
        return predictions
    
    def predict_from_smiles(self, smiles: str) -> Optional[float]:
        """
        Predict property directly from SMILES string
        
        Args:
            smiles: SMILES representation
        
        Returns:
            Predicted property value or None
        """
        features = extract_features_from_smiles(smiles)
        if features is None:
            return None
        
        # Reshape for single prediction
        features = features.reshape(1, -1)
        
        prediction = self.predict(features)
        return float(prediction[0])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores (for tree-based models)
        
        Returns:
            Dictionary of feature names and importance scores
        """
        if not self.trained:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            importance_dict = {
                name: float(imp) 
                for name, imp in zip(self.feature_names, importances)
            }
            # Sort by importance
            importance_dict = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
            return importance_dict
        else:
            return {}
    
    def save(self, filepath: str):
        """Save model to disk"""
        if not self.trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'feature_names': self.feature_names,
            'property_name': self.property_name
        }
        joblib.dump(model_data, filepath)
    
    def load(self, filepath: str):
        """Load model from disk"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_mean = model_data['feature_mean']
        self.feature_std = model_data['feature_std']
        self.feature_names = model_data['feature_names']
        self.property_name = model_data['property_name']
        self.trained = True


class MultiPropertyPredictor:
    """
    Predict multiple properties simultaneously
    """
    
    def __init__(self, property_names: List[str]):
        self.property_names = property_names
        self.models = {
            prop: PropertyPredictor(property_name=prop)
            for prop in property_names
        }
    
    def train_all(self, df: pd.DataFrame, smiles_col: str = 'smiles'):
        """
        Train models for all properties
        
        Args:
            df: DataFrame with SMILES and property columns
            smiles_col: Column name for SMILES
        """
        # Extract features for all molecules
        features_list = []
        valid_indices = []
        
        for idx, smiles in enumerate(df[smiles_col]):
            features = extract_features_from_smiles(smiles)
            if features is not None:
                features_list.append(features)
                valid_indices.append(idx)
        
        X = np.vstack(features_list)
        
        # Train each property model
        results = {}
        for prop in self.property_names:
            if prop in df.columns:
                y = df[prop].iloc[valid_indices].values
                score = self.models[prop].train(X, y, model_type='random_forest')
                results[prop] = score
        
        return results
    
    def predict_all(self, smiles: str) -> Dict[str, float]:
        """
        Predict all properties for a molecule
        
        Args:
            smiles: SMILES string
        
        Returns:
            Dictionary of property predictions
        """
        predictions = {}
        for prop, model in self.models.items():
            pred = model.predict_from_smiles(smiles)
            if pred is not None:
                predictions[prop] = pred
        
        return predictions


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
