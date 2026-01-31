"""
Model Evaluation Module for IMDB Sentiment Analysis
This module handles model evaluation and saves metrics to JSON files.
Loads model from pickle and features from data/feature folder.
"""

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import pandas as pd
import pickle
import json
import os
from datetime import datetime
import numpy as np
from scipy.sparse import load_npz


def load_model(model_path='models/logistic_regression_model.pkl'):
    """
    Load trained model from pickle file.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        model: Trained model object
    """
    print(f"Loading model from {model_path}...")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print("Model loaded successfully")
    return model


def load_test_features(feature_dir='data/feature'):
    """
    Load test features and labels from disk.
    
    Args:
        feature_dir (str): Directory containing feature files
        
    Returns:
        tuple: (X_test, y_test)
    """
    print(f"Loading test features from {feature_dir}...")
    
    # Load sparse matrices
    X_test = load_npz(os.path.join(feature_dir, 'X_test_tfidf.npz'))
    
    # Load labels
    y_test = pd.read_csv(os.path.join(feature_dir, 'y_test.csv'))['label']
    
    print(f"Test features loaded: {X_test.shape}")
    print(f"Test labels loaded: {len(y_test)}")
    
    return X_test, y_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model and calculate metrics.
    
    Args:
        model: Trained model object
        X_test: Test features
        y_test: True test labels
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    print("\nEvaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'model_name': 'Logistic Regression',
        'evaluation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_proba))
    }
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = {
        'true_negative': int(cm[0][0]),
        'false_positive': int(cm[0][1]),
        'false_negative': int(cm[1][0]),
        'true_positive': int(cm[1][1])
    }
    
    # Add sample count
    metrics['total_samples'] = int(len(y_test))
    metrics['positive_samples'] = int(np.sum(y_test == 1))
    metrics['negative_samples'] = int(np.sum(y_test == 0))
    
    return metrics


def print_metrics(metrics):
    """
    Print evaluation metrics in a readable format.
    
    Args:
        metrics (dict): Dictionary of metrics
    """
    print("\n" + "="*60)
    print(f"Model Evaluation Results: {metrics['model_name']}")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  TN: {cm['true_negative']:5d}  |  FP: {cm['false_positive']:5d}")
    print(f"  FN: {cm['false_negative']:5d}  |  TP: {cm['true_positive']:5d}")
    
    print(f"\nTotal Samples: {metrics['total_samples']}")
    print(f"Positive: {metrics['positive_samples']} | Negative: {metrics['negative_samples']}")
    print("="*60)


def save_metrics(metrics, output_dir='results'):
    """
    Save evaluation metrics to a JSON file.
    
    Args:
        metrics (dict): Dictionary of metrics
        output_dir (str): Directory to save metrics
        
    Returns:
        str: Path to saved metrics file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'evaluation_metrics_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nMetrics saved to {filepath}")
    return filepath


def main():
    """
    Main function to run model evaluation pipeline.
    """
    print("="*60)
    print("Model Evaluation Pipeline")
    print("="*60 + "\n")
    
    # Load model
    model = load_model()
    
    # Load test features
    X_test, y_test = load_test_features()
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Print metrics
    print_metrics(metrics)
    
    # Save metrics to JSON
    save_metrics(metrics)
    
    print("\n" + "="*60)
    print("Model Evaluation Completed Successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
