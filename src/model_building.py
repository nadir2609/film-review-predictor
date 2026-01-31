"""
Model Building Module for IMDB Sentiment Analysis
This module handles training machine learning models for sentiment classification.
Loads features from data/feature folder and saves model as pickle.
"""

from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
import os
from datetime import datetime
from scipy.sparse import load_npz


def load_features(feature_dir='data/feature'):
    """
    Load training features and labels from disk.
    
    Args:
        feature_dir (str): Directory containing feature files
        
    Returns:
        tuple: (X_train, y_train)
    """
    print(f"Loading features from {feature_dir}...")
    
    # Load sparse matrices
    X_train = load_npz(os.path.join(feature_dir, 'X_train_tfidf.npz'))
    
    # Load labels
    y_train = pd.read_csv(os.path.join(feature_dir, 'y_train.csv'))['label']
    
    print(f"Training features loaded: {X_train.shape}")
    print(f"Training labels loaded: {len(y_train)}")
    
    return X_train, y_train


def train_model(X_train, y_train, max_iter=1000, random_state=42):
    """
    Train logistic regression model.
    
    Args:
        X_train: Training features (sparse matrix)
        y_train: Training labels
        max_iter (int): Maximum iterations
        random_state (int): Random seed
        
    Returns:
        tuple: (trained_model, training_time)
    """
    print("Training Logistic Regression model...")
    
    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1
    )
    
    start_time = datetime.now()
    model.fit(X_train, y_train)
    end_time = datetime.now()
    
    training_time = (end_time - start_time).total_seconds()
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, training_time


def save_model(model, output_dir='models', model_name='logistic_regression_model.pkl'):
    """
    Save trained model to disk as pickle file.
    
    Args:
        model: Trained model object
        output_dir (str): Directory to save model
        model_name (str): Name of the model file
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, model_name)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {filepath}")
    return filepath


def main():
    """
    Main function to run model building pipeline.
    """
    print("="*60)
    print("Model Building Pipeline")
    print("="*60 + "\n")
    
    # Load features
    X_train, y_train = load_features()
    
    # Train model
    model, training_time = train_model(X_train, y_train)
    
    # Save model
    model_path = save_model(model)
    
    print("\n" + "="*60)
    print("Model Building Completed Successfully!")
    print(f"Training Time: {training_time:.2f} seconds")
    print("="*60)


if __name__ == "__main__":
    main()
