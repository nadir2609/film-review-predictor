"""
Feature Engineering Module for IMDB Sentiment Analysis
This module handles feature extraction from preprocessed text data.
Saves features to data/feature folder.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os
from scipy.sparse import save_npz


def load_preprocessed_data(filepath='data/processed/train_processed.csv'):
    """
    Load preprocessed data from CSV file.
    
    Args:
        filepath (str): Path to the preprocessed data file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    print(f"Loading preprocessed data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records")
    return df


def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        df (pd.DataFrame): Dataframe with 'review' and 'sentiment' columns
        test_size (float): Proportion of test set
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"Splitting data (test_size={test_size})...")
    
    # Encode sentiment labels (positive: 1, negative: 0)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    X = df['review']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Training set sentiment distribution:\n{y_train.value_counts()}")
    print(f"Test set sentiment distribution:\n{y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test


def create_tfidf_features(X_train, X_test, max_features=None):
    """
    Create TF-IDF features from text data.
    
    Args:
        X_train: Training text data
        X_test: Test text data
        max_features (int): Maximum number of features
        
    Returns:
        tuple: (X_train_vectorized, X_test_vectorized, vectorizer)
    """
    print("Creating TF-IDF features...")
    
    vectorizer = TfidfVectorizer(max_features=max_features)
    
    # Fit and transform training data
    X_train_vectorized = vectorizer.fit_transform(X_train)
    print(f"Training features shape: {X_train_vectorized.shape}")
    
    # Transform test data
    X_test_vectorized = vectorizer.transform(X_test)
    print(f"Test features shape: {X_test_vectorized.shape}")
    
    return X_train_vectorized, X_test_vectorized, vectorizer


def save_features(X_train_vec, X_test_vec, y_train, y_test, vectorizer, output_dir='data/feature'):
    """
    Save vectorized features, labels, and vectorizer to disk.
    
    Args:
        X_train_vec: Vectorized training features
        X_test_vec: Vectorized test features
        y_train: Training labels
        y_test: Test labels
        vectorizer: Fitted TF-IDF vectorizer
        output_dir (str): Directory to save features
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving features to {output_dir}...")
    
    # Save sparse matrices
    save_npz(os.path.join(output_dir, 'X_train_tfidf.npz'), X_train_vec)
    save_npz(os.path.join(output_dir, 'X_test_tfidf.npz'), X_test_vec)
    
    # Save labels
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False, header=['label'])
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False, header=['label'])
    
    # Save vectorizer
    with open(os.path.join(output_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("Features saved successfully!")
    print(f"  - X_train_tfidf.npz: {X_train_vec.shape}")
    print(f"  - X_test_tfidf.npz: {X_test_vec.shape}")
    print(f"  - y_train.csv: {len(y_train)} labels")
    print(f"  - y_test.csv: {len(y_test)} labels")
    print(f"  - tfidf_vectorizer.pkl")


def main():
    """
    Main function to run feature engineering pipeline.
    """
    print("="*60)
    print("Feature Engineering Pipeline")
    print("="*60 + "\n")
    
    # Load preprocessed data
    df = load_preprocessed_data('data/processed/train_processed.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Create TF-IDF features
    X_train_vec, X_test_vec, vectorizer = create_tfidf_features(X_train, X_test)
    
    # Save features
    save_features(X_train_vec, X_test_vec, y_train, y_test, vectorizer)
    
    print("\n" + "="*60)
    print("Feature Engineering Completed Successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
