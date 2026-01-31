import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_data(file_path):
    return pd.read_csv(file_path)

def split_data(data, test_size=0.2, random_state=42):
    return train_test_split(data, test_size=test_size, random_state=random_state, stratify=data['sentiment'])

def save_data(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

def main():
    # Load data from the root directory
    data = load_data('IMDB.csv')
    
    print(f"Loaded {len(data)} records")
    print(f"Columns: {data.columns.tolist()}")
    
    # Split data
    train_data, test_data = split_data(data)
    
    print(f"Train data: {len(train_data)} records")
    print(f"Test data: {len(test_data)} records")
    
    # Save data
    save_data(train_data, 'data/raw/train_data.csv')
    save_data(test_data, 'data/raw/test_data.csv')

if __name__ == "__main__":
    main()

