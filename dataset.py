# Sample data extraction file which generate a classification dataset using sklearn.datasets
from sklearn.datasets import make_classification
import pandas as pd
import os

def extract_data():
    if not os.path.exists("data"):
        os.mkdir("data")
    
    # Generate all samples deterministically at once to ensure DVC reproducibility
    X, y = make_classification(n_samples=100000, n_features=10, n_informative=8, n_redundant=2, n_classes=2, random_state=42)
        
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
        
    train_data = df.iloc[:80000]
    test_data = df.iloc[80000:]
        
    train_data.to_csv("data/train.csv", index=False)
    test_data.to_csv("data/test.csv", index=False)

    print("Extracted data from source successfully")

if __name__ == "__main__":
    extract_data()
