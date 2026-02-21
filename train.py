"""
Training entrypoint for the ML Housing Price Predictor.

This script loads a dataset from a CSV file, preprocesses the data,
trains a regression model, evaluates its performance, and saves
the trained model to disk.
"""
import sys
from ml_engine import train_model

def main() -> None:
    if len(sys.argv) != 2:
        print("Program usage: python train.py <data_csv_path>")
        sys.exit(1)

    path = sys.argv[1]
    try:
        train_model(path)
        print(f"Model trained successfully.")
    except (FileNotFoundError, ValueError) as e:
        print(e)
        sys.exit(1)    

if __name__ == "__main__":
    main()