"""
Inference entrypoint for the ML Housing Price Predictor.

This script loads a previously trained model and generates
a housing price prediction based on input features
provided via the command line.
"""
import sys
from ml_engine import predict_price

def main() -> None:
    if len(sys.argv) != 2:
        print("Program usage: python predict.py <features_path>")
        sys.exit(1)

    path = sys.argv[1]
    try:
        predict_price(path)
        print(f"Prediction generated successfully.")
    except (FileNotFoundError, ValueError) as e:
        print(e)
        sys.exit(1)    

if __name__ == "__main__":
    main()