"""
Inference entrypoint for the ML Housing Price Predictor.

This script loads a previously trained model and generates
a housing price prediction based on input features
provided via the command line.
"""
import sys
import argparse
from ml_engine import predict_price

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a housing price prediction.",
        usage="python predict.py --area AREA --rooms ROOMS --age AGE",
        epilog="Example: python predict.py --area 100 --rooms 3 --age 10",
    )
    parser.add_argument("--area", type=float, required=True, help="Area of the house (in square meters)")
    parser.add_argument("--rooms", type=int, required=True, help="Number of rooms")
    parser.add_argument("--age", type=int, required=True, help="Age of the house")
    args = parser.parse_args()
    features = {
        "area": args.area,
        "rooms": args.rooms,
        "age": args.age
    }

    try:
        predict_price(features)
        print(f"Prediction generated successfully.")
    except (FileNotFoundError, ValueError) as e:
        print(e)
        sys.exit(1)    

if __name__ == "__main__":
    main()