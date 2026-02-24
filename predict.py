"""
Inference entrypoint for the ML Housing Price Predictor.

This script loads a previously trained model and generates
a housing price prediction based on input features
provided via the command line.
"""
import sys
import argparse

def main() -> None:
    from ml_engine import predict_price
    parser = argparse.ArgumentParser(
        description="Generate a housing price prediction.",
        usage="python predict.py --area AREA --rooms ROOMS --age AGE --path MODEL_PATH",
        epilog="Example: python predict.py --area 100 --rooms 3 --age 10",
    )
    parser.add_argument("--area", type=float, required=True, help="Area of the house (in square meters)")
    parser.add_argument("--rooms", type=int, required=True, help="Number of rooms")
    parser.add_argument("--age", type=int, required=True, help="Age of the house")
    parser.add_argument("--path", type=str, required=False, help="Path to the trained model file")
    args = parser.parse_args()
    features = {
        "area": args.area,
        "rooms": args.rooms,
        "age": args.age
    }

    prediction = predict_price(features, model_path=args.path)
    print(f"Predicted price: {prediction}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrediction interrupted by user. Exiting.")
        sys.exit(130)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        sys.exit(1)