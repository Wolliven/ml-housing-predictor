"""
Core machine learning logic for the ML Housing Price Predictor.

This module contains the data preprocessing, model training,
evaluation, and prediction functions used by the CLI entrypoints.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle


def train_model(data_path : str, model_path : str ="model.pkl") -> tuple[dict, str]:
    if not model_path:
        model_path = "model.pkl"
    if not data_path.endswith(".csv"):
        raise ValueError("Invalid data file format. Please provide a CSV file.")
    df = pd.read_csv(data_path)
    missing_rows = df.isna().any(axis=1).sum()
    if missing_rows > 0:
        print(f"Warning: {missing_rows} row/s with missing values will be dropped.")
    df = df.dropna()
    features = ["area", "rooms", "age"]
    missing = [feat for feat in features if feat not in df.columns]
    if missing:
        raise ValueError(f"Missing required features in the dataset: {', '.join(missing)}")
    X = df[features]
    if "price" not in df.columns:
        raise ValueError("Missing target variable 'price' in the dataset.")
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {"r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred)}
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "features": features
        }, f)
    return metrics, model_path




def predict_price(features : dict, model_path : str ="model.pkl") -> float:
    if not model_path:
        model_path = "model.pkl"
    if not model_path.endswith(".pkl"):
        raise ValueError("Invalid model file format. Please provide a .pkl file.")
    area = features.get("area")
    rooms = features.get("rooms")
    age = features.get("age")
    if area is None or rooms is None or age is None:
        raise ValueError("Missing required features. Please provide 'area', 'rooms', and 'age'.")
    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
            model = model_data["model"]
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    expected = model_data["features"]
    missing = [feat for feat in expected if feat not in features]
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")
    input_data = pd.DataFrame([[features[c] for c in expected]], columns=expected)
    prediction = model.predict(input_data)[0]
    return prediction