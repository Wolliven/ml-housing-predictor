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


def train_model(data_path, model_path="model.pkl"):
    if not data_path.endswith(".csv"):
        raise ValueError("Invalid data file format. Please provide a CSV file.")
    df = pd.read_csv(data_path)
    df = df.dropna()
    X = df.drop(columns=["price"])
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {"r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred)}
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return metrics, model_path




def predict_price(features, model_path="model.pkl"):
    if not model_path:
        model_path = "model.pkl"
    if not model_path.endswith(".pkl"):
        raise ValueError("Invalid model file format. Please provide a .pkl file.")
    area = features.get("area")
    rooms = features.get("rooms")
    age = features.get("age")
    if area is None or rooms is None or age is None:
        raise ValueError("Missing required features. Please provide 'area', 'rooms', and 'age'.")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    input_data = pd.DataFrame([features])
    prediction = model.predict(input_data)[0]
    return prediction