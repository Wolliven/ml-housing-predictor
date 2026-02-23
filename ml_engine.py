"""
Core machine learning logic for the ML Housing Price Predictor.

This module contains the data preprocessing, model training,
evaluation, and prediction functions used by the CLI entrypoints.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle


def train_model(data_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=["price"])
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    return r2




def predict_price(features):
    # Placeholder for loading the model and making a prediction
    print(f"Predicting price for features: {features}...")