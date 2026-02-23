"""
Core machine learning logic for the ML Housing Price Predictor.

This module contains the data preprocessing, model training,
evaluation, and prediction functions used by the CLI entrypoints.
"""
import pandas as pd
from sklearn.model_selection import train_test_split



def train_model(data_path):
    # Placeholder for data loading and preprocessing
    print(f"Loading and preprocessing data from {data_path}...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=["price"])
    Y = df["price"]
    print(f"X info: {X.info()}, Y info: {Y.info()}")

def predict_price(features):
    # Placeholder for loading the model and making a prediction
    print(f"Predicting price for features: {features}...")