# ML Housing Price Predictor

A modular machine learning project that implements a complete supervised learning pipeline to predict housing prices from structured data.

This project focuses on understanding the fundamentals of:

- Data preprocessing
- Supervised regression models
- Model evaluation
- Model persistence
- Separation between training and inference
- Clean project architecture

---

## Project Goals

The objective of this project is to build a clean and extensible ML pipeline that:

1. Loads structured data from a CSV file
2. Preprocesses and prepares features
3. Trains a regression model
4. Evaluates performance using appropriate metrics
5. Saves the trained model to disk
6. Performs predictions using new input data

This project emphasizes architectural clarity over complexity.

---

## Features

- Structured data loading with pandas
- Train/test split
- Supervised regression model (scikit-learn)
- Evaluation metrics (e.g., MAE, MSE, R²)
- Model serialization using pickle
- Separate training and inference scripts
- CLI-based interaction

---

## Project Structure

```

ml-housing-predictor/
├── train.py        # Model training entrypoint
├── predict.py      # Inference entrypoint
├── ml_engine.py    # Core ML logic (data prep, training, prediction)
├── requirements.txt
├── README.md
└── .gitignore

```

---

## Usage

### Train a model

```

python train.py data.csv

```

This will:
- Load the dataset
- Train the model
- Print evaluation metrics
- Save the trained model as `model.pkl`

---

### Make predictions

```

python predict.py --area 70 --rooms 3 --age 5

```

This will:
- Load the saved model
- Generate a predicted housing price

---

## Dataset Schema

Columns:
- area (float) - square meters
- rooms (int) - number of rooms
- age (int) - age of property in years
- price (int) - target variable

---

## Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn

Install dependencies:

```

pip install -r requirements.txt

```

---

## Scope

This project intentionally focuses on classical machine learning techniques.

Out of scope for this version:

- Deep learning
- Hyperparameter optimization frameworks
- Deployment
- Distributed training
- Large-scale datasets

---

## Learning Focus

This project is designed as a stepping stone toward more advanced ML systems engineering.

It prioritizes:

- Correctness
- Modularity
- Clear separation of concerns
- Understanding model behavior

---

## Future Extensions

Possible improvements:

- Feature scaling pipelines
- Categorical feature encoding
- Logging
- API exposure with FastAPI
- Basic frontend for prediction
- Model versioning