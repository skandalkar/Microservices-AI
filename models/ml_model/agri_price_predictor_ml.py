"""
    Price Prediction Model for Krishi Unnati AI Services
    This module provides a machine learning model for predicting agricultural product prices
    based on various features such as product type, quality, season, state, quantity,
    market distance, and organic certification status.
"""

# Import necessary libraries

# Suppress warnings for cleaner output
# remove warnings in future
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)

# Data manipulation and visualization libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Machine learning libraries
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

"""
    Price Prediction Model for Krishi Unnati AI Services
"""


class agriPricePredictor:
    """
       A class to represent the agricultural price prediction model.
       This class uses a Random Forest Regressor to predict prices and provides methods
       for training, saving, loading, and making predictions.
    """

    def __init__(self):
        self.encoders = {}
        self.model = RandomForestRegressor()
        self.feature_names = []

    """
       Initializes the agriPricePredictor class.
       Sets up the Random Forest model, encoders for categorical features, and feature names.
    """

    # This function automatically converts text columns into numeric labels for models.
    # During training (fit=True), it learns mappings.
    # During prediction (fit=False), it reuses the same mappings to ensure consistency.

    def prepare_features(self, df, fit=True):

        """
               Encodes categorical features into numeric labels for the model.

               Args:
                   df (pd.DataFrame): The input data containing features.
                   fit (bool): If True, fits the encoders to the data. If False, uses existing encoders.

               Returns:
               pd.DataFrame: The encoded DataFrame.
        """

        # Encode categorical features
        categorical_cols = ['product', 'quality', 'season', 'state']
        df_encoded = df.copy()
        for col in categorical_cols:
            if fit:
                self.encoders[col] = LabelEncoder()
                df_encoded[col] = self.encoders[col].fit_transform(df[col])
            else:
                df_encoded[col] = self.encoders[col].transform(df[col])
        return df_encoded

    # for training
    def train(self, df):

        """
                Trains the Random Forest model on the provided dataset.

                Args:
                    df (pd.DataFrame): The training dataset containing features and target.

                Returns:
                    tuple: Mean Absolute Error (MAE) and R² score of the model.
        """

        print("Preparing features...")
        df_encoded = self.prepare_features(df, fit=True)

        # Features and target
        feature_cols = ['product', 'quality', 'season', 'state', 'quantity', 'market_distance', 'organic_certified']
        self.feature_names = feature_cols

        X = df_encoded[feature_cols]
        y = df_encoded['price_per_quintal']

        # Splitting dataset 80% for training and rest 20% for test and performance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Training on {len(X_train)} samples...")

        # Train Random Forest
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\nModel Performance:")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"R² Score: {r2:.4f}")

        # Feature importance
        importances = self.model.feature_importances_
        feature_importance = sorted(
            zip(feature_cols, importances),
            key=lambda x: x[1],
            reverse=True
        )

        print("\nFeature Importance:")
        for feat, imp in feature_importance:
            print(f"{feat}: {imp:.4f}")

        return mae, r2

    def predict(self, input_data):

        """
                Predicts the price and confidence interval for a given input.

                Args:
                    input_data (dict): A dictionary containing feature values for prediction.

                Returns:
                    dict: A dictionary containing predicted price, min price, max price, and confidence.
        """

        if self.model is None:
            raise ValueError("Model not trained yet!")

        df = pd.DataFrame([input_data])
        df_encoded = self.prepare_features(df, fit=False)
        X = df_encoded[self.feature_names]

        prediction = self.model.predict(X)[0]

        # Calculate confidence interval (using prediction std from trees)
        predictions_all = np.array([tree.predict(X)[0]
                                    for tree in self.model.estimators_])
        std = np.std(predictions_all)

        return {
            'price_per_quintal': round(prediction, 2),
            'min_price': round(prediction - 1.5 * std, 2),
            'max_price': round(prediction + 1.5 * std, 2),
            'confidence': min(95, max(75, 100 - (std / prediction) * 100))
        }

    # load and save model
    def save_model(self, filepath='agri_price_model.pkl'):

        """
                Saves the trained model and encoders to a file.

                Args:
                    filepath (str): The file path to save the model.
        """

        model_data = {
            'model': self.model,
            'encoders': self.encoders,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='agri_price_model.pkl'):

        """
                Loads a trained model and encoders from a file.

                Args:
                    filepath (str): The file path to load the model from.
        """

        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.encoders = model_data['encoders']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filepath}")


# load dataset
# Provide the absolute path to the CSV file
# absolute_path = C:\Users\Santosh\PycharmProjects\KrishiUnnati-AI\dataset\agri_training_data.csv

file_path = "/Users/Santosh/PycharmProjects/KrishiUnnati-AI/dataset/agri_training_data.csv"
#Please change the path according to your system

try:
    data = pd.read_csv(file_path)
    print(data.head())
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")


# Display dataset information and perform exploratory data analysis (EDA)
print("\nOriginal Data")
print(data)

# EDA for dataset
print("\nDataset head entries:\n", data.head())
print("\nDataset tail entries:\n", data.tail())
print("\nDataset no. of dims:", data.ndim)
print("\nDataset shapes total entries:", data.shape)
print("\nDataset statistical description:\n", data.describe())
print("\nDataset checking total count of missing values:\n", data.isnull().sum())
print("\nDataset checking any missing values:\n", data.isnull().any())
print("\nCounts any duplicate values:", data.duplicated().any())
print("\nCounts total duplicate values:", data.duplicated().sum())
print("\n\n")

# dataset visualization

# 1️. Average price per state
plt.figure(figsize=(10, 5))
sns.barplot(x='state', y='price_per_quintal', data=data, estimator='mean')
plt.title('Average Crop Price per State')
plt.xticks(rotation=45)

# 2. Relationship between distance and price
plt.figure(figsize=(8, 5))
sns.scatterplot(x='market_distance', y='price_per_quintal', hue='organic_certified', data=data)
plt.title('Distance vs Price (Colored by Organic Certification)')

# 3. Price distribution by season
plt.figure(figsize=(8, 5))
sns.barplot(x='season', y='price_per_quintal', data=data)
plt.title('Price Distribution Across Seasons')

# Calling class members

model = agriPricePredictor()
model.train(data)

new_data = {
    'product': 'rice',
    'quality': 'high',
    'season': 'rabi',
    'state': 'haryana',
    'quantity': 100,
    'market_distance': 20,
    'organic_certified': 1
}

predictions = model.predict(new_data)
# print(predictions)
# plt.show()
model.save_model()