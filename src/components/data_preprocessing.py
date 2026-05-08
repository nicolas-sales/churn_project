import os
import sys

import numpy as np
import pandas as pd
import joblib

from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.logger import logging
from src.utils.exception import CustomException


@dataclass


class DataIngestionConfig:

    
    train_data_path: str = os.path.join("artifacts","train.npy")
    test_data_path: str = os.path.join("artifacts","test.npy")
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataIngestionConfig()

    def get_data_transformer_object(self):

        # Create preprocessing pipeline

        try:
            numerical_columns = [
                "tenure",
                "MonthlyCharges",
                "TotalCharges"
            ]

            categorical_columns = [
                "gender",
                "Partner",
                "Dependents",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod"
            ]