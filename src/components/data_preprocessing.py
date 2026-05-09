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
            numeric_features = [
                "tenure",
                "MonthlyCharges",
                "TotalCharges"
            ]

            categorical_features = [
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

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        StandardScaler(),
                        numeric_features
                    ),
                    (
                        "cat",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        ),
                        categorical_features
                    )        
                ],
                remainder="passthrough"
            )

            logging.info("Preproessing object created successfully")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,raw_data_path):

        try:

            # Read raw dataset
            df=pd.read_csv(raw_data_path)

            logging.info("Raw dataset loaded successfully")

            # Convert TotalCharges
            df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce") # errors="coerce" -> transforme en Nan une valeur pas convertible

            # Remove missing values
            df=df.dropna()

            # Remove customerID
            df.drop("customerID",axis=1,inplace=True)

            logging.info("Data cleaning completed")

            # Convert target columns
            df["Churn"]=df["Churn"].map({"Yes":1,"No":0})

            logging.info("Target column converted")

            # Split the data
            X=df.drop("Churn",axis=1)
            y=df["Churn"]

            # Train Test Split
            X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

            logging.info("Train Test Split completed")

            # Get preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Fit transform train
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            # Transform test
            X_test_arr = preprocessing_obj.transform(X_test)

            logging.info("Data preprocessing completed")

            # Concatenate target column
            train_arr = np.c_[X_train_arr,np.array(y_train)]

            test_arr = np.c_[X_test_arr,np.array(y_test)]

            # Save train array
            np.save(self.data_transformation_config.train_data_path,train_arr)

            # Save test array
            np.save(self.data_transformation_config.test_data_path,test_arr)

            logging.info("Train and test arrays saved successfully")

            # Save preprocessor object
            joblib.dump(preprocessing_obj,self.data_transformation_config.preprocessor_obj_file_path)

            logging.info("Preprocessor object saved successfully")

            return (
                self.data_transformation_config.train_data_path,
                self.data_transformation_config.test_data_path,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":

    from src.components.data_ingestion import DataIngestion

    ingestion = DataIngestion()

    raw_data_path = ingestion.initiate_data_Ingestion()

    transformation = DataTransformation()

    train_path,test_path,preprocessor_path = transformation.initiate_data_transformation(raw_data_path)

    print(train_path)
    print(test_path)
    print(preprocessor_path)