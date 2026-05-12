import os
import sys

import joblib
import pandas as pd

from dataclasses import dataclass

from src.utils.logger import logging
from src.utils.exception import CustomException


@dataclass

class PredictionConfig:

    model_path = os.path.join("artifacts","model.pkl")

    preprocessor_path = os.path.join("artifacts","preprocessor.pkl")

class Prediction:

    def __init__(self):

        self.prediction_config = PredictionConfig()

    def load_object(self):

        try:

            logging.info("Loading trained model")

            model = joblib.load(self.prediction_config.model_path)

            logging.info("Model loaded successfully")

            logging.info("Loading preprocessor")

            preprocessor = joblib.load(self.prediction_config.preprocessor_path)

            logging.info("Preprocessor loaded successfully")

            return model, preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def predict_churn(self,input_data:str):

        try:

            logging.info("Starting churn prediction")

            # Convert dictionnary into dataframe

            input_df = pd.DataFrame([input_data])

            logging.info("Input converted to dataframe")

            # Load model and preprocessor

            model, preprocessor = self.load_object()

            # Transform input data

            transformed_data = preprocessor.transform(input_df)

            logging.info("Input data transformed successfully")

            # Predict probability

            churn_probability = model.predict_proba(transformed_data)[:,1][0] # cretourn la probabilité pour 1 en une valeur et pas un numpy

            # Predict class

            churn_prediction = model.predict(transformed_data)[0] # pour recuperer la valeur et pas un numpy

            logging.info("Prediction completed successfully")

            # Risk probability
            if churn_probability > 0.7:
                risk_level = "High churn risk"

            elif churn_probability > 0.4:
                risk_level = "Medium churn risk"

            else:
                risk_level = "Low churn risk"

            # Prediction response

            result = {
                "churn_probability": round(float(churn_probability),4),
                "prediction": int(churn_prediction),
                "risk_level": risk_level
            }

            return result
        
        except Exception as e:
            raise CustomException(e,sys)
        


if __name__=="__main__":

    sample_customer = {

        "gender": "Female",

        "SeniorCitizen": 0,

        "Partner": "No",

        "Dependents": "No",

        "tenure": 2,

        "PhoneService": "Yes",

        "MultipleLines": "No",

        "InternetService": "Fiber optic",

        "OnlineSecurity": "No",

        "OnlineBackup": "No",

        "DeviceProtection": "No",

        "TechSupport": "No",

        "StreamingTV": "No",

        "StreamingMovies": "No",

        "Contract": "Month-to-month",

        "PaperlessBilling": "Yes",

        "PaymentMethod": "Electronic check",

        "MonthlyCharges": 90,

        "TotalCharges": 180
    }

    predictor = Prediction()

    result = predictor.predict_churn(sample_customer)

    print("\nPrediction result:\n")

    print(result)

