import os 
import sys

import joblib
import shap
import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from src.utils.logger import logging
from src.utils.exception import CustomException


class ShapExplainer:

    def __init__(self):

        try:

            logging.info("Loading trained model")

            self.model = joblib.load(os.path.join("artifacts","model.pkl"))

            logging.info("model loaded successfully")

            logging.info("loading preprocessor")

            self.preprocessor = joblib.load(os.path.join("artifacts","preprocessor.pkl"))

            logging.info("preprocessor loaded successfully")


            # ==================================================
            # Initialize SHAP explainer
            # ==================================================


            logging.info("Initializing SHAP explainer")


            # ==================================================
            # Logistic Regression
            # ==================================================

            if isinstance(self.model,LogisticRegression):

                logging.info("Detected LogisticRegression model")

                self.explainer = shap.LinearExplainer(self.model,masker=None)

                self.model_type = ("linear")


            # ==================================================
            # RandomForest / XGBoost / Tree models
            # ==================================================

            elif isinstance(self.model,(XGBClassifier,GradientBoostingClassifier,AdaBoostClassifier,DecisionTreeClassifier,RandomForestClassifier)):

                logging.info(
                    "Detected tree-based model"
                )

                self.explainer = (
                    shap.TreeExplainer(
                        self.model
                    )
                )

                self.model_type = (
                    "tree"
                )

            else:

                raise Exception(
                    "Unsupported model type for SHAP"
                )

            logging.info(
                "SHAP explainer initialized successfully"
            )


        except Exception as e:
            raise CustomException(e,sys)
        
    
    def explain_prediction(self,input_data:dict,top_n:int=5):

        try:

            logging.info("Starting SHAP explanation")

            # ==================================================
            # Convert input to DataFrame
            # ==================================================

            df = pd.DataFrame([input_data])

            # ==================================================
            # Apply preprocessing
            # ==================================================

            transformed = self.preprocessor.transform(df)

            # ==================================================
            # Compute SHAP values
            # ==================================================

            shap_values = self.explainer.shap_values(transformed)

            # ==================================================
            # Handle tree-based model outputs
            # ==================================================

            if self.model_type == "tree":

                # RandomForest can return:
                # (samples, features, classes)

                if len(shap_values.shape) == 3:

                    # Select churn class
                    shap_values = (
                        shap_values[:, :, 1])


            # ==================================================
            # Extract first customer SHAP values
            # ==================================================


            shap_row = shap_values[0]
            
            
            # ==================================================
            # Get transformed feature names
            # ==================================================

            feature_names = self.preprocessor.get_feature_names_out() # Récupère les noms des features après preprocessing

            # ==================================================
            # Extract SHAP values for first customer
            # ==================================================

            #shap_row = shap_values[0]

            # ==================================================
            # Create dictionary:
            # feature -> shap value
            # ==================================================

            shap_dict = dict(
                zip(feature_names,shap_row)
                )

            # ==================================================
            # Sort features by absolute importance
            # ==================================================

            sorted_features = sorted(
                shap_dict.items(),
                key=lambda x: abs(x[1]),
                reverse=True
                )
            
            # ==================================================
            # Format explanations
            # ==================================================

            explanations = []

            for feature_name, shap_value in sorted_features[:top_n]:

                # Clean feature names

                clean_feature_name = (

                    feature_name
                    .replace("cat__","")
                    .replace("num__","")
                    .replace("_"," ")
                )

                # Determine impact direction

                if shap_value > 0:

                    impact = "Increases churn risk"

                else:

                    impact="Decreases churn risk"

                explanations.append({
                    "feature" : clean_feature_name,
                    "shap_value" : round(float(shap_value),4),
                    "impact" : impact

                })

            logging.info("SHAP explanation completed successfully")

            return explanations
            
        except Exception as e:
            raise CustomException(e,sys)
        


if __name__ == "__main__":

    sample_customer = {

        "gender": "Male",

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

    explainer = ShapExplainer()

    result = explainer.explain_prediction(input_data=sample_customer,top_n=5)

    print("\nSHAP Explanation:\n")

    for item in result:
        print(item)


