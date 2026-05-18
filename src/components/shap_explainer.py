import os 
import sys

import joblib
import shap
import pandas as pd

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
        # Create SHAP explainer
        # ==================================================


            logging.info("Initializing SHAP explainer")

            self.explainer = shap.TreeExplainer(self.model)

            logging.info("SHAP explainer initialized")

        except Exception as e:
            raise CustomException(e,sys)
