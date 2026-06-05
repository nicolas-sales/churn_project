import sys

from src.components.prediction import Prediction

from src.components.shap_explainer import ShapExplainer

from src.components.recommendation import Recommendation

from src.monitoring.prediction_logger import Predictionlogger

from src.utils.logger import logging
from src.utils.exception import CustomException


class PredictionPipeline:

    def __init__(self):

        # ==================================================
        # Initialize components
        # ==================================================

        self.predictor = (
            Prediction()
        )

        self.explainer = (
            ShapExplainer()
        )

        self.recommender = (
            Recommendation()
        )

        self.prediction_logger = (
            Predictionlogger()
        )

    def run_pipeline(
        self,
        input_data: dict
    ):

        try:

            logging.info(
                "Prediction pipeline started"
            )

            # ==================================================
            # CHURN PREDICTION
            # ==================================================

            prediction_result = (

                self.predictor
                .predict_churn(
                input_data
                )
            )
            

            self.prediction_logger.log_prediction(
                input_data=input_data,
                prediction_result=prediction_result
            )

            logging.info(
                "Prediction completed successfully"
            )

            # ==================================================
            # SHAP EXPLANATION
            # ==================================================

            shap_result = (

                self.explainer
                .explain_prediction(
                    input_data=input_data,
                    top_n=5
                )
            )

            logging.info(
                "SHAP explanation completed successfully"
            )

            # ==================================================
            # RECOMMENDATIONS
            # ==================================================

            recommendation_result = (

                self.recommender
                .generate_recommendations(
                    input_data
                )
            )

            logging.info(
                "Recommendations generated successfully"
            )

            # ==================================================
            # FINAL OUTPUT
            # ==================================================

            final_result = {

                "prediction":
                    prediction_result,

                "shap_explanation":
                    shap_result,

                "recommendations":
                    recommendation_result
            }

            logging.info(
                "Prediction pipeline completed successfully"
            )

            return final_result

        except Exception as e:

            raise CustomException(e, sys)


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

    pipeline = PredictionPipeline()

    result = pipeline.run_pipeline(sample_customer)

    print("\nPrediction Pipeline Result:\n")

    print(result)





