import sys
import copy

from src.components.prediction import Prediction

from src.utils.logger import logging
from src.utils.exception import CustomException


class Simulation:

    def __init__(self):

        self.predictor = Prediction()

    def simulate_change(self,input_data:dict, feature:str, new_value):

        try:

            # Original prediction
            original_result = self.predictor.predict_churn(input_data)

            original_probability = original_result["churn_probability"]

            logging.info(f"Original churn probability: "f"{original_probability}")

            # Copy customer data
            simulated_input = input_data.copy()

            # Modify selected feature
            simulated_input[feature] = new_value

            logging.info(f"Feature modified: " f"{feature} -> {new_value}")

            # New prediction

            simulated_result = self.predictor.predict_churn(simulated_input)

            simulated_probability = (simulated_result["churn_probability"])

            logging.info(f"Simulated churn probability: " f"{simulated_probability}")

            # Probability difference
            probability_difference = round(simulated_probability - original_probability,4)

            # Simulation interpretation
            if probability_difference > 0:
                impact = "Increased churn risk"

            elif probability_difference < 0:
                impact = "Reduced churn risk"

            else:
                impact = "No inpact"

            logging.info("Simulation completed successfully")

            # Return simulation result

            result = {
                "feature changed":feature,
                "new_value":new_value,
                "original_probability":original_probability,
                "new_probability":simulated_probability,
                "probability_change":probability_difference,
                "impact":impact
            }

            return result
        
        except Exception as e:
            CustomException(e,sys)


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

    simulator = Simulation()

    result = simulator.simulate_change(input_data=sample_customer, feature="Contract", new_value="One year")

    print("\nSimulation result:\n")

    print(result)

