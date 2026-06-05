import os
from datetime import datetime

import pandas as pd


class Predictionlogger:

    def __init__(self):

        self.monitoring_dir = "monitoring"
        os.makedirs(self.monitoring_dir, exist_ok=True)

        self.file_path = os.path.join(
            self.monitoring_dir,
            "predictions.csv"
        )

    def log_prediction(self, input_data: dict, prediction_result: dict):

        row = {
            "timestamp": datetime.now(),
            **input_data,
            "churn_probability": prediction_result["churn_probability"],
            "prediction": prediction_result["prediction"],
            "risk_level": prediction_result["risk_level"]
        }

        df = pd.DataFrame([row])

        if os.path.exists(self.file_path):

            df.to_csv(
                self.file_path,
                mode="a",
                header=False,
                index=False
            )

        else:

            df.to_csv(
                self.file_path,
                index=False
            )