import os
import sys

import pandas as pd


from src.utils.exception import CustomException
from src.utils.logger import logging

from dataclasses import dataclass

@dataclass

class DataIngestionConfig:

    raw_data_path = os.path.join(
        "artifacts",
        "raw.csv"
    )


class DataIngestion:

    def __init__(self):

        self.ingestion_config = DataIngestionConfig()

    def initiate_data_Ingestion(self):

        logging.info("Entered the data ingestion method")

        try:

            df = pd.read_csv("data/Telco-Customer-Churn.csv")

            logging.info("Dataset loaded successfully")

            # Create artifacts folder
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            # Save raw dataset
            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info("Ingestion completed")

            return (
                self.ingestion_config.raw_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        



if __name__ == "__main__":

    ingestion = DataIngestion()

    raw_data_path = (ingestion.initiate_data_Ingestion())

    print(raw_data_path)