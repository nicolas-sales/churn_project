import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.exception import CustomException
from src.utils.logger import logging

from dataclasses import dataclass

@dataclass

class DataIngestionConfig:

    raw_data_path: str = os.path.join("artifacts","raw.csv")
    train_data_path: str = os.path.join("artifacts","train.csv")
    test_data_path: str = os.path.join("artifacts","test.csv")


class DataIngestion:

    def __init__(self):

        self.ingestion_config = DataIngestionConfig()

    def initiate_data_Ingestion(self):

        logging.info("Entered the data ingestion method")

        try:

            df = pd.read_csv("data/Telco-Customer-Churn.csv")

            logging.info("Dataset loaded successfully")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info("Train Test Split initiated")

            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False)

            logging.info("Ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        



if __name__ == "__main__":

    obj = DataIngestion()

    train_data, test_data = (obj.initiate_data_Ingestion())

    print(train_data,test_data)