from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataTransformation
from src.components.model_training import ModelTrainer

from src.utils.logger import logging
from src.utils.exception import CustomException


class TrainingPipeline:

    def __init__(self):
        
        self.data_ingestion = DataIngestion()

        self.data_transformation = DataTransformation()

        self.model_trainer = ModelTrainer()

    def run_pipeline(self):

        try:

            logging.info("Traing pipeline started")


            # ==================================================
            # DATA INGESTION
            # ==================================================


            logging.info("Started data ingestion")

            raw_data_path = self.data_ingestion.initiate_data_Ingestion()

            logging.info("Data ingestion completed")


            # ==================================================
            # DATA TRANSFORMATION
            # ==================================================


            logging.info("Started data transformation")

            train_path,test_path,preprocessor_path = self.data_transformation.initiate_data_transformation(raw_data_path)

            logging.info("Data transformation completed")


            # ==================================================
            # MODEL TRAINING
            # ==================================================


            logging.info("Started model training")

            best_model_name,best_model_score,model_report = self.model_trainer.initiate_model_trainer(train_path,test_path)

            logging.info("model training completed")

            logging.info(f"Best model: {best_model_name}")

            logging.info(f"Best model score: {best_model_score}")


            # ==================================================
            # FINAL RESULTS
            # ==================================================


            result = {
                "best_model_name":  best_model_name,
                "best_model_score":  best_model_score,
                "model_report": model_report
            }

            logging.info("Training Pipeline completed successfully")

            return result
        
        except Exception as e:
            raise CustomException(e,sys)
        


if __name__ =="__main__":

    pipeline = TrainingPipeline()

    result = pipeline.run_pipeline()

    print("\nTraing Pipeline Result\n")

    print(result)