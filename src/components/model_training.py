import os
import sys

import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score)

from src.utils.logger import logging
from src.utils.exception import CustomException

@dataclass

class ModelTrainingConfig:

    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:

    def __init__(self):
        
        self.model_trainer_config = ModelTrainingConfig()

    def evaluate_model(self,X_train,y_train,X_test,y_test,model):

        # Train model
        model.fit(X_train,y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Probabilities
        y_train_proba = model.predict_proba(X_train)[:,1]
        y_test_proba = model.predict_proba(X_test)[:,1]

        # Metrics
        metrics = {

            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "train_precision": precision_score(y_train, y_train_pred),
            "train_recall":  recall_score(y_train, y_train_pred),
            "train_f1": f1_score(y_train, y_train_pred),
            "train_roc_auc": roc_auc_score(y_train, y_train_proba),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "test_precision": precision_score(y_test, y_test_pred),
            "test_recall":  recall_score(y_test, y_test_pred),
            "test_f1": f1_score(y_test, y_test_pred),
            "test_roc_auc": roc_auc_score(y_test, y_test_proba)
        }

        return metrics
    
    def initiate_model_trainer(self,train_array_path,test_array_path):

        try:

            logging.info("Loading train and testarrays")

            train_arr = np.load(train_array_path)

            test_arr = np.load(test_array_path)

            logging.info("Train and Test arrays loaded successfully")

            # Split train data
            X_train = train_arr[:,:-1]
            y_train = train_arr[:,-1]

            # Split test data
            X_test = test_arr[:,:-1]
            y_test = test_arr[:,-1]

            logging.info("Input and target features separated")

            # Models
            models= {
                "XgBoost":XGBClassifier(eval_metric="logloss"),
                "GradientBoosting":GradientBoostingClassifier(),
                "AdaBoost":AdaBoostClassifier(),
                "DecisionTree":DecisionTreeClassifier(class_weight="balanced"),
                "RandomForest":RandomForestClassifier(class_weight="balanced"),
                "LogisticRegression":LogisticRegression(class_weight="balanced",max_iter=1000)
            }

            # MLflow experiment
            # MLflow tracking
            mlflow.set_tracking_uri("file:./mlruns")

            # MLflow experiment
            mlflow.set_experiment("customer_churn_prediction")

            best_model_score = 0
            best_model_name = None
            best_model = None
            model_report = {}

            # Train and evaluate each model
            for model_name, model in models.items():

                with mlflow.start_run(run_name=model_name):

                    logging.info(f"Training started for {model_name}")

                    # Evaluate model
                    metrics = self.evaluate_model(X_train,y_train,X_test,y_test,model)

                    model_report[model_name] = metrics

                    test_recall = (metrics["test_recall"])

                    logging.info(f"{model_name} rcall: {test_recall}")

                    # Log model name
                    mlflow.log_param("model_name",model_name)

                    # Log all parameters
                    mlflow.log_params(model.get_params())

                    # Log metrics
                    mlflow.log_metric("test_precision",metrics["test_precision"])

                    mlflow.log_metric("test_recall",metrics["test_recall"])

                    mlflow.log_metric("test_f1",metrics["test_f1"])

                    mlflow.log_metric("test_roc_auc",metrics["test_roc_auc"])

                    # Log model
                    mlflow.sklearn.log_model(sk_model=model,artifact_path="model")

                    logging.info(f"{model_name} logged to MLflow")

                    # Select best model
                    if test_recall > best_model_score:

                        best_model_score = test_recall

                        best_model_name = model_name

                        best_model = model

            logging.info(f"Best model found: {best_model_name}")

            # Save best model locally
            joblib.dump(best_model,self.model_trainer_config.trained_model_file_path)

            logging.info("Best model saved successfully")

            # Print metrics
            for model_name,metrics in model_report.items():

                print("="*50)
                print(f"\nMODEL: {model_name}")
                print("\nTEST METRICS")
                print(f"Accuracy: " f"{metrics['test_accuracy']:.4f}")
                print(f"Precision: " f"{metrics['test_precision']:.4f}")
                print(f"Recall: " f"{metrics['test_recall']:.4f}")
                print(f"F1 Score: " f"{metrics['test_f1']:.4f}")
                print(f"ROC AUC: " f"{metrics['test_roc_auc']:.4f}")

            print("\n")
            print(f"BEST MODEL: {best_model_name}")
            print(f"BEST RECALL: {best_model_score:.4f}")

            return (
                best_model_name,
                best_model_score,
                model_report
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":

    from src.components.data_ingestion import DataIngestion
    from src.components.data_preprocessing import DataTransformation

    # Data Ingestion
    ingestion = DataIngestion()

    raw_data_path = ingestion.initiate_data_Ingestion()

    # Data transformation
    transformation = DataTransformation()

    train_path,test_path,preprocessor_path = transformation.initiate_data_transformation(raw_data_path)

    # Model training
    trainer = ModelTrainer()

    best_model_name,best_model_score,model_report = trainer.initiate_model_trainer(train_path,test_path)

    print("\nTraining completed successfully")
