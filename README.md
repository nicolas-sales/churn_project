# Customer Churn Prediction and AI Recommendation Agent

## Overview

This project focuses on predicting customer churn in a telecommunications company and building an AI-powered decision support agent capable of generating actionable retention strategies.

The system combines:

- Machine learning for churn prediction
- SHAP explainability for identifying churn drivers
- Counterfactual simulations for what-if analysis
- A LangChain agent orchestrating predictions, explanations, simulations, and business recommendations

The objective is to move beyond static prediction and provide interpretable, business-oriented retention insights.

---

## Objectives

The project aims to:

- Predict customer churn probability
- Explain the main drivers behind each prediction
- Simulate the impact of business actions on churn risk
- Generate prioritized retention recommendations
- Build an AI agent capable of reasoning over customer profiles

---

## Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains:

### Customer Demographics

- Gender
- SeniorCitizen
- Partner
- Dependents

### Account Information

- Tenure
- Contract type
- Payment method
- Paperless billing

### Subscribed Services

- Internet service
- Online security
- Online backup
- Device protection
- Technical support
- Streaming services

### Billing Information

- Monthly charges
- Total charges

### Target Variable

- Churn

---

## Project Architecture

The system is composed of four main layers:

### 1. Predictive Machine Learning Model

A supervised classification model predicts churn probability.

### 2. Explainability Layer

SHAP values are used to explain the main drivers behind churn predictions.

### 3. Simulation Engine

Counterfactual simulations estimate how changes in customer features impact churn probability.

### 4. AI Agent Layer

A LangChain agent orchestrates:

- churn prediction
- SHAP explanation
- business recommendations
- scenario simulations

---

## Data Preparation

The preprocessing pipeline includes:

### Data Cleaning

- Converted `TotalCharges` to numeric
- Removed missing values

### Feature Engineering

- Separation of numerical and categorical variables
- One-hot encoding for categorical features
- Standard scaling for numerical features

### Train/Test Split

- Stratified train-test split

### Preprocessing Pipeline

A `ColumnTransformer` pipeline was used to ensure consistent preprocessing during both training and inference.

---

## Modeling

Several machine learning models were evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting
- AdaBoost
- XGBoost
- Decision Tree

The project includes:

- Baseline model comparison
- Hyperparameter tuning using `GridSearchCV`
- Experiment tracking with MLflow
- Automatic best model selection based on recall score

The final selected model was:

### Tuned Random Forest Classifier

The selected model was obtained after hyperparameter optimization using `GridSearchCV`.

### Reasons for Selection

- Highest recall score among all evaluated models
- Better identification of customers likely to churn
- Improved performance after hyperparameter tuning
- Strong overall classification performance
- Suitable for business-oriented churn prevention use cases

---

## Experiment Tracking with MLflow

MLflow was integrated into the training pipeline to track:

- Model parameters
- Hyperparameters
- Evaluation metrics
- Training runs
- Serialized model artifacts

This enables:

- Reproducibility
- Model comparison
- Experiment management
- Better monitoring of tuning results

Tracked experiments include:

- Baseline model evaluation
- Tuned model evaluation using `GridSearchCV`

---

## Model Evaluation

The evaluation focused primarily on identifying churners correctly.

### Main Metrics

- Recall
- Precision
- F1-score
- ROC-AUC

### Final Performance (Tuned Random Forest)

- Recall (churn class): approximately 0.81
- ROC-AUC: approximately 0.83
- Accuracy: approximately 0.73

The model prioritizes recall to reduce false negatives and better identify customers at risk of leaving.

---

## Hyperparameter Optimization

Hyperparameter tuning was performed using `GridSearchCV` on selected models:

- Logistic Regression
- Random Forest
- XGBoost

The optimization objective was:

### Recall Score

This choice reflects the business objective of minimizing missed churners.

Example tuned parameters for Random Forest included:

- `n_estimators`
- `max_depth`
- `max_features`
- `min_samples_split`

The tuned Random Forest model achieved the best recall performance across all experiments.

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) was integrated to explain individual churn predictions.

The explainability layer identifies:

- which features increase churn risk
- which features reduce churn risk
- the relative importance of each feature for a specific customer

### Example Churn Drivers

- Month-to-month contract
- Low tenure
- High monthly charges
- Lack of technical support

This allows the system to generate more reliable and interpretable recommendations.

---

## Counterfactual Simulations

The simulation engine performs what-if analysis by modifying customer attributes and recomputing churn probability.

### Examples

- Switching from month-to-month to one-year contract
- Adding technical support
- Reducing monthly charges

This enables quantitative evaluation of retention strategies.

### Example Result

- Churn probability reduced from 66.9% to 47% after changing contract type to one year.

---

## AI Agent Design

The project includes a LangChain-based AI agent capable of orchestrating multiple tools.

### Agent Responsibilities

- Predict churn probability
- Explain churn drivers using SHAP
- Generate business recommendations
- Simulate retention actions
- Rank actions by expected impact

### Tools Used

- `predict_churn`
- `explain_prediction`
- `recommend_action`
- `simulate_change`

### Agent Workflow

The agent follows the workflow:

1. Predict churn probability
2. Explain prediction drivers using SHAP
3. Generate retention recommendations
4. Simulate high-impact actions
5. Produce a business-oriented synthesis

---

## Recommendation Logic

The recommendation engine combines:

- model predictions
- SHAP explanations
- business rules
- simulation results

### Typical Recommendations

- Offer long-term contracts
- Improve onboarding experience
- Promote technical support services
- Offer bundled services
- Improve customer engagement

### Risk Prioritization

| Churn Probability | Risk Level |
|---|---|
| > 0.70 | High Risk |
| 0.40 – 0.70 | Medium Risk |
| < 0.40 | Low Risk |

---

## Technologies Used

### Machine Learning

- Scikit-learn
- XGBoost

### Explainability

- SHAP

### AI Agent Framework

- LangChain

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

---

## Future Improvements

Potential future improvements include:

### Application Layer

- Streamlit interface
- FastAPI deployment
- Interactive dashboards

### AI Improvements

- Automated simulation of top SHAP features
- Multi-scenario optimization
- More advanced recommendation ranking

### MLOps

- Dockerization
- CI/CD pipeline
- Cloud deployment
- Model monitoring

---

## Conclusion

This project combines machine learning, explainable AI, counterfactual simulations, and LLM-based reasoning to create an intelligent customer retention decision-support system.

Rather than only predicting churn, the system helps identify why customers are likely to leave and which actions are most effective for reducing churn risk.