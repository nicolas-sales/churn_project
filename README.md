# Customer Churn Prediction and Agentic Recommendation System

## Overview

This project focuses on predicting customer churn in a telecommunications dataset and building a decision-support agent that provides actionable business recommendations.

The system goes beyond standard machine learning by combining:

- A predictive churn model
- Scenario simulation (what-if analysis)
- A rule-based agent that generates business recommendations based on customer profiles

The goal is to move from pure prediction to actionable insights.

## Objectives

- Predict the probability of customer churn
- Identify key drivers of churn
- Simulate the impact of changes in customer attributes
- Provide business-oriented recommendations through an intelligent agent

## Dataset

The dataset used is the Telco Customer Churn dataset, which includes:

- Customer demographics (gender, SeniorCitizen, Partner, Dependents)
- Account information (tenure, Contract, PaymentMethod, PaperlessBilling)
- Services subscribed (InternetService, TechSupport, StreamingTV, etc.)
- Billing information (MonthlyCharges, TotalCharges)
- Target variable: Churn

## Methodology

1. Data Preparation

Converted TotalCharges to numeric
Removed missing values
Encoded categorical variables using one-hot encoding
Split data into train and test sets
Applied StandardScaler for feature scaling

2. Modeling

Several models were tested:

Logistic Regression
Random Forest
Gradient Boosting
XGBoost

Final model selected:

Logistic Regression with class_weight="balanced"

Reason:

Better recall on churn class
Stable performance
Interpretable

3. Evaluation

Key metrics:

Recall (churn class) prioritized
ROC-AUC
Precision / Recall trade-off

Example result:

Recall (churn): ~0.79
ROC-AUC: ~0.83

This ensures most churners are correctly identified.

4. Model Persistence

The following components are saved:

Trained model (model.pkl)
Scaler (scaler.pkl)
Feature columns (features.pkl)

This allows consistent predictions in production.

## Simulation (What-if Analysis)

A simulation function allows testing how changes in customer attributes affect churn probability.

Example:

Increasing tenure
Changing contract type
Modifying pricing

This enables business-oriented analysis such as:

"What happens if we offer a long-term contract?"
"How does pricing impact churn risk?"

## Agent Design

The agent is designed to provide structured recommendations based on:

Customer context (non-actionable features)
Key churn drivers (insights)
Business actions (recommendations)

## Key Principles

Uses raw business features (not encoded variables)
Separates diagnosis from decision
Keeps recommendations consistent
Adjusts urgency based on churn probability

## Recommendation Logic

The agent:

Identifies risk factors such as:
Low tenure
Month-to-month contract
Lack of support services
High monthly charges

Generates actions such as:
Offer long-term contracts
Provide discounts or bundles
Promote support services
Improve onboarding

Adjusts urgency:
Churn Probability	Behavior
> 0.7 : Urgent actions
0.4 – 0.7 : Preventive actions
< 0.4 : Monitoring only

## Future Improvements
Integrate a Large Language Model for natural language recommendations
Build a Streamlit or web interface
Add model explainability (e.g. SHAP)
Deploy as an API (FastAPI)