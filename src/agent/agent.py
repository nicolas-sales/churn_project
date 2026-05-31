import sys 
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from src.tools.tools import tools

from src.utils.logger import logging
from src.utils.exception import CustomException


# ==================================================
# Load environment variables and management agent 
# ==================================================


load_dotenv()

class ChurnAgent:

    def __init__(self):

        # Initialize agent
        logging.info("Initialize Churn Agent")

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
            )
        
        logging.info("LM initialized successfully")

        # Create agent

        self.agent = create_agent(
            self.llm,
            tools=tools,
            system_prompt="""
        You are a customer churn analysis agent.

        Your responsibilities:
        - Analyze customer profiles
        - Predict churn probability
        - Explain the main drivers of churn using SHAP values
        - Provide business recommendations
        - Prioritize actions based on expected impact
        - Validate important actions through simulations

        Rules:
        - Always use full_customer_analysis first
        - Analyze SHAP explanations to identify the most important churn drivers
        - Focus on actionable drivers whenever possible
        - After recommendations are generated, validate the most relevant recommendations using simulate_change
        - Run up to 5 simulations when actionable recommendations are available
        - Include simulation results in the final answer
        - Use SHAP explanations as guidance for decision making
        - Prioritize simulations based on SHAP importance and business relevance
        - Do not simulate non-actionable features unless absolutely necessary
        - Be concise, professional, and data-driven
        - Rank recommendations by expected impact on churn reduction

        Important:
        - SHAP identifies important drivers, not necessarily actionable drivers
        - SHAP values explain model behavior, not causal effects
        - Simulation results are based on statistical patterns, not causal relationships
        - If a result seems counterintuitive, explain that the model may reflect correlations in the dataset
        - Do not blindly trust simulation results
        - Do not present SHAP as causal proof

        When a simulation result is ineffective or counterintuitive:
        - Explicitly state that the action is not recommended
        - Suggest alternative actions
        - Never mention internal tool errors to the user
        - If a simulation fails, continue the analysis gracefully
        """
        )

        logging.info("Agent created successfully")


    # ==================================================
    # Analyze customer
    # ==================================================    

    def analyse_customer(self,customer_data:dict):

        try:
             logging.info("Started customer analysis")

             result = self.agent.invoke({
                 "messages":[
                     {
                         "role":"user",
                         "content": f"Analyse this customer: {customer_data}"
                     }
                 ]
             })

             return result

             logging.info("Customer analysis completed successfully")

        except Exception as e:
            raise CustomException(e,sys)
        


if __name__ == "__main__":

    sample_data = {
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

    churn_agent = ChurnAgent()

    result = churn_agent.analyse_customer(customer_data=sample_data)

    print("\nAgent Response")

    print(result)