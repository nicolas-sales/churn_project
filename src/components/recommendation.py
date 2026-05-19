import sys

from src.components.prediction import Prediction

from src.utils.logger import logging
from src.utils.exception import CustomException


class Recommandation:

    def __init__(self):
        
        self.predictor = Prediction()

    def generate_recommandations(self,input_data:dict):

        try:

            logging.info("Starting recommandation generation")


            # ==================================================
            # Predict churn probability
            # ==================================================


            prediction_result = self.predictor.predict_churn(input_data)

            churn_probability = prediction_result["churn_probability"]

            logging.info(f"Churn_probability: {churn_probability}" )


            # ==================================================
            # Initialize outputs
            # ==================================================


            context = []
            insights = []
            recommendations = []    


            # ==================================================
            # CUSTOMER CONTEXT
            # ==================================================


            if input_data.get("SeniorCitizen") == 1:
                context.append("Senior customer → may require more support")
        
            if input_data.get("Partner") == "No":
                context.append("No partner → potentially lower engagement")
            
            if input_data.get("Dependents") == "No":
                context.append("No dependents → lower switching cost")
            
            if input_data.get("gender") == "Female":
                context.append("Segment may require tailored engagement")


            # ==================================================
            # ENGAGEMENT ANALYSIS
            # ==================================================


            if input_data.get("tenure", 0) < 12:
                insights.append("Low tenure")
                recommendations.append("Implement onboarding and loyalty program")
            
            if input_data.get("Contract") == "Month-to-month":
                insights.append("Short-term contract")
                recommendations.append("Offer long-term contract")


            # ==================================================
            # SERVICES ANALYSIS
            # ==================================================


            if input_data.get("TechSupport") == "No":
                insights.append("No technical support")
                recommendations.append("Promote technical support services")
            
            if input_data.get("OnlineSecurity") == "No":
                recommendations.append("Offer security add-on")
            
            if input_data.get("OnlineBackup") == "No":
                recommendations.append("Offer backup service")
            
            if input_data.get("DeviceProtection") == "No":
                recommendations.append("Promote device protection plan")


            # ==================================================
            # INTERNET ANALYSIS
            # ==================================================

            if input_data.get("InternetService") == "Fiber optic":
                insights.append("Fiber optic users tend to churn more")
                recommendations.append("Improve perceived value or support")


            # ==================================================
            # PRICING ANALYSIS
            # ==================================================


            if input_data.get("MonthlyCharges", 0) > 80:
                insights.append("High monthly charges")
                recommendations.append("Propose discount or bundle")

            
            # ==================================================
            # PAYMENT ANALYSIS
            # ==================================================


            if input_data.get("PaymentMethod") == "Electronic check":
                insights.append("Risky payment method")
                recommendations.append("Encourage automatic payment")


            # ==================================================
            # ENTERTAINMENT ANALYSIS
            # ==================================================


            if input_data.get("StreamingTV") == "No" and input_data.get("StreamingMovies") == "No":
                insights.append("Low entertainment engagement")
                recommendations.append("Promote entertainment bundle")


            # ==================================================
            # PHONE SERVICE ANALYSIS
            # ==================================================


            if input_data.get("MultipleLines") == "No":
                recommendations.append("Upsell multiple lines")


            if input_data.get("PhoneService") == "No":
                recommendations.append("Offer phone service bundle")


            # ==================================================
            # BILLING ANALYSIS
            # ==================================================


            if input_data.get("PaperlessBilling") == "Yes":
                recommendations.append("Ensure billing transparency and clarity")


            # ==================================================
            # PRIORITIZATION
            # ==================================================


            if churn_probability > 0.7:
                risk_level = "High churn risk"
                recommendations = [f"URGENT: {rec}" for rec in recommendations]
            
            elif churn_probability > 0.4:
                risk_level = "Medium churn risk"
            
            else:
                risk_level = "Low churn risk"
                recommendations = ["Maintain engagement and monitor customer satisfaction"]

            logging.info("Recommandations generated successfully")


            # ==================================================
            # Return results
            # ==================================================


            result = {
                "churn_probability": float(round(churn_probability, 3)),
                "risk_level": risk_level,
                "context": context[:3],
                "key_insights": insights[:5],
                "recommendations": recommendations[:5]
            }

            return result
        
        except Exception as e:
            raise CustomException(e,sys)
        


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


recommander = Recommandation()

result = recommander.generate_recommandations(sample_customer)

print("\nRecommandation result:\n")

print(result)
            
            
                

