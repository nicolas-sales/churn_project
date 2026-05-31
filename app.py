import streamlit as st

from src.agent.agent import ChurnAgent

from src.pipeline.prediction_pipeline import PredictionPipeline

# Page config

st.set_page_config(
    page_title="Customer Churn AI Assitant",
    page_icon="📉",
    layout="wide"
)

# Title

st.title("📉 Customer churn assistant")

st.markdown(
    """
    Predict chur risk, explain predictions,
    generate retention recommendations,
    and validate actions through simulations"""
)

# Customer form

with st.form("customer_form"):

    col1,col2,col3,col4 = st.columns(4)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure",
            min_value=0,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

    with col2:

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No"]
        )

    with col3:

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No"]
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=500.0
    )

    submitted = st.form_submit_button(
        "Analyze Customer"
    )

# Run analysis

if submitted:

    customer_data = {

        "gender": gender,

        "SeniorCitizen": senior,

        "Partner": partner,

        "Dependents": dependents,

        "tenure": tenure,

        "PhoneService": phone_service,

        "MultipleLines": multiple_lines,

        "InternetService": internet_service,

        "OnlineSecurity": online_security,

        "OnlineBackup": online_backup,

        "DeviceProtection": device_protection,

        "TechSupport": tech_support,

        "StreamingTV": streaming_tv,

        "StreamingMovies": streaming_movies,

        "Contract": contract,

        "PaperlessBilling": paperless,

        "PaymentMethod": payment_method,

        "MonthlyCharges": monthly_charges,

        "TotalCharges": total_charges
    }

    with st.spinner("Analyzing customer..."):

        try:

            pipeline = PredictionPipeline()

            pipeline_result = pipeline.run_pipeline(customer_data)

            agent = ChurnAgent()

            result = agent.analyse_customer(customer_data)

            final_answer = result["messages"][-1].content

            st.success("Analysis completed")

            # KPI Cards

            prediction = pipeline_result["prediction"]

            col1,col2 = st.columns(2)

            with col1:
                st.metric("Churn Probability", f"{prediction['churn_probability']*100:.2f}%")

            with col2:
                st.metric("Risk Level", prediction["risk_level"])

            # Churn Gauge

            import plotly.graph_objects as go

            probability = (
                prediction["churn_probability"] * 100
            )

            fig = go.Figure(

                go.Indicator(

                    mode="gauge+number",

                    value=probability,

                    title={"text":"Churn Probability"},

                    gauge={
                        "axis":{"range":[0,100]}
                    }
                )
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

            # SHAP Table

            import pandas as pd

            shap_df = pd.DataFrame(pipeline_result["shap_explanation"])

            #st.subheader("Top SHAP Drivers")

            #st.dataframe(shap_df,width="stretch")

            # SHAP Bar Chart

            import plotly.express as px

            fig = px.bar(
                shap_df,
                x="shap_value",
                y="feature",
                orientation="h",
                color="impact",
                title="Top SHAP Drivers"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

            # Recommendations

            recommendation_result = pipeline_result["recommendations"]

            st.subheader("Recommendations")

            for recommendation in recommendation_result["recommendations"]:
                st.info(recommendation)

            # Agent Summary

            st.subheader("AI Executive Summary")

            # st.markdown("## Analysis Result")

            st.markdown(final_answer)

        except Exception as e:

            st.error(f"Error: {str(e)}")


