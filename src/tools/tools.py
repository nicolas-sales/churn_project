from langchain_core.tools import StructuredTool
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.components.simulation import Simulation


# ==================================================
# FULL ANALYSIS TOOL
# ==================================================


prediction_pipeline = PredictionPipeline()

simulation = Simulation()


# ==================================================
# PREDICTION TOOL
# ==================================================


def full_analysis_tool_func(input_data: dict):

    return prediction_pipeline.run_pipeline(input_data)

full_analysis_tool = StructuredTool.from_function(
    func=full_analysis_tool_func,
    name="full_customer_analysis",
    description="""
    Run a complete churn analysis:
    - churn prediction
    - SHAP explanation
    - recommendations
    """
)


# ==================================================
# SIMULATION TOOL
# ==================================================


def simulation_tool_func(input_data:dict, feature:str, new_value):

    return simulation.simulate_change(
        input_data=input_data,
        feature=feature,
        new_value=new_value
    )


simulation_tool = StructuredTool.from_function(
    func=simulation_tool_func,
    name="simulate_change",
    description="""
    Simulate the impact of changing
    customer features on churn probability
    """
)

tools = [full_analysis_tool,simulation_tool]