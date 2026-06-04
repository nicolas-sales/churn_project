from fastapi import APIRouter

from src.api.schemas import CustomerInput

from src.pipeline.prediction_pipeline import PredictionPipeline

from src.agent.agent import ChurnAgent

router = APIRouter()

# pipeline = PredictionPipeline()

# agent = ChurnAgent()

# Endpoints

@router.get("/")
def root():
    return {
        "message" : "Customer Churn API is running"
    }

@router.get("/health")
def health_check():
    return {
        "status" : "healthy"
    }

@router.post("/pipeline")
def run_pipeline(customer: CustomerInput):

    pipeline = PredictionPipeline()

    result = pipeline.run_pipeline(customer.model_dump()) # model.dump pour transformer un dictionnaire pydantic en dictionnaire python

    return result

@router.post("/analyse")
def analyse_customer(customer: CustomerInput):

    agent = ChurnAgent()

    result = agent.analyse_customer(customer.model_dump())

    return {"response": result["messages"][-1].content}