from fastapi import FastAPI

from src.api.routes import router

app = FastAPI(title="Customer Churn API")

app.include_router(router)