from fastapi import FastAPI

from src.backend.api.routes import router

app = FastAPI(
    title="ML Disease Prediction App",
    version="1.0.0",
    description="Multi-disease Prediction backend"
)

app.include_router(router, prefix = "/api")

