from fastapi import FastAPI

app = FastAPI()

# Import endpoints
from app.api.endpoints import predictions

app.include_router(predictions.router)