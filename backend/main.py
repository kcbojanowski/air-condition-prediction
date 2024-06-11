from fastapi import FastAPI
import uvicorn
from app.api.endpoints.predictions import router

if __name__ == "__main__":
    app = FastAPI()

    app.include_router(router, prefix="/api")
