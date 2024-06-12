from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.api.endpoints.predictions import app, router

if __name__ == "__main__":
    app.include_router(router, prefix="/api")
    uvicorn.run(app, host="0.0.0.0", port=8000)