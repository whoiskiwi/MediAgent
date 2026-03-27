from fastapi import FastAPI
from api.v1.router import api_router
from api.v1.auth import router as auth_router

app = FastAPI(title="MediAgent API")

app.include_router(auth_router, prefix="/api/v1")
app.include_router(api_router, prefix="/api/v1")
