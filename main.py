from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.v1.router import api_router
from api.v1.auth import router as auth_router
from db.backend import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="MediAgent API", lifespan=lifespan)

app.include_router(auth_router, prefix="/api/v1")
app.include_router(api_router, prefix="/api/v1")
