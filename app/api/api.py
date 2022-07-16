from api.endpoints import login, users, ml_function_router
from fastapi import APIRouter

from . import deps

api_router = APIRouter()
api_router.include_router(login.router, tags=["login"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(ml_function_router.router, prefix="/mltask", tags=["mltask"])
