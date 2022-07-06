from fastapi import FastAPI
import uvicorn

from api.api import api_router
from core.config import settings


app = FastAPI(
    title="AI Fight Bot",
)


@app.get("/hello")
async def root():
    return {"message": "AI Fight Bot"}


app.include_router(api_router, prefix=settings.API_V1_STR)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")
