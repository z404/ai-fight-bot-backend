from fastapi import FastAPI
import uvicorn

app = FastAPI(
    title="AI Fight Bot",
)


@app.get("/")
async def root():
    return {"message": "AI Fight Bot"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")
