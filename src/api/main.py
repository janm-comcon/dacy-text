from fastapi import FastAPI
from .routes import router

app = FastAPI(title="stdtext API", version="0.1.0")
app.include_router(router)

def run():
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
