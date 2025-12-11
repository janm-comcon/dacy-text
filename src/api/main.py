import sys
from pathlib import Path

from fastapi import FastAPI

# Ensure project root on sys.path so `python src/api/main.py` works
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.routes import router

app = FastAPI(title="stdtext API", version="0.1.0")
app.include_router(router)

def run():
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    run()
