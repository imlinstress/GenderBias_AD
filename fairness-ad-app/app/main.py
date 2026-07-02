"""main.py — FastAPI app entry point. Run with: uvicorn app.main:app"""

import os, sys
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.routes import router as api_router
from app.model_service import load_ad_model
from app.database import init_db
import app.routes as routes_module

app = FastAPI(title="Fair AD Predictor", version="2.0.0")

static_dir = os.path.join(os.path.dirname(__file__), 'static')
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

app.include_router(api_router, prefix="/api")


@app.on_event("startup")
async def startup():
    init_db()
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
    if os.path.exists(os.path.join(model_dir, 'ad_model.ckpt.index')):
        svc = load_ad_model(model_dir)
        routes_module.model_service = svc
        print(f"Model loaded from {model_dir}")
    else:
        print("WARNING: No model artifacts found. Run scripts/extract_ad_model.py first.")


@app.get("/")
async def index():
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    return FileResponse(os.path.join(templates_dir, 'index.html'))
