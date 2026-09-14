"""DPO 微调一站式平台 - FastAPI 主应用。"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .models.database import init_db
from .api.training import router as training_router
from .api.dataset import router as dataset_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="DPO FineTune Platform",
    description="DPO 直接偏好优化 + 领域微调一站式平台",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(training_router)
app.include_router(dataset_router)


@app.get("/")
def root():
    return {
        "app": "DPO FineTune Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "api": {
            "training": "/api/training/jobs",
            "dataset": "/api/dataset/datasets",
        }
    }


@app.get("/healthz")
def healthz():
    return {"status": "ok", "app": settings.APP_NAME, "env": settings.APP_ENV}
