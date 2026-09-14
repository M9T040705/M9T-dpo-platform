"""配置中心：统一读取环境变量。"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "DPO FineTune Platform")
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"

    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/data/dpo_platform.db")

    BASE_MODEL_PATH: str = os.getenv("BASE_MODEL_PATH", str(BASE_DIR / "data/models/base"))
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", str(BASE_DIR / "data/outputs")))
    MAX_LENGTH: int = int(os.getenv("MAX_LENGTH", "1024"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "4"))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "5e-5"))
    NUM_EPOCHS: int = int(os.getenv("NUM_EPOCHS", "3"))
    BETA: float = float(os.getenv("BETA", "0.1"))

    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

    JWT_SECRET: str = os.getenv("JWT_SECRET", "change-me-in-production")
    JWT_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))

    def ensure_dirs(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (BASE_DIR / "data/datasets").mkdir(parents=True, exist_ok=True)
        (BASE_DIR / "data/preferences").mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
