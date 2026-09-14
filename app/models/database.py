"""数据库模型：训练任务、偏好数据集、评测结果。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer,
                        String, Text, create_engine)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from ..config import settings


class Base(DeclarativeBase):
    pass


class TrainingJob(Base):
    """训练任务表。"""
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_name = Column(String(256), nullable=False)
    status = Column(String(32), default="pending")  # pending/running/completed/failed
    model_name = Column(String(256), default="")
    dataset_path = Column(String(512), default="")
    config_json = Column(Text, default="{}")
    output_dir = Column(String(512), default="")
    final_loss = Column(Float, nullable=True)
    final_reward_margin = Column(Float, nullable=True)
    epochs_trained = Column(Integer, default=0)
    steps_trained = Column(Integer, default=0)
    error_message = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.now)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    metrics = relationship("TrainingMetric", back_populates="job", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "job_name": self.job_name,
            "status": self.status,
            "model_name": self.model_name,
            "dataset_path": self.dataset_path,
            "config": json.loads(self.config_json) if self.config_json else {},
            "output_dir": self.output_dir,
            "final_loss": self.final_loss,
            "final_reward_margin": self.final_reward_margin,
            "epochs_trained": self.epochs_trained,
            "steps_trained": self.steps_trained,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class TrainingMetric(Base):
    """训练指标表（记录每个 step 的损失和奖励）。"""
    __tablename__ = "training_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("training_jobs.id"), nullable=False)
    step = Column(Integer, nullable=False)
    epoch = Column(Integer, default=0)
    loss = Column(Float, nullable=True)
    chosen_reward = Column(Float, nullable=True)
    rejected_reward = Column(Float, nullable=True)
    reward_margin = Column(Float, nullable=True)
    learning_rate = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

    job = relationship("TrainingJob", back_populates="metrics")


class PreferenceDataset(Base):
    """偏好数据集表。"""
    __tablename__ = "preference_datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(256), nullable=False)
    description = Column(Text, default="")
    file_path = Column(String(512), default="")
    format = Column(String(32), default="json")  # json / jsonl / trl
    num_samples = Column(Integer, default=0)
    build_strategy = Column(String(64), default="")
    avg_prompt_length = Column(Float, default=0)
    avg_chosen_length = Column(Float, default=0)
    avg_rejected_length = Column(Float, default=0)
    created_at = Column(DateTime, default=datetime.now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "file_path": self.file_path,
            "format": self.format,
            "num_samples": self.num_samples,
            "build_strategy": self.build_strategy,
            "avg_prompt_length": self.avg_prompt_length,
            "avg_chosen_length": self.avg_chosen_length,
            "avg_rejected_length": self.avg_rejected_length,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class EvaluationResult(Base):
    """评测结果表。"""
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("training_jobs.id"), nullable=True)
    eval_name = Column(String(256), nullable=False)
    model_path = Column(String(512), default="")
    dataset_path = Column(String(512), default="")
    metrics_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "eval_name": self.eval_name,
            "model_path": self.model_path,
            "dataset_path": self.dataset_path,
            "metrics": json.loads(self.metrics_json) if self.metrics_json else {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# 数据库初始化
engine = create_engine(settings.DATABASE_URL, echo=settings.DEBUG, connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """初始化数据库表。"""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """获取数据库会话。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
