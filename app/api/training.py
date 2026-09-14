"""训练任务 API：创建、查询、启动、停止训练任务。"""
from __future__ import annotations

import json
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..models.database import TrainingJob, TrainingMetric, get_db
from ..schemas.models import (ApiResponse, TrainingJobCreate, TrainingJobOut,
                                TrainingMetricOut)

router = APIRouter(prefix="/api/training", tags=["训练任务"])


@router.get("/jobs", response_model=List[TrainingJobOut])
def list_jobs(status: str = None, limit: int = 20, offset: int = 0, db: Session = Depends(get_db)):
    """获取训练任务列表。"""
    query = db.query(TrainingJob)
    if status:
        query = query.filter(TrainingJob.status == status)
    jobs = query.order_by(TrainingJob.created_at.desc()).offset(offset).limit(limit).all()
    return [j.to_dict() for j in jobs]


@router.get("/jobs/{job_id}", response_model=TrainingJobOut)
def get_job(job_id: int, db: Session = Depends(get_db)):
    """获取训练任务详情。"""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在")
    return job.to_dict()


@router.post("/jobs", response_model=TrainingJobOut)
def create_job(req: TrainingJobCreate, db: Session = Depends(get_db)):
    """创建训练任务（待启动状态）。"""
    job = TrainingJob(
        job_name=req.job_name,
        status="pending",
        model_name=req.model_name,
        dataset_path=req.dataset_path,
        config_json=json.dumps(req.config.model_dump(), ensure_ascii=False),
        output_dir="",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job.to_dict()


@router.post("/jobs/{job_id}/start", response_model=ApiResponse)
def start_job(job_id: int, db: Session = Depends(get_db)):
    """启动训练任务。"""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在")
    if job.status == "running":
        raise HTTPException(status_code=400, detail="任务已在运行中")

    job.status = "running"
    job.started_at = datetime.now()
    db.commit()

    # 实际生产环境中这里会通过 Celery 启动异步训练任务
    # from ..workers.training_worker import run_dpo_training.delay
    # run_dpo_training.delay(job_id)

    return ApiResponse(ok=True, message=f"任务 {job_id} 已启动", data={"job_id": job_id})


@router.post("/jobs/{job_id}/stop", response_model=ApiResponse)
def stop_job(job_id: int, db: Session = Depends(get_db)):
    """停止训练任务。"""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在")
    if job.status != "running":
        raise HTTPException(status_code=400, detail="任务未在运行中")

    job.status = "failed"
    job.error_message = "用户手动停止"
    job.completed_at = datetime.now()
    db.commit()

    return ApiResponse(ok=True, message=f"任务 {job_id} 已停止")


@router.get("/jobs/{job_id}/metrics", response_model=List[TrainingMetricOut])
def get_job_metrics(job_id: int, limit: int = 100, db: Session = Depends(get_db)):
    """获取训练任务的指标曲线。"""
    metrics = db.query(TrainingMetric).filter(
        TrainingMetric.job_id == job_id
    ).order_by(TrainingMetric.step.asc()).limit(limit).all()
    return [
        TrainingMetricOut(
            step=m.step, epoch=m.epoch, loss=m.loss,
            chosen_reward=m.chosen_reward, rejected_reward=m.rejected_reward,
            reward_margin=m.reward_margin, learning_rate=m.learning_rate,
        )
        for m in metrics
    ]


@router.delete("/jobs/{job_id}", response_model=ApiResponse)
def delete_job(job_id: int, db: Session = Depends(get_db)):
    """删除训练任务。"""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在")
    db.delete(job)
    db.commit()
    return ApiResponse(ok=True, message=f"任务 {job_id} 已删除")
