"""偏好数据集 API：构造、管理、导出偏好对数据。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..config import settings
from ..core.dpo_trainer import PreferencePair
from ..core.preference_builder import (BuildConfig, ModelCompareBuilder,
                                         PreferenceDatasetExporter,
                                         RuleBasedBuilder, ScoreRankingBuilder)
from ..models.database import PreferenceDataset, get_db
from ..schemas.models import (ApiResponse, DatasetBuildRequest, DatasetOut,
                                LossComputeRequest, LossComputeResponse)

router = APIRouter(prefix="/api/dataset", tags=["偏好数据集"])


@router.get("/datasets", response_model=List[DatasetOut])
def list_datasets(limit: int = 20, offset: int = 0, db: Session = Depends(get_db)):
    """获取数据集列表。"""
    datasets = db.query(PreferenceDataset).order_by(
        PreferenceDataset.created_at.desc()
    ).offset(offset).limit(limit).all()
    return [d.to_dict() for d in datasets]


@router.get("/datasets/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """获取数据集详情。"""
    ds = db.query(PreferenceDataset).filter(PreferenceDataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")
    return ds.to_dict()


@router.post("/datasets/build", response_model=DatasetOut)
def build_dataset(req: DatasetBuildRequest, db: Session = Depends(get_db)):
    """构造偏好数据集。"""
    config = BuildConfig(strategy=req.strategy, output_format=req.output_format)

    # 根据策略选择构造器
    if req.strategy == "score_ranking":
        builder = ScoreRankingBuilder(config)
    elif req.strategy == "rule_based":
        builder = RuleBasedBuilder(config)
    elif req.strategy == "model_compare":
        builder = ModelCompareBuilder(config)
    else:
        raise HTTPException(status_code=400, detail=f"不支持的构造策略：{req.strategy}")

    pairs = builder.build(req.input_data)
    if not pairs:
        raise HTTPException(status_code=400, detail="构造失败：未生成有效偏好对")

    # 保存到文件
    output_dir = Path(settings.OUTPUT_DIR) / "datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{req.name}_{len(pairs)}.{req.output_format}"

    if req.output_format == "jsonl":
        PreferenceDatasetExporter.to_jsonl(pairs, file_path)
    elif req.output_format == "trl":
        PreferenceDatasetExporter.to_trl_format(pairs, file_path)
    else:
        PreferenceDatasetExporter.to_json(pairs, file_path)

    # 统计信息
    stats = PreferenceDatasetExporter.stats(pairs)

    # 保存到数据库
    ds = PreferenceDataset(
        name=req.name,
        description=req.description,
        file_path=str(file_path),
        format=req.output_format,
        num_samples=len(pairs),
        build_strategy=req.strategy,
        avg_prompt_length=stats["avg_prompt_length"],
        avg_chosen_length=stats["avg_chosen_length"],
        avg_rejected_length=stats["avg_rejected_length"],
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)

    return ds.to_dict()


@router.post("/datasets/{dataset_id}/export/{fmt}", response_model=ApiResponse)
def export_dataset(dataset_id: int, fmt: str, db: Session = Depends(get_db)):
    """导出数据集为指定格式。"""
    ds = db.query(PreferenceDataset).filter(PreferenceDataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

    # 读取原始数据
    try:
        with open(ds.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pairs = [PreferencePair(**item) for item in data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取数据集失败：{e}")

    # 导出
    output_path = Path(ds.file_path).with_suffix(f".{fmt}")
    if fmt == "jsonl":
        PreferenceDatasetExporter.to_jsonl(pairs, output_path)
    elif fmt == "trl":
        PreferenceDatasetExporter.to_trl_format(pairs, output_path)
    else:
        PreferenceDatasetExporter.to_json(pairs, output_path)

    return ApiResponse(ok=True, message=f"已导出为 {fmt} 格式", data={"output_path": str(output_path)})


@router.delete("/datasets/{dataset_id}", response_model=ApiResponse)
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """删除数据集。"""
    ds = db.query(PreferenceDataset).filter(PreferenceDataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")
    db.delete(ds)
    db.commit()
    return ApiResponse(ok=True, message=f"数据集 {dataset_id} 已删除")


@router.post("/loss/compute", response_model=LossComputeResponse)
def compute_dpo_loss(req: LossComputeRequest):
    """计算 DPO 损失（纯数学实现，用于理解原理和快速验证）。"""
    import numpy as np
    from ..core.dpo_trainer import DPOLossCalculator

    calculator = DPOLossCalculator(beta=req.beta)
    loss, metrics = calculator.compute_loss(
        policy_chosen_logps=np.array(req.policy_chosen_logps),
        policy_rejected_logps=np.array(req.policy_rejected_logps),
        ref_chosen_logps=np.array(req.ref_chosen_logps),
        ref_rejected_logps=np.array(req.ref_rejected_logps),
    )
    return LossComputeResponse(
        loss=metrics["loss"],
        chosen_rewards=metrics["chosen_rewards"],
        rejected_rewards=metrics["rejected_rewards"],
        reward_margin=metrics["reward_margin"],
        accuracy=metrics["accuracy"],
        chosen_logratios=metrics["chosen_logratios"],
        rejected_logratios=metrics["rejected_logratios"],
    )
