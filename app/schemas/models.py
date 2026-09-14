"""Pydantic 数据模型（请求/响应校验）。"""
from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ==================== 训练任务 ====================

class DPOConfigIn(BaseModel):
    """DPO 训练配置输入。"""
    beta: float = Field(default=0.1, description="温度系数")
    learning_rate: float = Field(default=5e-5, description="学习率")
    num_epochs: int = Field(default=3, description="训练轮数")
    batch_size: int = Field(default=4, description="批次大小")
    max_length: int = Field(default=1024, description="最大序列长度")
    warmup_ratio: float = Field(default=0.1, description="预热比例")
    weight_decay: float = Field(default=0.01, description="权重衰减")


class TrainingJobCreate(BaseModel):
    """创建训练任务。"""
    job_name: str = Field(..., description="任务名称")
    model_name: str = Field(default="", description="基座模型名称或路径")
    dataset_path: str = Field(..., description="偏好数据集路径")
    config: DPOConfigIn = Field(default_factory=DPOConfigIn)
    implementation: str = Field(default="custom", description="实现方式：custom / trl")


class TrainingJobOut(BaseModel):
    """训练任务输出。"""
    id: int
    job_name: str
    status: str
    model_name: str
    dataset_path: str
    config: Dict
    output_dir: str
    final_loss: Optional[float] = None
    final_reward_margin: Optional[float] = None
    epochs_trained: int = 0
    steps_trained: int = 0
    error_message: str = ""
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class TrainingMetricOut(BaseModel):
    """训练指标输出。"""
    step: int
    epoch: int
    loss: Optional[float] = None
    chosen_reward: Optional[float] = None
    rejected_reward: Optional[float] = None
    reward_margin: Optional[float] = None
    learning_rate: Optional[float] = None


# ==================== 偏好数据集 ====================

class PreferencePairIn(BaseModel):
    """偏好对输入。"""
    prompt: str
    chosen: str
    rejected: str
    chosen_score: Optional[float] = None
    rejected_score: Optional[float] = None


class DatasetBuildRequest(BaseModel):
    """数据集构造请求。"""
    name: str = Field(..., description="数据集名称")
    strategy: str = Field(default="score_ranking", description="构造策略")
    input_data: List[Dict] = Field(..., description="原始输入数据")
    description: str = Field(default="", description="数据集描述")
    output_format: str = Field(default="json", description="输出格式")


class DatasetOut(BaseModel):
    """数据集输出。"""
    id: int
    name: str
    description: str
    file_path: str
    format: str
    num_samples: int
    build_strategy: str
    avg_prompt_length: float
    avg_chosen_length: float
    avg_rejected_length: float
    created_at: Optional[str] = None


# ==================== 评测 ====================

class EvaluationRequest(BaseModel):
    """评测请求。"""
    eval_name: str = Field(..., description="评测名称")
    model_path: str = Field(..., description="模型路径")
    dataset_path: str = Field(..., description="评测数据集路径")
    metrics: List[str] = Field(default=["loss", "reward_margin", "accuracy"], description="评测指标")


class EvaluationOut(BaseModel):
    """评测结果输出。"""
    id: int
    eval_name: str
    model_path: str
    dataset_path: str
    metrics: Dict
    created_at: Optional[str] = None


# ==================== 损失计算 ====================

class LossComputeRequest(BaseModel):
    """DPO 损失计算请求。"""
    policy_chosen_logps: List[float]
    policy_rejected_logps: List[float]
    ref_chosen_logps: List[float]
    ref_rejected_logps: List[float]
    beta: float = Field(default=0.1, description="温度系数")


class LossComputeResponse(BaseModel):
    """DPO 损失计算响应。"""
    loss: float
    chosen_rewards: float
    rejected_rewards: float
    reward_margin: float
    accuracy: float
    chosen_logratios: float
    rejected_logratios: float


# ==================== 通用响应 ====================

class ApiResponse(BaseModel):
    """通用 API 响应。"""
    ok: bool = True
    message: str = ""
    data: Optional[Dict] = None
