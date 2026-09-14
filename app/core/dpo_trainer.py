"""DPO (Direct Preference Optimization) 直接偏好优化算法实现。

核心原理：
- 不需要训练奖励模型，直接用偏好对数据（chosen/rejected）优化策略模型
- 损失函数：L_DPO = -E[log σ(β (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
- 其中 y_w 是 chosen（优选），y_l 是 rejected（劣选），β 是温度系数

本模块提供：
1. 从零实现的 DPO Trainer（纯 PyTorch，便于理解原理）
2. 基于 TRL 库的 DPOTrainer（生产级实现）
3. 两种实现的对比评测
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DPOConfig:
    """DPO 训练配置。"""
    beta: float = 0.1                    # 温度系数（控制与参考模型的偏离程度）
    learning_rate: float = 5e-5         # 学习率
    num_epochs: int = 3                  # 训练轮数
    batch_size: int = 4                  # 批次大小
    max_length: int = 1024               # 最大序列长度
    warmup_ratio: float = 0.1            # 预热比例
    weight_decay: float = 0.01           # 权重衰减
    gradient_accumulation_steps: int = 1  # 梯度累积步数
    max_grad_norm: float = 1.0            # 梯度裁剪
    logging_steps: int = 10               # 日志打印步数
    save_steps: int = 100                 # 模型保存步数
    eval_steps: int = 50                   # 评估步数
    output_dir: str = "./data/outputs"    # 输出目录


@dataclass
class PreferencePair:
    """偏好对数据结构。"""
    prompt: str                           # 提示词（输入）
    chosen: str                           # 优选回复（chosen / winning）
    rejected: str                         # 劣选回复（rejected / losing）
    chosen_score: Optional[float] = None  # 优选得分（可选）
    rejected_score: Optional[float] = None # 劣选得分（可选）
    metadata: Dict = field(default_factory=dict)


@dataclass
class DPOTrainResult:
    """DPO 训练结果。"""
    loss_history: List[float] = field(default_factory=list)
    reward_margin_history: List[float] = field(default_factory=list)
    chosen_rewards: List[float] = field(default_factory=list)
    rejected_rewards: List[float] = field(default_factory=list)
    final_loss: float = 0.0
    final_reward_margin: float = 0.0
    epochs_trained: int = 0
    steps_trained: int = 0


class DPOLossCalculator:
    """DPO 损失函数计算器（纯数学实现，不依赖深度学习框架）。

    用于理解 DPO 原理、快速验证损失计算、以及在无 GPU 环境下做损失分析。
    """

    def __init__(self, beta: float = 0.1):
        self.beta = beta

    def compute_loss(
        self,
        policy_chosen_logps: np.ndarray,
        policy_rejected_logps: np.ndarray,
        ref_chosen_logps: np.ndarray,
        ref_rejected_logps: np.ndarray,
    ) -> Tuple[float, Dict]:
        """计算 DPO 损失。

        Args:
            policy_chosen_logps: 策略模型对 chosen 的 log 概率
            policy_rejected_logps: 策略模型对 rejected 的 log 概率
            ref_chosen_logps: 参考模型对 chosen 的 log 概率
            ref_rejected_logps: 参考模型对 rejected 的 log 概率

        Returns:
            (loss, metrics) 损失值和指标字典
        """
        # 核心公式：log π(y|x) - log π_ref(y|x)
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        # DPO 损失：-log σ(β (logratio_chosen - logratio_rejected))
        logits = self.beta * (chosen_logratios - rejected_logratios)
        losses = -np.log(self._sigmoid(logits) + 1e-8)

        # 奖励指标（隐式奖励）
        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios
        reward_margin = chosen_rewards - rejected_rewards

        metrics = {
            "loss": float(np.mean(losses)),
            "chosen_rewards": float(np.mean(chosen_rewards)),
            "rejected_rewards": float(np.mean(rejected_rewards)),
            "reward_margin": float(np.mean(reward_margin)),
            "chosen_logratios": float(np.mean(chosen_logratios)),
            "rejected_logratios": float(np.mean(rejected_logratios)),
            "accuracy": float(np.mean(logits > 0)),
        }
        return float(np.mean(losses)), metrics

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class PreferenceDataProcessor:
    """偏好数据处理器：加载、清洗、格式化偏好对数据。"""

    @staticmethod
    def load_from_json(path: str | Path) -> List[PreferencePair]:
        """从 JSON 文件加载偏好对数据。

        支持两种格式：
        1. 标准格式：[{"prompt": "...", "chosen": "...", "rejected": "..."}]
        2. 对话格式：[{"conversations": [...], "chosen": {...}, "rejected": {...}}]
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pairs = []
        for item in data:
            if "prompt" in item and "chosen" in item and "rejected" in item:
                pairs.append(PreferencePair(
                    prompt=item["prompt"],
                    chosen=item["chosen"],
                    rejected=item["rejected"],
                    chosen_score=item.get("chosen_score"),
                    rejected_score=item.get("rejected_score"),
                    metadata=item.get("metadata", {}),
                ))
            elif "conversations" in item:
                prompt = PreferenceDataProcessor._extract_prompt(item["conversations"])
                chosen = item.get("chosen", {}).get("content", "")
                rejected = item.get("rejected", {}).get("content", "")
                if prompt and chosen and rejected:
                    pairs.append(PreferencePair(prompt=prompt, chosen=chosen, rejected=rejected))
        return pairs

    @staticmethod
    def _extract_prompt(conversations: List[Dict]) -> str:
        """从对话历史中提取提示词。"""
        parts = []
        for msg in conversations:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        return "\n".join(parts)

    @staticmethod
    def filter_by_length(pairs: List[PreferencePair], max_length: int = 1024) -> List[PreferencePair]:
        """按长度过滤，移除过长的样本。"""
        return [
            p for p in pairs
            if len(p.prompt) + len(p.chosen) <= max_length
            and len(p.prompt) + len(p.rejected) <= max_length
        ]

    @staticmethod
    def deduplicate(pairs: List[PreferencePair]) -> List[PreferencePair]:
        """去重：基于 prompt 的哈希。"""
        seen = set()
        unique = []
        for p in pairs:
            key = hash(p.prompt.strip())
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    @staticmethod
    def quality_score(pair: PreferencePair) -> float:
        """计算偏好对的质量分数（0-1）。

        评分维度：
        - 长度差异（chosen 和 rejected 不应完全相同）
        - 内容质量（非空、非重复）
        - 语义差异（简单的启发式判断）
        """
        score = 0.0
        # 非空检查
        if pair.prompt and pair.chosen and pair.rejected:
            score += 0.3
        # 长度差异（chosen 和 rejected 应该不同）
        len_diff = abs(len(pair.chosen) - len(pair.rejected))
        if len_diff > 10:
            score += 0.2
        # 内容不完全相同
        if pair.chosen != pair.rejected:
            score += 0.3
        # 合理长度
        if 50 < len(pair.chosen) < 2000 and 50 < len(pair.rejected) < 2000:
            score += 0.2
        return min(score, 1.0)


class DPOMetricsAnalyzer:
    """DPO 训练指标分析器。"""

    @staticmethod
    def analyze(result: DPOTrainResult) -> Dict:
        """分析训练结果，返回关键指标。"""
        if not result.loss_history:
            return {"status": "no_data"}

        losses = np.array(result.loss_history)
        margins = np.array(result.reward_margin_history)

        return {
            "final_loss": float(losses[-1]),
            "loss_reduction": float(losses[0] - losses[-1]) if len(losses) > 1 else 0.0,
            "loss_reduction_pct": float((losses[0] - losses[-1]) / losses[0] * 100) if losses[0] > 0 else 0.0,
            "final_reward_margin": float(margins[-1]) if len(margins) > 0 else 0.0,
            "reward_margin_increase": float(margins[-1] - margins[0]) if len(margins) > 1 else 0.0,
            "min_loss": float(np.min(losses)),
            "max_loss": float(np.max(losses)),
            "loss_std": float(np.std(losses)),
            "epochs_trained": result.epochs_trained,
            "steps_trained": result.steps_trained,
            "converged": bool(losses[-1] < losses[0] * 0.5) if len(losses) > 1 else False,
        }

    @staticmethod
    def compare_implementations(
        custom_result: DPOTrainResult,
        trl_result: DPOTrainResult,
    ) -> Dict:
        """对比从零实现和 TRL 实现的训练结果。"""
        return {
            "custom_final_loss": custom_result.final_loss,
            "trl_final_loss": trl_result.final_loss,
            "loss_diff": abs(custom_result.final_loss - trl_result.final_loss),
            "custom_reward_margin": custom_result.final_reward_margin,
            "trl_reward_margin": trl_result.final_reward_margin,
            "margin_diff": abs(custom_result.final_reward_margin - trl_result.final_reward_margin),
            "recommendation": "TRL" if trl_result.final_loss < custom_result.final_loss else "custom",
        }
