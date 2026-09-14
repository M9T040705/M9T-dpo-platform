"""偏好数据构造器：从原始数据自动构造 DPO 训练所需的偏好对。

支持多种构造策略：
1. 打分排序法：用奖励模型/人工打分对多个回复排序，取最高和最低
2. 模型对比法：用多个模型生成回复，人工/自动标注优劣
3. 规则构造法：基于规则（长度、格式、关键词）自动构造正负样本
4. 自我对弈法：用当前模型生成多个候选，用参考模型打分排序
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .dpo_trainer import PreferencePair


@dataclass
class BuildConfig:
    """偏好数据构造配置。"""
    strategy: str = "score_ranking"       # 构造策略：score_ranking / model_compare / rule_based / self_play
    min_length: int = 20                    # 最小长度
    max_length: int = 2000                   # 最大长度
    min_quality_score: float = 0.5          # 最低质量分数
    deduplicate: bool = True                 # 是否去重
    shuffle: bool = True                     # 是否打乱
    seed: int = 42                           # 随机种子
    output_format: str = "json"              # 输出格式：json / jsonl


class ScoreRankingBuilder:
    """打分排序法：对多个回复按分数排序，取最高为 chosen，最低为 rejected。"""

    def __init__(self, config: BuildConfig):
        self.config = config
        random.seed(config.seed)

    def build(self, samples: List[Dict]) -> List[PreferencePair]:
        """
        输入格式：
        [
            {
                "prompt": "问题",
                "responses": [
                    {"text": "回复1", "score": 0.9},
                    {"text": "回复2", "score": 0.5},
                    {"text": "回复3", "score": 0.2}
                ]
            }
        ]
        """
        pairs = []
        for sample in samples:
            prompt = sample.get("prompt", "")
            responses = sample.get("responses", [])
            if not prompt or len(responses) < 2:
                continue

            scored = [(r["text"], r.get("score", 0)) for r in responses if r.get("text")]
            if len(scored) < 2:
                continue

            scored.sort(key=lambda x: x[1], reverse=True)
            chosen_text, chosen_score = scored[0]
            rejected_text, rejected_score = scored[-1]

            if chosen_text == rejected_text:
                continue

            pair = PreferencePair(
                prompt=prompt,
                chosen=chosen_text,
                rejected=rejected_text,
                chosen_score=chosen_score,
                rejected_score=rejected_score,
                metadata={"strategy": "score_ranking", "num_candidates": len(scored)},
            )
            pairs.append(pair)

        return self._post_process(pairs)

    def _post_process(self, pairs: List[PreferencePair]) -> List[PreferencePair]:
        if self.config.deduplicate:
            seen = set()
            unique = []
            for p in pairs:
                key = hash(p.prompt + p.chosen[:50])
                if key not in seen:
                    seen.add(key)
                    unique.append(p)
            pairs = unique
        if self.config.shuffle:
            random.shuffle(pairs)
        return pairs


class RuleBasedBuilder:
    """规则构造法：基于启发式规则自动构造正负样本。

    适用场景：没有人工标注，只有单轮问答数据，用规则自动判断优劣。
    """

    def __init__(self, config: BuildConfig):
        self.config = config
        random.seed(config.seed)

    def build(self, qa_pairs: List[Dict]) -> List[PreferencePair]:
        """
        输入格式：
        [
            {"question": "问题", "answer": "优质回答", "bad_answer": "劣质回答（可选）"}
        ]
        如果没有 bad_answer，自动生成一个劣质版本。
        """
        pairs = []
        for item in qa_pairs:
            prompt = item.get("question", "") or item.get("prompt", "")
            good = item.get("answer", "") or item.get("good_answer", "")
            bad = item.get("bad_answer", "")

            if not prompt or not good:
                continue

            if not bad:
                bad = self._generate_bad_answer(good)

            if bad == good or len(bad) < self.config.min_length:
                continue

            pair = PreferencePair(
                prompt=prompt,
                chosen=good,
                rejected=bad,
                metadata={"strategy": "rule_based"},
            )
            pairs.append(pair)

        return self._post_process(pairs)

    def _generate_bad_answer(self, good: str) -> str:
        """自动生成劣质回答（用于数据增强）。"""
        strategies = [
            self._truncate,
            self._repeat,
            self._remove_keywords,
            self._add_noise,
        ]
        strategy = random.choice(strategies)
        return strategy(good)

    @staticmethod
    def _truncate(text: str) -> str:
        """截断：只保留前30%。"""
        cut = max(int(len(text) * 0.3), 20)
        return text[:cut] + "..."

    @staticmethod
    def _repeat(text: str) -> str:
        """重复：重复开头部分。"""
        repeat_part = text[:len(text) // 3]
        return repeat_part + " " + repeat_part + " " + repeat_part

    @staticmethod
    def _remove_keywords(text: str) -> str:
        """移除关键词：删除数字和专业术语。"""
        result = re.sub(r'\d+', '', text)
        result = re.sub(r'[A-Za-z]{2,}', '', result)
        return result if len(result) > 20 else text[:30]

    @staticmethod
    def _add_noise(text: str) -> str:
        """添加噪声：插入无关内容。"""
        noise_phrases = ["我不知道。", "可能吧。", "这个问题很难。", "请参考其他资料。"]
        noise = random.choice(noise_phrases)
        return noise + " " + text[:len(text) // 2]

    def _post_process(self, pairs: List[PreferencePair]) -> List[PreferencePair]:
        if self.config.shuffle:
            random.shuffle(pairs)
        return pairs


class ModelCompareBuilder:
    """模型对比法：用多个模型生成回复，对比质量构造偏好对。"""

    def __init__(self, config: BuildConfig):
        self.config = config

    def build(self, multi_model_outputs: List[Dict]) -> List[PreferencePair]:
        """
        输入格式：
        [
            {
                "prompt": "问题",
                "outputs": {
                    "model_a": {"text": "回复A", "score": 0.8},
                    "model_b": {"text": "回复B", "score": 0.4}
                }
            }
        ]
        """
        pairs = []
        for item in multi_model_outputs:
            prompt = item.get("prompt", "")
            outputs = item.get("outputs", {})
            if not prompt or len(outputs) < 2:
                continue

            scored = [(name, out["text"], out.get("score", 0))
                      for name, out in outputs.items() if out.get("text")]
            if len(scored) < 2:
                continue

            scored.sort(key=lambda x: x[2], reverse=True)
            chosen_name, chosen_text, chosen_score = scored[0]
            rejected_name, rejected_text, rejected_score = scored[-1]

            if chosen_text == rejected_text:
                continue

            pair = PreferencePair(
                prompt=prompt,
                chosen=chosen_text,
                rejected=rejected_text,
                chosen_score=chosen_score,
                rejected_score=rejected_score,
                metadata={
                    "strategy": "model_compare",
                    "chosen_model": chosen_name,
                    "rejected_model": rejected_name,
                },
            )
            pairs.append(pair)

        return pairs


class PreferenceDatasetExporter:
    """偏好数据集导出器：导出为多种格式。"""

    @staticmethod
    def to_json(pairs: List[PreferencePair], path: str | Path) -> None:
        """导出为 JSON 数组格式。"""
        data = []
        for p in pairs:
            item = {
                "prompt": p.prompt,
                "chosen": p.chosen,
                "rejected": p.rejected,
            }
            if p.chosen_score is not None:
                item["chosen_score"] = p.chosen_score
            if p.rejected_score is not None:
                item["rejected_score"] = p.rejected_score
            if p.metadata:
                item["metadata"] = p.metadata
            data.append(item)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def to_jsonl(pairs: List[PreferencePair], path: str | Path) -> None:
        """导出为 JSONL 格式（每行一个样本）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for p in pairs:
                item = {"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected}
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    @staticmethod
    def to_trl_format(pairs: List[PreferencePair], path: str | Path) -> None:
        """导出为 TRL 库的 DPO 训练格式。"""
        data = []
        for p in pairs:
            data.append({
                "prompt": p.prompt,
                "chosen": [
                    {"role": "user", "content": p.prompt},
                    {"role": "assistant", "content": p.chosen},
                ],
                "rejected": [
                    {"role": "user", "content": p.prompt},
                    {"role": "assistant", "content": p.rejected},
                ],
            })

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def stats(pairs: List[PreferencePair]) -> Dict:
        """统计数据集信息。"""
        prompt_lengths = [len(p.prompt) for p in pairs]
        chosen_lengths = [len(p.chosen) for p in pairs]
        rejected_lengths = [len(p.rejected) for p in pairs]

        return {
            "total": len(pairs),
            "avg_prompt_length": sum(prompt_lengths) / len(pairs) if pairs else 0,
            "avg_chosen_length": sum(chosen_lengths) / len(pairs) if pairs else 0,
            "avg_rejected_length": sum(rejected_lengths) / len(pairs) if pairs else 0,
            "min_prompt_length": min(prompt_lengths) if prompt_lengths else 0,
            "max_prompt_length": max(prompt_lengths) if prompt_lengths else 0,
            "strategies": {},
        }
