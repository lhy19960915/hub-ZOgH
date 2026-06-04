"""
NER 数据集类：span 标注→BIO 转换 + BERT 子词对齐

教学重点：
  1. cluener2020 的 span 格式转为 BIO 格式
     - span: {"name": {"叶老桂": [[9, 11]]}}
     - BIO:  ['O','O',...,'B-name','I-name','I-name',...]
  2. BERT 子词对齐（word_ids 策略）
     - 中文字符通常一字一token，但 [UNK] 和特殊字符可能例外
     - 非首子词标记为 -100，在 loss 计算中被忽略
  3. DataLoader 工厂函数统一封装

使用方式：
  from dataset import build_label_schema, build_dataloaders
"""

import json
from pathlib import Path
from typing import Optional, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "cluener"

ENTITY_TYPES = [
    "address", "book", "company", "game",
    "government", "movie", "name", "organization",
    "position", "scene",
]


def build_label_schema(dataset_name: str = "cluener") -> tuple[list[str], dict[str, int], dict[int, str]]:
    """构建 BIO 标签体系，返回 (labels, label2id, id2label)。
    
    Args:
        dataset_name: 数据集名称，支持 "cluener" 或 "peoples_daily"
    """
    if dataset_name == "peoples_daily":
        labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    else:
        labels = ["O"]
        for etype in ENTITY_TYPES:
            labels.append(f"B-{etype}")
            labels.append(f"I-{etype}")

    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return labels, label2id, id2label

# ... existing code ...
