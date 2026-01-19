from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class InferenceJob:
    job_id: str
    sku_id: str
    shard_id: str
    input_matrix: List[List[int]]
    weights: List[List[List[int]]]


@dataclass(frozen=True)
class InferenceOutput:
    output_matrix: List[List[int]]
    gemm_outputs: List[List[List[int]]]
