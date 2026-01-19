from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Worker:
    pubkey: str
    stake: int
    supported_skus: List[str]
    reputation_score: int = 0


@dataclass(frozen=True)
class Job:
    job_id: str
    sku_id: str
    input_root: str
    shard_size: int
    payment: int


@dataclass(frozen=True)
class GemmCommitment:
    layer_index: int
    gemm_index: int
    merkle_root: str


@dataclass(frozen=True)
class Receipt:
    worker_pubkey: str
    job_id: str
    shard_id: str
    sku_id: str
    output_root: str
    gemm_commitments: List[GemmCommitment]


@dataclass(frozen=True)
class Challenge:
    receipt_id: str
    verifier_pubkey: str
    gemm_indices: List[Tuple[int, int]]
    random_vectors: List[str]


@dataclass(frozen=True)
class Verification:
    receipt_id: str
    verifier_pubkey: str
    gemm_indices: List[Tuple[int, int]]
    random_vectors: List[str]
    verdict: bool


@dataclass
class RewardAccount:
    credits: int = 0
    balance: int = 0


@dataclass
class ChainState:
    workers: Dict[str, Worker] = field(default_factory=dict)
    jobs: Dict[str, Job] = field(default_factory=dict)
    receipts: Dict[str, Receipt] = field(default_factory=dict)
    challenges: Dict[str, Challenge] = field(default_factory=dict)
    verifications: Dict[str, Verification] = field(default_factory=dict)
    accounts: Dict[str, RewardAccount] = field(default_factory=dict)
