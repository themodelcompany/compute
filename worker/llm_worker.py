from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .llm_backend import LLMBackend, LLMBackendConfig


@dataclass(frozen=True)
class LLMJob:
    job_id: str
    sku_id: str
    shard_id: str
    prompts: List[str]


@dataclass(frozen=True)
class LLMOutput:
    responses: List[str]


class LLMWorkerNode:
    def __init__(self, pubkey: str, backend_config: LLMBackendConfig) -> None:
        self.pubkey = pubkey
        self.backend = LLMBackend(backend_config)
        self.backend.load()

    def run_job(self, job: LLMJob) -> LLMOutput:
        responses = self.backend.generate(job.prompts)
        return LLMOutput(responses=responses)
