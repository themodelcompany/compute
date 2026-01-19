from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional


try:
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("PyTorch is required for LLM hooks. Install torch.") from exc

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("transformers is required for LLM hooks.") from exc


@dataclass
class LLMBackendConfig:
    model_name: str
    dtype: str = "bfloat16"
    max_new_tokens: int = 32
    trust_remote_code: bool = True
    use_distributed: bool = False
    dist_backend: str = "nccl"


def _parse_dtype(dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def init_distributed(backend: str = "nccl", init_method: str = "env://") -> None:
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend, init_method=init_method)


def get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


class LLMBackend:
    def __init__(self, config: LLMBackendConfig) -> None:
        self.config = config
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        if self.config.use_distributed:
            init_distributed(backend=self.config.dist_backend)

        dtype = _parse_dtype(self.config.dtype)
        world_size = get_world_size()
        device_map = "auto" if world_size > 1 else None

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=self.config.trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=self.config.trust_remote_code,
        )

        if device_map is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)

        self.model.eval()

    def generate(self, prompts: List[str]) -> List[str]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("LLMBackend.load() must be called before generate().")

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        if not hasattr(self.model, "device"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
        else:
            input_ids = input_ids.to(self.model.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def build_backend_from_env() -> LLMBackend:
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B")
    dtype = os.getenv("MODEL_DTYPE", "bfloat16")
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "32"))
    use_distributed = os.getenv("USE_DISTRIBUTED", "0") == "1"
    dist_backend = os.getenv("DIST_BACKEND", "gloo" if not torch.cuda.is_available() else "nccl")
    return LLMBackend(
        LLMBackendConfig(
            model_name=model_name,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            use_distributed=use_distributed,
            dist_backend=dist_backend,
        )
    )
