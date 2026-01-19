## Worker Node (MVP)

Provides a deterministic inference pipeline and commitment generation for a
toy GEMM-only model. This is a reference implementation for integration tests.

### LLM Hooks (PyTorch + Transformers)
- `llm_backend.py` loads Llama/Qwen via `transformers` with optional distributed inference.
- `llm_worker.py` exposes a minimal worker interface for batch prompts.
