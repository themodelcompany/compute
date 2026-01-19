import os

from worker.llm_backend import build_backend_from_env, get_rank, get_world_size


def main() -> None:
    backend = build_backend_from_env()
    backend.load()

    prompts = [
        "Explain Freivalds' algorithm in one sentence.",
        "What is deterministic inference?",
    ]
    outputs = backend.generate(prompts)

    rank = get_rank()
    world = get_world_size()
    for idx, text in enumerate(outputs):
        print(f"rank={rank}/{world} output[{idx}]: {text}")


if __name__ == "__main__":
    os.environ.setdefault("USE_DISTRIBUTED", "1")
    main()
