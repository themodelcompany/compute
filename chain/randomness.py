import hashlib
from typing import List


def derive_random_vectors(seed: str, count: int) -> List[str]:
    vectors = []
    for idx in range(count):
        data = f"{seed}:{idx}".encode("utf-8")
        vectors.append(hashlib.sha256(data).hexdigest())
    return vectors


def select_gemm_indices(seed: str, total_gemms: int, count: int) -> List[int]:
    if total_gemms <= 0:
        return []
    indices = []
    digest = seed.encode("utf-8")
    cursor = 0
    while len(indices) < count:
        digest = hashlib.sha256(digest).digest()
        value = int.from_bytes(digest[:8], "little")
        idx = value % total_gemms
        if idx not in indices:
            indices.append(idx)
        cursor += 1
    return indices
