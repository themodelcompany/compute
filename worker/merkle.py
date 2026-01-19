import hashlib
from typing import List, Tuple


def _hash(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def serialize_row(row_index: int, row_values: List[int]) -> bytes:
    data = row_index.to_bytes(4, "little", signed=False)
    for value in row_values:
        data += int(value).to_bytes(4, "little", signed=True)
    return data


class MerkleTree:
    def __init__(self, rows: List[List[int]]) -> None:
        self.rows = rows
        self.leaves = [_hash(serialize_row(idx, row)) for idx, row in enumerate(rows)]
        self.levels = [self.leaves]
        self._build()

    def _build(self) -> None:
        level = self.leaves
        while len(level) > 1:
            next_level = []
            for idx in range(0, len(level), 2):
                left = level[idx]
                right = level[idx + 1] if idx + 1 < len(level) else left
                next_level.append(_hash(left + right))
            self.levels.append(next_level)
            level = next_level

    def root(self) -> str:
        if not self.levels:
            return ""
        return self.levels[-1][0].hex()

    def get_proof(self, index: int) -> List[str]:
        proof = []
        idx = index
        for level in self.levels[:-1]:
            sibling = idx + 1 if idx % 2 == 0 else idx - 1
            if sibling < len(level):
                proof.append(level[sibling].hex())
            idx //= 2
        return proof


def verify_proof(row_index: int, row_values: List[int], proof: List[str], root: str) -> bool:
    computed = _hash(serialize_row(row_index, row_values))
    idx = row_index
    for sibling_hex in proof:
        sibling = bytes.fromhex(sibling_hex)
        if idx % 2 == 0:
            computed = _hash(computed + sibling)
        else:
            computed = _hash(sibling + computed)
        idx //= 2
    return computed.hex() == root
