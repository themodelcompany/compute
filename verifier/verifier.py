from __future__ import annotations

from dataclasses import dataclass
from typing import List

from chain.types import Verification
from worker.merkle import verify_proof
from worker.worker import matvec_int32


PRIME_MODULUS = (1 << 61) - 1


@dataclass(frozen=True)
class VerificationResult:
    receipt_id: str
    verdict: bool
    reason: str


def mod_reduce(value: int, modulus: int = PRIME_MODULUS) -> int:
    value = value % modulus
    return value


class VerifierNode:
    def __init__(self, pubkey: str) -> None:
        self.pubkey = pubkey

    def verify_challenge(
        self,
        receipt_id: str,
        input_matrix: List[List[int]],
        merkle_root: str,
        response_layer_index: int,
        response_gemm_index: int,
        r_vector: List[int],
        wr_vector: List[int],
        yr_vector: List[int],
        merkle_proofs: List[List],
    ) -> VerificationResult:
        for row_index, row_values, proof in merkle_proofs:
            if not verify_proof(row_index, row_values, proof, merkle_root):
                return VerificationResult(receipt_id=receipt_id, verdict=False, reason="merkle_proof_failed")

        x_wr = matvec_int32(input_matrix, wr_vector)
        for idx, value in enumerate(x_wr):
            if mod_reduce(value) != mod_reduce(yr_vector[idx]):
                return VerificationResult(receipt_id=receipt_id, verdict=False, reason="freivalds_mismatch")

        return VerificationResult(receipt_id=receipt_id, verdict=True, reason="ok")

    def build_verification_receipt(
        self,
        receipt_id: str,
        gemm_indices: List,
        random_vectors: List[str],
        verdict: bool,
    ) -> Verification:
        return Verification(
            receipt_id=receipt_id,
            verifier_pubkey=self.pubkey,
            gemm_indices=gemm_indices,
            random_vectors=random_vectors,
            verdict=verdict,
        )
