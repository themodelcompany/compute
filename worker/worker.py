from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from chain.types import GemmCommitment, Receipt

from .merkle import MerkleTree
from .types import InferenceJob, InferenceOutput


def matmul_int32(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
    rows = len(a)
    cols = len(b[0]) if b else 0
    inner = len(b)
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            aik = int(a[i][k])
            for j in range(cols):
                result[i][j] += aik * int(b[k][j])
    return result


def matvec_int32(matrix: List[List[int]], vector: List[int]) -> List[int]:
    result = []
    for row in matrix:
        acc = 0
        for idx, value in enumerate(row):
            acc += int(value) * int(vector[idx])
        result.append(acc)
    return result


@dataclass
class ChallengeResponse:
    layer_index: int
    gemm_index: int
    r_vector: List[int]
    wr_vector: List[int]
    yr_vector: List[int]
    merkle_proofs: List[Tuple[int, List[int], List[str]]]


class WorkerNode:
    def __init__(self, pubkey: str) -> None:
        self.pubkey = pubkey
        self.gemm_inputs: Dict[Tuple[int, int], List[List[int]]] = {}
        self.gemm_weights: Dict[Tuple[int, int], List[List[int]]] = {}
        self.gemm_outputs: Dict[Tuple[int, int], List[List[int]]] = {}
        self.gemm_trees: Dict[Tuple[int, int], MerkleTree] = {}

    def run_job(self, job: InferenceJob) -> Tuple[InferenceOutput, Receipt]:
        current = job.input_matrix
        gemm_outputs = []
        for idx, weights in enumerate(job.weights):
            gemm_key = (0, idx)
            self.gemm_inputs[gemm_key] = current
            self.gemm_weights[gemm_key] = weights
            output = matmul_int32(current, weights)
            self.gemm_outputs[gemm_key] = output
            self.gemm_trees[gemm_key] = MerkleTree(output)
            gemm_outputs.append(output)
            current = output

        output_matrix = current
        output_tree = MerkleTree(output_matrix)
        commitments = [
            GemmCommitment(layer_index=0, gemm_index=idx, merkle_root=self.gemm_trees[(0, idx)].root())
            for idx in range(len(job.weights))
        ]
        receipt = Receipt(
            worker_pubkey=self.pubkey,
            job_id=job.job_id,
            shard_id=job.shard_id,
            sku_id=job.sku_id,
            output_root=output_tree.root(),
            gemm_commitments=commitments,
        )
        return InferenceOutput(output_matrix=output_matrix, gemm_outputs=gemm_outputs), receipt

    def respond_challenge(
        self, layer_index: int, gemm_index: int, r_vector: List[int], row_indices: List[int]
    ) -> ChallengeResponse:
        key = (layer_index, gemm_index)
        weights = self.gemm_weights[key]
        output = self.gemm_outputs[key]
        wr_vector = matvec_int32(weights, r_vector)
        yr_vector = matvec_int32(output, r_vector)
        tree = self.gemm_trees[key]
        proofs = [(idx, output[idx], tree.get_proof(idx)) for idx in row_indices]
        return ChallengeResponse(
            layer_index=layer_index,
            gemm_index=gemm_index,
            r_vector=r_vector,
            wr_vector=wr_vector,
            yr_vector=yr_vector,
            merkle_proofs=proofs,
        )
