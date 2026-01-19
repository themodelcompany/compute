import time
from typing import List

from verifier.verifier import VerifierNode
from worker.worker import WorkerNode


def build_matrix(rows: int, cols: int) -> List[List[int]]:
    matrix = []
    value = 1
    for _ in range(rows):
        row = []
        for _ in range(cols):
            row.append(value % 17)
            value += 1
        matrix.append(row)
    return matrix


def main() -> None:
    worker = WorkerNode(pubkey="worker-bench")
    verifier = VerifierNode(pubkey="verifier-bench")

    rows = 64
    inner = 64
    cols = 64
    input_matrix = build_matrix(rows, inner)
    weights = [build_matrix(inner, cols)]

    job = {
        "job_id": "bench",
        "sku_id": "llama3_8b_int8_batch_v1",
        "shard_id": "bench-shard",
        "input_matrix": input_matrix,
        "weights": weights,
    }

    start = time.perf_counter()
    output, receipt = worker.run_job(type("Job", (), job))
    inference_time = time.perf_counter() - start

    r_vector = [1 for _ in range(cols)]
    response = worker.respond_challenge(0, 0, r_vector, row_indices=[0, 1, 2])
    start = time.perf_counter()
    result = verifier.verify_challenge(
        receipt_id="bench",
        input_matrix=input_matrix,
        merkle_root=receipt.gemm_commitments[0].merkle_root,
        response_layer_index=0,
        response_gemm_index=0,
        r_vector=response.r_vector,
        wr_vector=response.wr_vector,
        yr_vector=response.yr_vector,
        merkle_proofs=response.merkle_proofs,
    )
    verification_time = time.perf_counter() - start

    ratio = (verification_time / inference_time) if inference_time > 0 else 0
    print("verification verdict:", result.verdict)
    print("inference_time_sec:", round(inference_time, 6))
    print("verification_time_sec:", round(verification_time, 6))
    print("verification_ratio:", round(ratio, 4))


if __name__ == "__main__":
    main()
