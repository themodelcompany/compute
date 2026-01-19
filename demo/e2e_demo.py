import hashlib
from typing import List

from chain.chain import Chain
from sdk.sdk import SDKClient
from verifier.verifier import VerifierNode
from worker.types import InferenceJob
from worker.worker import WorkerNode


def vector_from_seed(seed: str, length: int) -> List[int]:
    data = seed.encode("utf-8")
    values = []
    while len(values) < length:
        data = hashlib.sha256(data).digest()
        for i in range(0, len(data), 4):
            if len(values) >= length:
                break
            values.append(int.from_bytes(data[i : i + 4], "little", signed=False))
    return values


def main() -> None:
    chain = Chain()
    sdk = SDKClient(chain)
    worker = WorkerNode(pubkey="worker-1")
    verifier = VerifierNode(pubkey="verifier-1")

    sdk.register_worker("worker-1", stake=1000, supported_skus=["llama3_8b_int8_batch_v1"])
    sdk.create_job(job_id="job-1", sku_id="llama3_8b_int8_batch_v1", input_root="input-root", shard_size=4, payment=10)

    input_matrix = [[1, 2], [3, 4], [5, 6], [7, 8]]
    weights = [
        [[1, 0, 2], [0, 1, 1]],
        [[2, 1], [1, 0], [0, 1]],
    ]
    job = InferenceJob(job_id="job-1", sku_id="llama3_8b_int8_batch_v1", shard_id="shard-1", input_matrix=input_matrix, weights=weights)
    output, receipt = worker.run_job(job)

    receipt_id = sdk.submit_receipt(receipt)
    sdk.assign_challenge(receipt_id, verifier.pubkey)
    challenge = chain.state.challenges[receipt_id]

    gemm_layer, gemm_index = challenge.gemm_indices[0]
    r_vector = vector_from_seed(challenge.random_vectors[0], len(weights[gemm_index][0]))
    response = worker.respond_challenge(gemm_layer, gemm_index, r_vector, row_indices=[0, 1])

    merkle_root = receipt.gemm_commitments[gemm_index].merkle_root
    result = verifier.verify_challenge(
        receipt_id=receipt_id,
        input_matrix=worker.gemm_inputs[(gemm_layer, gemm_index)],
        merkle_root=merkle_root,
        response_layer_index=response.layer_index,
        response_gemm_index=response.gemm_index,
        r_vector=response.r_vector,
        wr_vector=response.wr_vector,
        yr_vector=response.yr_vector,
        merkle_proofs=response.merkle_proofs,
    )

    verification = verifier.build_verification_receipt(
        receipt_id=receipt_id,
        gemm_indices=challenge.gemm_indices,
        random_vectors=challenge.random_vectors,
        verdict=result.verdict,
    )
    sdk.submit_verification(verification)

    print("verification:", result.verdict, result.reason)
    print("worker balance:", chain.state.accounts["worker-1"].balance)


if __name__ == "__main__":
    main()
