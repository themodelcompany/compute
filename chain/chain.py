import hashlib
from dataclasses import asdict
from typing import List, Tuple

from .randomness import derive_random_vectors, select_gemm_indices
from .types import (
    Challenge,
    ChainState,
    GemmCommitment,
    Job,
    Receipt,
    RewardAccount,
    Verification,
    Worker,
)


class Chain:
    def __init__(self) -> None:
        self.state = ChainState()

    def register_worker(self, pubkey: str, stake: int, supported_skus: List[str]) -> Worker:
        worker = Worker(pubkey=pubkey, stake=stake, supported_skus=supported_skus)
        self.state.workers[pubkey] = worker
        self.state.accounts.setdefault(pubkey, RewardAccount())
        return worker

    def create_job(self, job_id: str, sku_id: str, input_root: str, shard_size: int, payment: int) -> Job:
        job = Job(job_id=job_id, sku_id=sku_id, input_root=input_root, shard_size=shard_size, payment=payment)
        self.state.jobs[job_id] = job
        return job

    def submit_receipt(self, receipt: Receipt) -> str:
        receipt_id = self._hash_receipt(receipt)
        self.state.receipts[receipt_id] = receipt
        return receipt_id

    def assign_challenge(
        self,
        receipt_id: str,
        verifier_pubkey: str,
        rounds: int = 20,
        sample_count: int = 2,
    ) -> Challenge:
        receipt = self.state.receipts[receipt_id]
        seed = f"{receipt_id}:{verifier_pubkey}"
        gemm_indices = self._select_gemms(receipt.gemm_commitments, seed, sample_count)
        random_vectors = derive_random_vectors(seed, rounds)
        challenge = Challenge(
            receipt_id=receipt_id,
            verifier_pubkey=verifier_pubkey,
            gemm_indices=gemm_indices,
            random_vectors=random_vectors,
        )
        self.state.challenges[receipt_id] = challenge
        return challenge

    def submit_verification(self, verification: Verification) -> None:
        self.state.verifications[verification.receipt_id] = verification
        if verification.verdict:
            self._settle_reward(verification.receipt_id)
        else:
            self._slash_worker(verification.receipt_id)

    def _settle_reward(self, receipt_id: str) -> None:
        receipt = self.state.receipts[receipt_id]
        job = self.state.jobs[receipt.job_id]
        account = self.state.accounts.setdefault(receipt.worker_pubkey, RewardAccount())
        account.credits += job.shard_size
        account.balance += job.payment

    def _slash_worker(self, receipt_id: str) -> None:
        receipt = self.state.receipts[receipt_id]
        worker = self.state.workers[receipt.worker_pubkey]
        slashed = max(worker.stake // 10, 1)
        self.state.workers[worker.pubkey] = Worker(
            pubkey=worker.pubkey,
            stake=max(worker.stake - slashed, 0),
            supported_skus=worker.supported_skus,
            reputation_score=max(worker.reputation_score - 1, 0),
        )

    def _hash_receipt(self, receipt: Receipt) -> str:
        payload = str(asdict(receipt)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _select_gemms(
        self, commitments: List[GemmCommitment], seed: str, sample_count: int
    ) -> List[Tuple[int, int]]:
        indices = select_gemm_indices(seed, len(commitments), sample_count)
        return [(commitments[idx].layer_index, commitments[idx].gemm_index) for idx in indices]
