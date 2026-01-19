from typing import List

from chain.chain import Chain
from chain.types import Receipt, Verification


class SDKClient:
    def __init__(self, chain: Chain) -> None:
        self.chain = chain

    def register_worker(self, pubkey: str, stake: int, supported_skus: List[str]) -> None:
        self.chain.register_worker(pubkey, stake, supported_skus)

    def create_job(self, job_id: str, sku_id: str, input_root: str, shard_size: int, payment: int) -> None:
        self.chain.create_job(job_id, sku_id, input_root, shard_size, payment)

    def submit_receipt(self, receipt: Receipt) -> str:
        return self.chain.submit_receipt(receipt)

    def assign_challenge(self, receipt_id: str, verifier_pubkey: str) -> None:
        self.chain.assign_challenge(receipt_id, verifier_pubkey)

    def submit_verification(self, verification: Verification) -> None:
        self.chain.submit_verification(verification)
