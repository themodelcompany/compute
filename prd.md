# PRD

## Inference-Secured Blockchain for Distributed LLM Inference

**(Llama / Qwen, Batch-First, Freivalds-Verified)**

---

## 1. Executive Summary

Build a blockchain where **validators and workers earn block rewards and fees by performing verifiable LLM inference**.
The dominant computation is **real LLM GEMMs**, not hash puzzles.

Correctness is enforced using **randomized linear-algebra verification (Freivalds checks)** on selected GEMMs inside Llama/Qwen inference, rather than full re-execution.

Key properties:

* **All compute is useful** (no nonce grinding).
* **Verification is asymptotically cheaper** than re-running inference.
* **Security scales with inference throughput**.
* **Batch inference only** for v1 to minimize cost.
* **Deterministic inference SKUs** for auditability.

---

## 2. Design Constraints (Hard Requirements)

1. **Inference is the work**

   * Validators/workers must execute LLM inference to earn rewards.
2. **Verification must be cheaper than inference**

   * No design that requires re-running full inference for every block.
3. **LLM-realistic GEMMs**

   * Must support Llama/Qwen shapes (hidden sizes 4k–8k, MLP expansions).
4. **Deterministic outputs**

   * Consensus disputes must be resolvable exactly.
5. **Low cost**

   * Buyers must pay less than retail inference.
   * Workers must earn more consistently than PoW mining.
6. **Permissionless**

   * Anyone with stake + GPU can participate.

---

## 3. Supported Models (v1)

### Llama

* Llama-3-8B
* Llama-3-70B (later, after sharding maturity)

### Qwen

* Qwen2.5-7B
* Qwen2.5-14B

### Inference Mode

* **Batch only**
* **Greedy decoding only** (no temperature)
* **Fixed tokenizer + model hash**
* **Quantized inference** (int8 or int4)

---

## 4. Typical GEMM Shapes (Used for Verification Design)

For a batch of `T` tokens and hidden size `H`:

### Attention projections

* `X @ Wqkv`

  * `X`: `(T × H)`
  * `Wqkv`: `(H × 3H)`
  * `Y`: `(T × 3H)`

### Attention output

* `A @ V`

  * `A`: `(T × T)`
  * `V`: `(T × H)`
  * `Y`: `(T × H)`

### MLP

* `X @ W1`

  * `(T × H) @ (H × 4H)`
* `Z @ W2`

  * `(T × 4H) @ (4H × H)`

These GEMMs dominate runtime and are the **only operations verified**.

---

## 5. Core Insight

**We do not verify full inference.**
We verify **random linear projections of selected GEMMs**:

[
X(Wr) \stackrel{?}{=} Yr
]

This detects *any incorrect GEMM* with probability ≥ (1 - 2^{-k}) after `k` rounds.

This is **orders of magnitude cheaper** than recomputation.

---

## 6. Chain Architecture Overview

### Consensus Model

**Inference-PoS**

* Stake prevents Sybil attacks
* Inference work earns rewards
* Validators must maintain verified inference throughput

### Roles

| Role      | Responsibility                   |
| --------- | -------------------------------- |
| Worker    | Executes LLM inference           |
| Verifier  | Performs Freivalds checks        |
| Validator | Produces blocks, settles rewards |
| Buyer     | Pays for inference jobs          |

A single node may play multiple roles.

---

## 7. Deterministic Inference Specification (Critical)

Every SKU defines:

* Model weight hash
* Tokenizer hash
* Quantization scheme
* Exact GEMM ordering
* Fixed arithmetic spec:

  * **int8 / int4 integer math**
  * accumulation in int32
  * modulo large prime for verification

Floating-point nondeterminism is **not allowed**.

---

## 8. Work Unit Definition

### Atomic Work Unit

**One shard of batch inference**, defined as:

* Model SKU
* Input token batch
* Exact token positions
* Deterministic outputs
* Set of internal GEMMs executed

Workers earn credit only after verification.

---

## 9. Verification Primitive (Freivalds-Based)

### What is verified

* A random subset of GEMMs per shard:

  * Always MLP GEMMs
  * Sometimes attention GEMMs
* GEMM selection is random and unpredictable

### Verification Protocol (Per GEMM)

1. Worker commits:

   * Merkle root of output matrix `Y`
2. Chain randomness generates:

   * Random vector `r`
3. Worker (or verifier) supplies:

   * `Wr`
   * `Yr`
   * Merkle proofs for slices
4. Verifier checks:
   [
   X(Wr) = Yr
   ]

### Soundness

* 1 round → ≤ 50% false accept
* 20 rounds → ~1e-6
* 40 rounds → cryptographic-grade

Rounds are tunable per worker reputation.

---

## 10. Commit–Challenge–Response Flow

### Step 1: Commit

Worker submits:

* Output commitment
* Intermediate GEMM commitments (hashes only)

### Step 2: Challenge

Chain randomness determines:

* Which GEMMs to verify
* Which random vectors `r`
* Which verifier is assigned

### Step 3: Response

Worker provides:

* Required vector products
* Merkle proofs
* Metadata (layer index, GEMM index)

### Step 4: Verification

Verifier recomputes:

* `X(Wr)` locally
* Compares to `Yr`

Failure → slash.

---

## 11. On-Chain Objects

### Worker Registration

```
Worker {
  pubkey
  stake
  supported_SKUs
  reputation_score
}
```

### Job

```
Job {
  job_id
  sku
  input_root
  shard_size
  payment
}
```

### Work Receipt

```
Receipt {
  worker
  job_id
  shard_id
  output_root
  gemm_commitments
}
```

### Verification Receipt

```
Verification {
  receipt_id
  verifier
  gemm_indices
  random_vectors
  verdict
}
```

---

## 12. Reward Model

### Inference Credits

Workers earn:
[
\text{credits} = \text{verified tokens processed}
]

### Epoch Rewards

At end of epoch:

* Inflation + job fees distributed proportionally to credits
* Validators must meet minimum credits to stay active

### Why this is better than PoW

* No variance lottery
* Predictable earnings
* All compute is billable

---

## 13. Slashing Rules

| Violation                 | Penalty           |
| ------------------------- | ----------------- |
| GEMM verification failure | Heavy slash       |
| Missed response deadline  | Partial slash     |
| Repeated failures         | Validator removal |
| Invalid commitments       | Full shard slash  |

Slashed stake partially funds verifiers.

---

## 14. Verifier Incentives

* Paid per successful verification
* Paid bounty for catching fraud
* Slashed for false reports

Verifier selection is random and stake-weighted.

---

## 15. Throughput & Cost Controls

### Why this stays cheap

* Verification cost scales as **O(N²)** vs inference **O(N³)**
* Only a fraction of GEMMs are checked
* Batch inference amortizes overhead

### Typical cost ratio

* Inference: 100%
* Verification: 1–3%

---

## 16. Security Model

### Attacks Covered

* Fake inference
* Partial inference
* Skipped layers
* Corrupted outputs
* Selective cheating

### Attacks Mitigated by Design

* Pre-computation (random challenges)
* Collusion (random verifier assignment)
* Precision attacks (integer arithmetic)

---

## 17. MVP Scope

### Included

* Llama/Qwen inference
* Batch jobs
* Freivalds verification
* Inference-PoS rewards
* Slashing

### Excluded

* Interactive chat
* ZK proofs
* Private prompts
* Governance DAO

---

## 18. Implementation Breakdown (Agent-Ready)

### Chain

* Inference-PoS logic
* VRF randomness
* Receipt settlement
* Slashing

### Worker Node

* Deterministic inference engine
* GEMM commitment tracking
* Challenge response API

### Verifier Node

* GEMM recomputation engine
* Merkle verification
* Receipt submission

### SDK

* Job submission
* Output verification
* Receipt tracking

---

## 19. Acceptance Criteria

* Worker can earn rewards by running Llama/Qwen inference
* Verifier can catch incorrect GEMMs with high probability
* Verification cost < 5% of inference cost
* Fraud is economically irrational
* System runs cheaper than retail inference

---

## 20. One-Sentence Summary

> This chain replaces hash puzzles with **real LLM inference**, and replaces re-execution with **randomized algebraic verification**, making inference itself the scarce, secure resource.
