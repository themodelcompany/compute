## Spec Overview

This directory defines deterministic inference and commitment formats used by
the chain, worker, and verifier. These specs are intended to be stable and
versioned. The MVP targets a single SKU and expands later.

### Documents
- `deterministic_inference.md`: arithmetic, quantization, and GEMM ordering
- `commitments.md`: Merkle roots, GEMM commitments, and proof layout
- `test_vectors.md`: sample hashes and vector formats for cross-impl checks
