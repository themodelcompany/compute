## Deterministic Inference Spec (MVP)

### SKU Identifier
- `sku_id`: `llama3_8b_int8_batch_v1`
- Model weights hash: `sha256(weights.bin)`
- Tokenizer hash: `sha256(tokenizer.json)`

### Inference Mode
- Batch only; no streaming.
- Greedy decoding only (temperature = 0).
- Fixed max sequence length per job; defined in job payload.

### Arithmetic
- Quantized int8 weights and activations.
- Accumulation in int32.
- For verification, all dot products are also reduced mod `P`.
- Prime modulus `P = 2^61 - 1` (fits in 64-bit; use fast Mersenne reduction).

### GEMM Ordering
All GEMMs are executed in fixed order per layer:
1. `X @ Wqkv`
2. `A @ V`
3. `X @ W1`
4. `Z @ W2`

Layer index and GEMM index are included in commitment metadata.

### Tensor Layout
- Row-major contiguous layout.
- Shapes follow `prd.md` section 4.
- Input batch tokens are padded to fixed length with a deterministic pad token.

### Hashing And Serialization
- Matrix rows are serialized as little-endian int32 for commitments.
- Hash function: SHA-256.

### Determinism Requirements
- No floating point math.
- No non-deterministic kernels.
- All randomization for verification is chain-derived only.
