## Commitment And Proof Spec (MVP)

### Commitment Types
- Output commitment: Merkle root of final output matrix `Y`.
- GEMM commitments: Merkle roots per GEMM output matrix.

### Merkle Tree
- Binary Merkle tree over matrix rows.
- Leaf = `sha256(serialize_row(row_index, row_values))`.
- Internal node = `sha256(left || right)`.

### Row Serialization
- `row_index` as uint32 LE.
- `row_values` as int32 LE array (post-quantization output).
- No padding beyond fixed row length.

### Receipt Layout
```
Receipt {
  worker_pubkey
  job_id
  shard_id
  sku_id
  output_root
  gemm_commitments: [{layer_index, gemm_index, merkle_root}]
}
```

### Challenge Response
```
Response {
  layer_index
  gemm_index
  r_vector
  wr_vector
  yr_vector
  merkle_proofs: [{row_index, row_values, siblings[]}]
}
```

### Freivalds Check
- Verifier computes `X(Wr)` modulo `P`.
- Accept if `X(Wr) == Yr` for all rounds.

### Hash And Encoding
- Hash function: SHA-256.
- Vectors are serialized as int32 LE values.
