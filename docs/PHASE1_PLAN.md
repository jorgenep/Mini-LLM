# Phase 1 — Model and Tokenizer Foundation

## Deliverables

- Reproducible byte-level BPE implementation
- Stable reserved special-token IDs
- Configurable decoder-only transformer
- GQA, RoPE, RMSNorm, and SwiGLU
- CPU shape/save-load/context tests
- Exact target parameter report
- Data source manifests, exact deduplication, deterministic splits
- Causal and PSM/SPM FIM example transforms
- Packed fixed-length sequence support

## Frozen provisional target

The instantiated target model has **65,546,880 trainable parameters** with:

- 8 layers
- Hidden size 640
- 10 query heads
- 2 KV heads
- Head dimension 64
- FFN dimension 3072
- Vocabulary 16,384
- Context length 2,048
- Tied embeddings

This is a measured count from the actual modules.

## Observed B70 smoke result

- PyTorch XPU: available
- BF16 forward: passed and finite
- Backward and optimizer step: passed
- Checkpoint save and restore: passed
- Tested shape: `[2, 128, 16384]`

## Phase 1 stop point

This implementation stops before downloading datasets or starting training. Dataset collection begins only after the B70 target-model forward/backward smoke test passes.
