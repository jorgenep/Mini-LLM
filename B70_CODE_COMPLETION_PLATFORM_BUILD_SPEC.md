# Intel Arc Pro B70 Local Code Completion Platform
## Master Engineering Specification and Build Plan

**Document version:** 1.2  
**Status:** Build specification  
**Primary target:** Intel Arc Pro B70, 32 GB GDDR6  
**Primary languages:** Python, JavaScript, TypeScript  
**Primary product:** Local-first coding completion platform with a custom-trained code model

---

## 1. Product objective

Build a complete local-first AI coding platform, including:

- A custom code model trained on the Intel Arc Pro B70.
- A tokenizer and reproducible data pipeline.
- Pretraining and quality-improvement fine-tuning.
- Syntax-, type-, and execution-aware evaluation.
- A local inference server and VS Code extension.
- A desktop/web management interface for hardware, data, training, evaluation, model versions, inference, and quantization.
- Optional cloud teacher-model access through a provider-independent API.

The model's job is **not to reproduce training files or memorize code examples**. Its job is to understand the code surrounding the cursor and produce the next useful piece of code: a line, expression, statement, loop, conditional, function, class, or fill-in-the-middle block.

### 1.1 Core quality principle

**Syntactic validity is a hard acceptance requirement for a completion presented as valid code.**

The system must strongly prioritize completions that parse successfully in the target language and in their insertion context. Syntax validation is separate from runtime correctness: code can parse correctly and still have type errors, undefined names, incorrect behavior, or runtime crashes. The platform must measure these separately.

The product should optimize, in order:

1. Valid syntax and correct insertion into the existing source.
2. Contextual and structural correctness.
3. Type correctness where applicable.
4. Functional correctness and runtime reliability where verifiable.
5. Relevance, usefulness, and appropriate completion length.
6. Latency and throughput, subject to quality gates.

Do not claim that static syntax validation guarantees code will never crash. No general-purpose verifier can prove that for arbitrary programs. The system must instead combine parsers, type checkers, tests, sandboxed execution, and clear quality metrics.

---

## 2. Decisions already made

These are project requirements, not open questions:

| Area | Decision |
|---|---|
| Existing prototype | Start from scratch |
| Deployment | Local first, with eventual cloud deployment capability |
| Accounts | Individual accounts with passwords for the management platform |
| Student model training | Entirely on the local Intel Arc Pro B70 |
| Teacher model | Cloud API through OpenRouter |
| Training data | Open-source code plus synthetic code |
| Languages | Python and JavaScript/TypeScript |
| Model features | Inline autocomplete and fill-in-the-middle (FIM) |
| User interface | Full desktop/web management interface |
| Product ambition | Complete development platform |
| First-release priority | Best possible model quality |
| Quantization | Benchmark all supported variants and choose automatically |
| Completion objective | Generate the next useful code construct in context, not recreate training examples |
| Syntax objective | Generated completions must pass language-appropriate syntax validation before being classified as valid |

Cloud teacher calls may be used for data generation, filtering, critique, or synthetic examples. The student model's training and inference must run locally on the B70. Do not upload private repositories or confidential source code to a cloud teacher without explicit, separate authorization.

---

## 3. Hardware and software environment

### 3.1 Target hardware

- Intel Arc Pro B70
- 32 GB GDDR6 VRAM
- Advertised memory bandwidth: 608 GB/s
- Xe2/XMX acceleration

Treat these specifications as target hardware information, not proof that every framework, precision, kernel, or serving backend is supported.

### 3.2 Hardware discovery

The application must include a hardware diagnostic that records:

- GPU model and device ID.
- GPU memory capacity and current availability.
- Driver version.
- Linux distribution and version.
- Kernel version.
- Intel compute runtime and Level Zero availability.
- oneAPI components when installed.
- PyTorch version and XPU availability.
- Supported dtypes and tested matrix operations.
- CPU model and core count.
- System RAM.
- NVMe/storage capacity and free space.
- Python version and relevant package versions.

Save diagnostic output to a report that can be attached to bug reports.

### 3.3 Mandatory accelerator gates

Do not begin a long training run until all applicable checks pass:

1. The operating system detects the B70.
2. The Intel runtime and Level Zero stack are functional.
3. `torch.xpu.is_available()` returns true in the chosen PyTorch environment.
4. A BF16 matrix multiplication completes and produces finite, plausible values.
5. A representative model forward pass completes on XPU.
6. A backward pass completes on XPU.
7. An optimizer step completes on XPU.
8. A checkpoint can be saved and restored without corruption.
9. A short multi-step training run has stable loss and no unexplained NaNs/Infs.
10. GPU utilization and memory usage are observable.
11. The actual training throughput is measured before estimating full-run duration.

Example BF16 smoke test:

```python
import torch

assert torch.xpu.is_available()

a = torch.randn(1024, 1024, device="xpu", dtype=torch.bfloat16)
b = torch.randn(1024, 1024, device="xpu", dtype=torch.bfloat16)
out = a @ b

assert torch.isfinite(out).all()
assert out.abs().max() < 1e4
print("BF16 XPU GEMM passed")
```

Run `torch.compile` only as a benchmarked optimization. The system must retain a working non-compiled path if compilation is unstable or unsupported for an operation.

### 3.4 OS policy

Do not assume compatibility based solely on distribution name. Test the actual installed driver, runtime, kernel, PyTorch build, and GPU. If the current Linux environment cannot pass the mandatory gates, document the failure and provide a supported fallback environment rather than silently proceeding on CPU.

---

## 4. Model architecture

### 4.1 Initial architecture target

Start with a compact dense decoder-only transformer, approximately 68 million parameters.

Proposed configuration:

| Parameter | Initial value |
|---|---:|
| Layers | 8 |
| Hidden dimension | 640 |
| Query heads | 10 |
| Key/value heads | 2 |
| Head dimension | 64 |
| FFN dimension | 3072, provisional |
| FFN | SwiGLU |
| Vocabulary | 16,384 |
| Embeddings | Tied input/output embeddings |
| Position encoding | RoPE |
| RoPE theta | 100,000 |
| Initial context length | 2,048 tokens |
| Initial training precision | BF16 if accelerator gates pass |

**Parameter-count verification is a mandatory implementation gate.** The original plan's stated 68.5M parameter count does not match its listed `ffn_dim=1792` configuration. The implementation must calculate the exact number of trainable parameters from the actual modules and print a layer-by-layer breakdown. `ffn_dim=3072` is a provisional correction, not a guaranteed exact 68.5M configuration. Adjust the FFN width or other architecture dimensions if necessary, then record and freeze the verified configuration before full training.

Do not hard-code a claimed parameter count that the instantiated model does not have.

### 4.2 Model implementation requirements

- Decoder-only causal transformer.
- Grouped-query attention (GQA).
- Rotary positional embeddings.
- SwiGLU feed-forward blocks.
- RMSNorm or another explicitly documented normalization choice.
- Tied embeddings unless a measured experiment supports a change.
- Explicit FIM special-token handling.
- Configurable context length.
- Deterministic model configuration serialization.
- Unit tests for tensor shapes, masks, position handling, and save/load equivalence.

The implementation must have a small reference configuration for CPU/unit tests and the target configuration for B70 training.

---

## 5. Tokenizer

### 5.1 Initial tokenizer target

- Byte-level BPE.
- 16,384 vocabulary entries.
- Trained on representative, high-quality Python, JavaScript, and TypeScript code.
- Preserve indentation, whitespace, punctuation, Unicode, and programming-language symbols.
- Store tokenizer configuration and vocabulary with every model checkpoint.

Reserve and document stable IDs for:

- `<|endoftext|>`
- `<|pad|>`
- `<|fim_prefix|>`
- `<|fim_suffix|>`
- `<|fim_middle|>`
- `<|file_sep|>`
- `<|repo_name|>`
- `<|unk|>`

The implementation must verify that special-token IDs are stable and that encoding/decoding round-trips correctly.

### 5.2 Tokenizer tests

Test at minimum:

- Four-space and eight-space indentation.
- Tabs and mixed indentation.
- Newlines and blank lines.
- Nested blocks.
- Braces, brackets, and parentheses.
- Quotes, template strings, and escape sequences.
- TypeScript generics and type annotations.
- Python decorators, type hints, and async syntax.
- Unicode identifiers and comments.
- Long identifiers and unusual but valid symbols.
- FIM prefix/suffix/middle boundaries.

Do not require indentation to be a single token if the learned tokenizer does not naturally produce that result. Measure tokenization efficiency and correctness instead.

---

## 6. Training-data strategy

### 6.1 Data objective

The data pipeline must teach the model to continue code in context. It must not optimize for copying entire training files or reproducing specific examples.

Use a mixture of:

- Permissively licensed or otherwise approved open-source code.
- High-quality synthetic code generated by the cloud teacher.
- Contextual continuation examples.
- FIM examples with meaningful prefix and suffix.
- Examples of incomplete code that should be completed.
- Examples that test indentation, scope, imports, variable use, and language constructs.
- Negative or corrupted examples only when used deliberately for a documented training objective; do not allow malformed code to dominate.

### 6.2 Open-source data

Potential source families include The Stack v2 and StarCoderData, subject to license and access review.

For every source file or record, preserve provenance metadata:

- Repository/source identifier.
- File path or stable record identifier where permitted.
- License identifier and license text/reference.
- Collection date.
- Language classification.
- Deduplication fingerprint.
- Filtering decisions.
- Any transformation history.

Do not treat a dataset as permissively licensed without checking its actual license and usage terms. Exclude data whose license or provenance cannot be established under the project's policy.

### 6.3 Filtering

Filter or separately classify:

- Generated files and generated bundles.
- Minified JavaScript.
- Lockfiles and vendored dependencies unless deliberately needed.
- Binary or corrupt files.
- Exact and near duplicates.
- Empty or extremely short files.
- Boilerplate-heavy content.
- Secrets, credentials, access tokens, private keys, and personal data.
- Files with invalid syntax, unless retained in a separate robustness dataset.
- Unsupported languages.

Preserve valid but incomplete source snippets when useful for completion training. A source file does not need to be a complete application to be useful.

### 6.4 Synthetic data via OpenRouter

Implement a provider-independent teacher interface, with OpenRouter as the initial provider.

Teacher tasks may include:

- Generate realistic code contexts and next-construct completions.
- Produce FIM prefix/suffix/completion triples.
- Create examples targeting specific language constructs.
- Review candidate code for syntax and contextual errors.
- Explain why a completion is invalid.
- Generate edge cases for parser and type-checker evaluation.

Store teacher model name, prompt template version, generation settings, timestamp, and validation results with each synthetic sample.

Synthetic examples must be validated locally. Do not assume teacher-generated code is correct.

### 6.5 Data volume

The original plan proposed 15–20 billion tokens. Treat this as a research target, not a release blocker or guaranteed requirement. First measure data quality, storage, training throughput, validation behavior, and model learning curves. The platform must support smaller pilot datasets and progressively larger corpora.

### 6.6 Train/validation/test separation

Maintain strict separation:

- Training data for gradient updates.
- Validation data for checkpoint selection and tuning.
- Held-out test data for final reporting only.

Deduplicate across all splits to reduce leakage. Keep test prompts and repositories out of teacher-generated training data and fine-tuning.

---

## 7. Training examples and objectives

### 7.1 FIM and causal training

Initial proposed mixture:

- 50% Prefix-Suffix-Middle (PSM) FIM.
- 10% Suffix-Prefix-Middle (SPM) FIM.
- 40% standard causal continuation.

This is a starting experiment, not a permanent assumption. Log the mixture and compare alternative ratios on held-out completion tests.

### 7.2 Completion-oriented examples

Build examples around real editor insertion behavior:

- Cursor at end of an incomplete line.
- Cursor after a function or method signature.
- Cursor inside a loop.
- Cursor inside an if/else block.
- Cursor inside a class.
- Cursor inside a function body.
- Cursor between matching braces.
- Cursor between prefix and suffix code.
- Cursor where the next construct may span multiple lines.
- Cursor where the correct completion is short or empty.

The model should learn when to stop. It must not always generate a large block when a single token or line is appropriate.

### 7.3 Context representation

Where available, examples should include:

- Language identifier.
- File path or file type.
- Prefix before the cursor.
- Suffix after the cursor for FIM.
- Nearby declarations and imports.
- Relevant surrounding code.
- Optional bounded repository context.

Do not assume the entire repository fits in the context window. Context selection should be deterministic, bounded, and configurable.

---

## 8. Pretraining

### 8.1 Optimizer starting point

Initial experiment:

- AdamW.
- Betas `(0.9, 0.95)`.
- Epsilon `1e-8`.
- Weight decay `0.1` on eligible weight matrices; exclude biases and normalization parameters.
- BF16 if stable on the B70.
- Gradient clipping with a configurable threshold.
- Warmup followed by a measured schedule such as WSD or cosine decay.

The original plan proposed peak learning rate `8e-4`, microbatch size 96, gradient accumulation 3, and a 2,048-token sequence length. These are hypotheses to benchmark, not fixed requirements. Determine safe batch size from actual memory use and stable throughput.

### 8.2 Training loop requirements

Implement:

- Gradient accumulation.
- Gradient clipping.
- Learning-rate scheduler.
- Mixed precision where verified.
- Periodic validation.
- Loss and learning-rate logging.
- GPU utilization and VRAM monitoring.
- Checkpoint save/resume.
- Optimizer and scheduler state persistence.
- Random seed persistence.
- Training configuration persistence.
- Graceful interruption and recovery.
- Detection and reporting of NaN/Inf loss or gradients.

### 8.3 Checkpoint policy

Never overwrite the only good checkpoint.

Store:

- Model weights.
- Optimizer state.
- Scheduler state.
- Global step and epoch.
- RNG states.
- Model configuration.
- Tokenizer files.
- Dataset manifest and versions.
- Training configuration.
- Validation metrics.
- Software/runtime versions.

Support automatic retention policies, best-checkpoint tracking, and manual promotion/rollback.

### 8.4 Training estimates

Do not promise the original 20–28 hour pretraining or 30–44 hour end-to-end estimate. Benchmark tokens/second and step time on the actual B70, then calculate an estimate with a documented margin. Display measured progress and ETA in the dashboard.

---

## 9. Syntax-aware generation and validation

This is a core differentiator of the project.

### 9.1 Validation stages

For every candidate in offline evaluation and quality-improvement pipelines, reconstruct the source as it would appear after insertion, then apply the relevant checks.

**Gate 1 — Syntax**

- Python: `ast.parse()` and/or `compile()` on reconstructed source.
- JavaScript: a maintained JavaScript parser.
- TypeScript: TypeScript compiler/parser.
- Optionally Tree-sitter for fast incremental parsing and editor feedback.

A completion is not syntax-valid merely because the generated fragment parses by itself. The reconstructed source with the completion inserted must parse in the intended context. When a file is intentionally incomplete during editing, use a context-aware incremental parser or a carefully defined completion-boundary strategy; do not incorrectly reject every temporarily incomplete editor buffer.

**Gate 2 — Type and static analysis**

- Python: Pyright or another configured checker where project configuration permits.
- TypeScript: `tsc --noEmit` using the project's compiler settings.
- JavaScript: configured linting/type analysis where applicable.

Record “not checked” separately from “passed.” A missing type configuration is not a type-check pass.

**Gate 3 — Runtime and functional behavior**

- Run relevant unit tests where available.
- Use small targeted tests for generated examples.
- Run only in isolated sandboxes.
- Apply CPU, memory, wall-time, process, filesystem, and network restrictions.
- Capture exit code, stdout, stderr, timeout, and resource use.

**Gate 4 — Context and usefulness**

Assess whether the completion:
- Fits the cursor location.
- Uses existing names and imports correctly.
- Respects indentation and scope.
- Preserves surrounding code.
- Completes the intended construct.
- Stops at a useful boundary.
- Avoids unnecessary rewrites.

### 9.2 Hard syntax acceptance policy

For any candidate labeled or returned as “syntax-valid,” the reconstructed source must pass the relevant parser/checker. Invalid candidates must not be silently counted as valid.

In interactive autocomplete, do not introduce a slow test-suite execution step for every keystroke. Use fast parsing/incremental syntax checks for interactive feedback. Use type checking, sandboxed execution, and full tests in offline evaluation, training-data verification, and optional deeper validation workflows.

If the model produces an invalid candidate, the system may:
- Try another candidate.
- Retry with adjusted decoding.
- Use a repair prompt or constrained strategy if implemented and evaluated.
- Return no completion rather than knowingly presenting malformed code as validated.

Do not fabricate a successful validation result when a checker was unavailable.

---

## 10. Quality-improvement fine-tuning

### 10.1 Verified candidate generation

The original plan proposed 200,000 FIM prompts and 16 candidates per prompt. Treat those numbers as scale targets after a small pilot demonstrates that the process is affordable and useful.

For each prompt:

1. Generate multiple candidates.
2. Reconstruct the complete source.
3. Run syntax validation.
4. Run configured type checks.
5. Run relevant tests in an isolated sandbox.
6. Record each gate independently.
7. Retain high-quality examples with full provenance.
8. Deduplicate before fine-tuning.

### 10.2 Reward and ranking

Use a transparent, configurable ranking system. Syntax validity is a hard gate for examples labeled syntax-valid. Additional signals can include type-check success, test success, context fit, minimality, and usefulness.

Do not use a single binary “reward == 1” for all tasks if it discards useful distinctions. Preserve detailed per-gate results.

### 10.3 Fine-tuning

Use verified high-quality examples for supervised fine-tuning or rejection-sampling fine-tuning. Start conservatively, compare against the base checkpoint, and retain the base model for rollback.

A fine-tuned checkpoint is promoted only if held-out results improve without unacceptable regressions in another language or capability.

---

## 11. Evaluation framework

Evaluation must reflect actual editor use, not merely next-token loss or similarity to training data.

### 11.1 Required metrics

Report separately by Python, JavaScript, TypeScript, autocomplete, and FIM:

- Syntax-valid completion rate.
- Syntax error rate.
- Type-check pass rate where applicable.
- Test pass rate.
- Runtime crash/failure rate in the test harness.
- Functional correctness.
- Exact match (secondary metric only).
- Pass@1 and, where useful, Pass@k.
- Completion acceptance/usefulness in human review.
- Unnecessary completion length or over-generation.
- Context preservation.
- Time to first token (TTFT).
- End-to-end editor response latency.
- Tokens per second.
- Peak VRAM and system memory.

Do not combine syntax validity and runtime correctness into one number.

### 11.2 Benchmark design

Create a held-out benchmark with realistic editor states:

- Partial expressions.
- Function signatures awaiting bodies.
- Loops and conditionals.
- Nested scopes.
- Class methods.
- Imports and symbol references.
- Async/await code.
- Error-prone delimiters and indentation.
- JavaScript closures and callbacks.
- TypeScript interfaces, generics, and type annotations.
- FIM cases with meaningful suffix constraints.

Use repository-level tests where permitted. Prevent training/test leakage.

### 11.3 Candidate comparison

Compare every candidate model against:
- The previous checkpoint.
- The BF16 reference model.
- Relevant quantized variants.
- A documented baseline when available.

Use identical prompts, context, decoding settings, hardware conditions, and evaluation versions. Save raw results and aggregate reports.

### 11.4 Release quality gates

A release candidate must:
- Load reliably.
- Produce correct output format and special-token behavior.
- Have a measured syntax-validity rate.
- Have no unexplained regression in either primary language.
- Pass the platform's configured minimum test thresholds.
- Report latency and memory under repeatable conditions.
- Preserve the original BF16 checkpoint.

Set numerical release thresholds after the first reliable baseline benchmark. Do not invent benchmark scores in advance.

---

## 12. Quantization and automatic deployment selection

### 12.1 Required policy

**Benchmark all supported variants and select automatically.** Do not assume one quantization format is best.

- BF16 is the master/reference checkpoint.
- Test FP8 only if the actual Intel software stack and kernels support it correctly.
- Test Q8_0 only if the chosen inference backend supports the custom architecture and conversion path.
- Add other formats only when there is a verified implementation and a measurable reason to test them.

FP8 and Q8_0 are different formats and must be evaluated independently. Do not assume NVIDIA-specific FP8 acceleration is available on Intel XMX. Do not assume GGUF or llama.cpp supports the custom architecture until an end-to-end conversion and equivalence test succeeds.

### 12.2 Automatic benchmark pipeline

For each supported precision/quantization variant:

1. Convert from the immutable BF16 master checkpoint.
2. Verify model load and tokenizer/special-token compatibility.
3. Run numerical and generation smoke tests.
4. Run the same held-out autocomplete and FIM benchmark.
5. Run syntax, type, and functional checks.
6. Measure TTFT, end-to-end latency, tokens/sec, memory use, and stability.
7. Compare quality against BF16.
8. Reject variants that fail correctness, stability, or configured minimum-quality constraints.
9. Score the remaining variants using configurable objectives.
10. Select the best eligible variant for deployment.
11. Save the full report and selected-model rationale.
12. Support manual override and rollback.

### 12.3 Selection policy

The selection algorithm must be configurable. Since first-release priority is model quality, default scoring should prioritize correctness and quality. Speed and memory are optimization criteria among variants that meet configured quality and stability requirements.

Do not hard-code a universal 1%, 2%, or 5% quality-loss allowance. Expose quality constraints and weights in settings. The system must explain why a candidate was selected and which alternatives were rejected.

Never overwrite or delete the BF16 master checkpoint during quantization.

### 12.4 Dashboard

Include a Quantization & Optimization page showing:
- Available formats.
- Compatibility status.
- Conversion and benchmark progress.
- Syntax/type/test quality comparison.
- TTFT and throughput comparison.
- Memory usage.
- Selected deployment variant.
- Selection settings and scoring rationale.
- Historical benchmark runs.
- Manual selection and rollback.

---

## 13. Local inference service

### 13.1 Requirements

Build a local inference service with:

- OpenAI-compatible API where practical.
- Streaming responses.
- Health/status endpoint.
- Model and tokenizer metadata.
- Autocomplete endpoint.
- FIM endpoint.
- Configurable temperature, top-p, top-k, maximum tokens, and stop sequences.
- Cancellation and request timeouts.
- Context-length enforcement.
- Concurrent request handling appropriate to the hardware.
- Structured logs and latency metrics.

The model runtime must be chosen based on verified support for the custom architecture and Intel B70. Begin with a reliable PyTorch XPU inference path if necessary. Investigate optimized runtimes and exports only after equivalence tests pass.

### 13.2 Inference correctness

- Ensure FIM tokens are inserted and interpreted consistently.
- Prevent special tokens from leaking into visible completions.
- Apply stop sequences correctly.
- Test empty, short, long, and malformed contexts.
- Compare outputs before and after export/quantization.
- Test cancellation and repeated requests.
- Measure actual editor latency, not only synthetic tokens/sec.

### 13.3 Network and security defaults

- Bind to localhost by default.
- Require explicit configuration before exposing the server to a LAN or public network.
- Do not log source code or secrets by default.
- Provide an option to disable request-body logging entirely.
- Protect management endpoints with authentication.
- Keep API keys encrypted or in the operating system's secret store where available.

---

## 14. VS Code extension

Build a TypeScript VS Code extension for inline completions.

### 14.1 User experience

- Ghost-text inline suggestions.
- Tab to accept.
- Escape to dismiss.
- Manual trigger command.
- Automatic trigger with debounce.
- Configurable enable/disable.
- Configurable local server URL.
- Configurable completion length and latency behavior.
- Python, JavaScript, and TypeScript language support.
- FIM support when suffix context is available.
- Graceful handling of server unavailable, timeout, and cancellation.

### 14.2 Context collection

Send only the context needed for completion:

- Current language.
- Current file type.
- Text before cursor.
- Text after cursor when using FIM.
- Bounded nearby context.
- Optional relevant declarations/imports.
- Cursor and document version information for cancellation/stale-response handling.

Do not upload private source code to cloud services. The default extension path must communicate with the local inference server.

### 14.3 Editor safety

- Never modify existing source until the user accepts a completion.
- Do not overwrite newer document changes with stale results.
- Respect user cancellation.
- Handle incomplete editor buffers without treating every temporary parse error as a model failure.
- Do not run generated code inside the editor extension.

---

## 15. Management interface

Build a full management interface, initially as a local web application. Package as a desktop app later if practical.

### 15.1 Suggested stack

- Backend: Python with FastAPI.
- Frontend: React + TypeScript.
- Persistent metadata: SQLite initially.
- Background jobs: a worker/task system suitable for local execution.
- Training/inference: PyTorch XPU.
- Desktop packaging: evaluate Tauri after the web application is stable.

Keep components modular so the UI, training engine, and inference server can be developed and tested independently.

### 15.2 Required pages

1. **Overview** — hardware status, active model, training status, recent runs, key metrics.
2. **Hardware** — GPU/runtime diagnostics, memory, compatibility, benchmark history.
3. **Datasets** — sources, licenses, filtering, deduplication, manifests, splits, token counts.
4. **Teacher Models** — OpenRouter configuration, provider/model selection, prompts, API usage, synthetic-data jobs.
5. **Training** — configuration, start/stop/pause, progress, logs, GPU metrics, checkpoint management.
6. **Evaluation** — benchmarks, syntax/type/test results, model comparisons, regression reports.
7. **Models** — checkpoints, model metadata, promotion, rollback, tokenizer/config files.
8. **Quantization & Optimization** — variant generation, automatic benchmark/selection, manual override.
9. **Inference** — local API status, test prompt, streaming output, latency, memory.
10. **Settings** — accounts, authentication, paths, API keys, privacy, network access, logging, retention.

### 15.3 User accounts

Implement individual accounts with password authentication for the management interface.

- Store password hashes using a modern password-hashing algorithm; never store plaintext passwords.
- Provide secure session handling.
- Protect sensitive settings and API credentials.
- Default to a single-user local deployment configuration while preserving the ability to support multiple accounts.
- Do not expose the management interface publicly by default.

---

## 16. Data and execution safety

- Treat all downloaded code, repository content, prompts, and generated code as untrusted input.
- Never execute untrusted code directly on the host.
- Use isolated containers or another robust sandbox with resource limits.
- Disable network access in test sandboxes unless a test explicitly requires it and is separately approved.
- Use read-only input mounts and temporary output directories.
- Apply CPU, RAM, process, disk, and wall-clock limits.
- Prevent access to host credentials, SSH keys, browser profiles, and personal files.
- Record sandbox failures and timeouts distinctly from syntax/type failures.
- Keep teacher API keys out of source control and logs.
- Maintain license and provenance records for training data.

---

## 17. Architecture and repository structure

Use a modular monorepo or clearly versioned multi-package repository. A suggested starting layout:

```text
b70-code-platform/
  apps/
    dashboard/              # React + TypeScript UI
    vscode-extension/       # VS Code extension
    desktop/                # Optional later Tauri wrapper
  services/
    api/                    # FastAPI management API
    inference/              # Local model serving
    workers/                # Background jobs
  packages/
    model/                  # Transformer implementation/config
    tokenizer/              # Tokenizer training and runtime
    data/                   # Ingestion, filtering, packing
    training/               # Pretraining and fine-tuning
    evaluation/             # Syntax/type/test/quality benchmarks
    quantization/           # Conversion and variant benchmarking
    sandbox/                # Isolated validation/execution
    common/                 # Shared schemas/config/logging
  configs/
  scripts/
  tests/
  docs/
  artifacts/                # Local generated artifacts; not source controlled
  pyproject.toml
  README.md
```

The exact structure may evolve, but keep training, evaluation, serving, UI, and extension responsibilities separated.

---

## 18. Development phases and exit criteria

Build incrementally. Each phase must be runnable and tested before moving to the next. Do not generate the entire application in one unverified code-generation pass.

### Phase 0 — Repository and environment preflight

Deliver:
- Repository scaffold.
- Environment diagnostic.
- B70 XPU smoke tests.
- Dependency and runtime report.

Exit criteria:
- Hardware is correctly identified.
- XPU/BF16 checks pass or a documented blocker and fallback plan exists.
- A small forward/backward/optimizer/checkpoint test passes.

### Phase 1 — Model and tokenizer foundation

Deliver:
- Tokenizer implementation and tests.
- Model configuration and model implementation.
- Exact parameter-count report.
- CPU unit-test configuration.
- B70 forward/backward test.

Exit criteria:
- Tokenizer round-trip and special-token tests pass.
- Model shape/mask/save-load tests pass.
- Exact parameter count is reported and architecture is frozen.
- B70 smoke training step succeeds.

### Phase 2 — Data pipeline

Deliver:
- Source ingestion.
- License/provenance manifest.
- Filtering and deduplication.
- Syntax classification.
- FIM/causal transformations.
- Dataset packing and reproducible train/validation/test splits.

Exit criteria:
- Repeated runs with the same inputs/configuration produce reproducible manifests.
- No train/test overlap under configured deduplication.
- Sampled records decode correctly.
- Syntax and language labels are measured and reported.

### Phase 3 — Training

Deliver:
- Pretraining loop.
- Checkpoint/resume.
- Validation and metrics.
- GPU monitoring.
- Small-run and medium-run profiles.

Exit criteria:
- Training remains stable over a meaningful pilot.
- Interrupted run resumes correctly.
- Checkpoint contains all required state.
- Measured throughput and memory use are reported.
- Validation behavior is documented before scaling data volume.

### Phase 4 — Completion quality and fine-tuning

Deliver:
- Realistic autocomplete/FIM benchmark.
- Candidate generation and syntax/type/test verification.
- Verified-example store.
- Fine-tuning pipeline.
- Base-versus-fine-tuned comparisons.

Exit criteria:
- Syntax-validity is measured separately by language/task.
- Runtime and type outcomes are separate metrics.
- Fine-tuning demonstrates measurable held-out improvement before promotion.

### Phase 5 — Inference service

Deliver:
- Local inference server.
- Streaming API.
- Autocomplete and FIM endpoints.
- Latency/throughput instrumentation.

Exit criteria:
- Server starts locally and loads the model.
- API and cancellation tests pass.
- Model output and tokenizer behavior are verified.
- Realistic latency baseline is recorded.

### Phase 6 — Quantization and runtime optimization

Deliver:
- Supported model conversion paths.
- BF16 reference benchmark.
- Quantization variant benchmark pipeline.
- Automatic selection, manual override, and rollback.

Exit criteria:
- Every selectable variant passes load and correctness smoke tests.
- Same held-out benchmark is used across variants.
- Selected model has a recorded rationale.
- BF16 master remains intact.

### Phase 7 — VS Code extension

Deliver:
- Inline completion provider.
- FIM support.
- User acceptance/dismissal behavior.
- Settings and error handling.

Exit criteria:
- Suggestions appear in supported languages.
- Tab accepts and Escape dismisses.
- Stale responses do not overwrite newer edits.
- Local-only default behavior is verified.
- Syntax-validation behavior is measured.

### Phase 8 — Management dashboard and packaging

Deliver:
- Hardware, data, teacher, training, evaluation, model, quantization, inference, and settings pages.
- Authentication.
- Background job visibility.
- Local deployment documentation.
- Optional desktop packaging.

Exit criteria:
- User can manage the full local workflow from the UI.
- Sensitive credentials are protected.
- Jobs and errors are visible.
- Local deployment works from documented clean-install instructions.

---

## 19. Performance targets

The original concept listed:
- 1,000–1,400+ tokens/second single-stream decode.
- Less than 15 ms time to first token in VS Code.

These are **stretch goals, not guaranteed specifications**. Do not optimize for headline token throughput at the expense of syntax validity, completion usefulness, or functional correctness.

Measure:
- Model-only prefill and decode.
- End-to-end API latency.
- End-to-end VS Code latency.
- p50 and p95 latency.
- Cold and warm model startup.
- GPU and system memory.
- Single-user and concurrent-request behavior.

Record hardware, driver, runtime, model, quantization, context length, and benchmark version with every performance result.

---

## 20. Documentation and operational requirements

Maintain:

- Root README with setup and quick start.
- Architecture decision records.
- Hardware compatibility report.
- Model configuration and parameter-count report.
- Dataset/license/provenance documentation.
- Training runbooks.
- Evaluation methodology.
- Quantization benchmark reports.
- API documentation.
- VS Code extension setup guide.
- Troubleshooting guide.
- Security and sandboxing notes.
- Release notes and known limitations.

Every experiment must be reproducible from saved configuration and version identifiers.

---

## 21. Instructions for the coding agent (Codex / Claude Code)

You are implementing this specification as a real software project. Follow these rules:

1. Start by inspecting the repository and current machine. Do not assume the hardware or software stack works.
2. Implement one phase at a time in the order above.
3. Before writing a large subsystem, propose its file layout and interfaces, then implement it.
4. Write tests alongside implementation. Run relevant tests after every meaningful change.
5. Do not claim a test passed unless you actually ran it and observed success.
6. Do not claim GPU training if operations silently ran on CPU. Log and verify the device used.
7. Do not invent benchmark results, model capabilities, compatibility, or training-time estimates.
8. Keep the model architecture configurable and report its exact parameter count.
9. Preserve BF16 master checkpoints and all training state.
10. Treat syntactic correctness as a hard acceptance gate for candidates labeled syntax-valid.
11. Keep syntax validity, type correctness, and runtime/functional correctness as separate metrics.
12. Never execute untrusted generated code outside the sandbox.
13. Never send private project source to OpenRouter or another cloud service unless the user explicitly enables that workflow.
14. Keep secrets out of source control and logs.
15. Prefer small, verifiable milestones over broad, untested implementations.
16. If a dependency or backend is incompatible with Intel B70, document the evidence and evaluate alternatives instead of pretending it works.
17. Do not force GGUF, llama.cpp, Vulkan, FP8, or any other export/runtime path before proving compatibility with the custom architecture.
18. Keep a working baseline before attempting performance optimizations.
19. At the end of each phase, provide:
    - Files created/changed.
    - Commands executed.
    - Tests run and their actual results.
    - Hardware/device used.
    - Known limitations and blockers.
    - Exact next step.
20. Do not silently change product decisions in this document. If a change is necessary, explain the reason and update the specification and decision log.

### First task

Begin with **Phase 0 only**:

- Inspect the machine and repository.
- Create the initial project scaffold and environment diagnostic.
- Run the B70/XPU/BF16 preflight checks.
- Report the exact results and any blockers.

Do not begin full dataset collection or long-running training until Phase 0 passes.

---

## 22. Definition of success

The project succeeds when it provides a locally running coding assistant that:

- Understands Python, JavaScript, and TypeScript context.
- Completes the next useful line, statement, loop, function, or other code construct.
- Supports both inline autocomplete and FIM.
- Produces syntactically valid completions when presenting them as valid.
- Demonstrates measured type-checking and functional reliability improvements.
- Does not knowingly label invalid code as syntax-valid.
- Runs student training and inference locally on the Intel Arc Pro B70.
- Provides a usable VS Code extension.
- Provides a full management interface for the local development lifecycle.
- Automatically benchmarks supported quantization variants and selects the best eligible deployment model.
- Preserves reproducibility, provenance, security, and rollback capability.

**The primary measure of success is useful, context-aware, syntactically valid code completion—not the ability to reproduce training data and not raw tokens per second alone.**
