# Cosmos3-Edge on Ascend 910B

This deployment recipe complements [the Cosmos3-Edge model recipe](Cosmos3-Edge.md).
The action-policy evaluator and compatibility frontend are documented in
[the benchmark guide](../../benchmarks/cosmos3/README.md).

## Runtime and model requirements

Use aarch64 Ascend 910B hardware and the vLLM/vLLM-Ascend versions supported by this Omni revision.
The official baseline uses the 0.34 serving stack; an existing 0.33 environment must not be
silently reused as evidence that this revision runs. Follow the repository's Ascend installation
instructions for the selected revision, and validate driver/CANN/Torch-NPU compatibility.
Do not overwrite a known-good deployment environment while preparing a PR.

Model weights, CANN, Python environments and LIBERO task assets are external to Git.
Select the complete Diffusers checkpoint using `COSMOS3_MODEL`; record weight hashes and the
corresponding statistics file. The action checkpoint's raw dimension, rotation convention and
normalization statistics must agree.

## Dense Cosmos initialization

The diffusion worker supplies OmniDiffusionConfig to the platform initialization hook.
Only recognized dense Cosmos3 execution skips the unrelated worker-local AR `adapt_patch`.
Ascend configuration and LoRA registration remain initialized. Paged execution, unknown
configurations and other models retain their original initialization path.

## Action generation and evaluation

The model supports the optional per-action timestep and Euler/HAS schedules. The benchmark
expands its LIBERO presets into explicit prepared-input, frame-layout and schedule controls.
The original image/prompt preprocessing, independent NumPy noise streams, q01/q99 values,
precision settings and timestep scale must remain aligned for comparisons.

For incremental actions, enable `--diffusion-streaming-output` in request execution mode.
The model emits detached action-ready output; serving delivers optional HTTP callbacks.
Neither the model nor the diffusion worker sends HTTP requests. Non-media action events do
not suppress the final output. The compatibility frontend uses FastAPI/uvicorn and runs
blocking evaluation work outside its event loop.

For early stop, K means the first K action positions are ready. It stops the joint sampler;
requesting a complete video instead requires continuing denoising. File migration does not
change the default schedule, early-stop setting or return-video behavior.

## Rendering and reproducibility

CPU rendering can be used on the evaluator host. GPU rendering uses the preserved client
renderer on a separate NVIDIA host; its source can reside in the NPU checkout without being
executed on the NPU. Keep source and external LIBERO versions aligned across both hosts.

Before submitting, run regression tests and a real 910B smoke on the actual candidate
revision. Report skipped/unavailable checks explicitly. No performance or numerical-equivalence
claim follows from a clean three-way merge or successful syntax compilation.
