# Cosmos3 action-policy benchmark

This directory contains the LIBERO evaluator, input preparation, action normalization,
and a FastAPI compatibility frontend for existing `/predict*` clients.
Model execution uses the ordinary vLLM-Omni API server; no benchmark module is imported by the model.

The two comparison variants are **Baseline** and **Improved Model**. Baseline uses
constant Euler scheduling and returns the complete action chunk. Improved Model uses
HAS scheduling to return ready actions progressively.

## Dependencies

Use the vLLM/vLLM-Ascend release supported by the selected Omni revision.
The previous 0.33 deployment environment is not sufficient for the 0.34 official baseline.
Install the client from `benchmarks/cosmos3/client` in the evaluator environment.
LIBERO runtime code and task assets are external dependencies. Set `LIBERO_PYTHONPATH`
to the parent of an importable `libero` package, and preserve the exact LIBERO revision,
controller configuration, task assets and initial states used for the comparison.
A freshly selected upstream LIBERO release is not assumed numerically equivalent.

The default statistics file is `libero_native_frame_wise_relative_rot6d.json` in this directory.
It contains the original 10-dimensional q01/q99 values; override it with `--action-stats-path`
when evaluating another checkpoint. Do not mix statistics from different action conventions.

## Entry points

- `python -m benchmarks.cosmos3.policy_server --help`: compatibility frontend.
- `python -m benchmarks.cosmos3.smoke --help`: model/normalizer smoke.
- `python -m vllm_omni_cosmos3_client --help`: standalone evaluator after client installation.
- `bash benchmarks/cosmos3/run_worker.sh`: worker; configure `COSMOS3_MODEL`, Python and device first.
- **Baseline**: `bash benchmarks/cosmos3/run_server.sh` launches the constant-Euler policy.
- **Improved Model**: `bash benchmarks/cosmos3/run_server_faster.sh` launches the HAS policy.

The launchers enable request-mode `--diffusion-streaming-output` so action-ready events
can cross the worker boundary. Early stop and full video generation have distinct contracts:
a truncated denoising run does not produce a fully denoised video.

## GPU rendering

GPU rendering source is preserved under
`client/vllm_omni_cosmos3_client/rendering/`, including the renderer, bridge and transport.
Deploy the same source revision on the NVIDIA renderer host. Physics and action inference
remain on the evaluator/NPU host; the GPU host renders camera frames.
Use `run_renderer.sh` and `run_render_tunnel.sh` with explicit Python, device and port settings.
By default GLVND discovers installed EGL drivers. If selecting a vendor JSON explicitly,
set `__EGL_VENDOR_LIBRARY_FILENAMES` to an existing file on that GPU machine.
No machine-specific JSON, driver, uv executable or runtime archive is bundled here.

## Validation

Run model/action regressions under `tests/diffusion/cosmos3_actions` and client regressions
under `tests/benchmarks/cosmos3`. The hardware comparison must record the source revision,
checkpoint hashes, external LIBERO version, seed, precision and actual imported module paths.
Historical results from another checkout are not validation of a new PR revision.

## Acceptance and success quality

The acceptance evaluator retains the success-quality analysis added in the source snapshot.
Use `run_acceptance_suite.sh` for paired runs and retain the generated acceptance and quality
reports. A timing improvement alone does not establish success-rate or completion-quality parity.
