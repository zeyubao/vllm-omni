# LIBERO action and rendering client

Install this optional benchmark client in the evaluator or GPU-renderer environment.
It does not load the vLLM server package on import.
LIBERO code and assets are supplied separately, using `LIBERO_PYTHONPATH` when needed;
they are not embedded in the wheel. See the parent benchmark README for version alignment,
normalization statistics and the GPU/NPU execution split.

Use **Baseline** and **Improved Model** as the comparison names in documentation
and result descriptions. Improved Model refers to the HAS policy variant.
