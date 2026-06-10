"""
Model-specific transformations cogflow ships as a permanent fork.

These are NOT KServe Transformer container concepts — the name covers
post-processing of model artifacts (e.g. converting an ntkmirror
``SignedLogMaskState`` controller into a standard PEFT LoRA adapter).
Lives here so prebuilt platform components have a single import path,
not so cogflow imposes them as required deps. Each submodule lazy-imports
its heavy dependencies (torch, safetensors, etc.) inside its functions
so ``import cogflow`` stays cheap.
"""
