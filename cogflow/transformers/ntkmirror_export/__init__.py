"""
ntkmirror → PEFT LoRA conversion.

Public entry point: :func:`controller_to_lora`. Converts a trained
ntkmirror ``SignedLogMaskState`` controller into a standard PEFT
``adapter_model.safetensors`` + ``adapter_config.json`` pair that vLLM's
``--enable-lora`` path serves natively.

The conversion is the **post-residual approximation** from
:file:`/home/ali/.claude/plans/we-have-big-desigin-linear-dijkstra.md`
§Phase 1.C: for each gated (layer i, channel c) pair with effective gate
``g = exp(max_log_gate · tanh(raw))``, scale row c of layer i's
``o_proj`` and ``mlp.down_proj`` by ``g - 1``. Ignores the residual
``(g − 1) ⊙ h_{i−1}`` term — bounded error, see plan §C and the §E
benchmark for the empirical magnitude. If the benchmark misses the
threshold, the exporter switches to ntkmirror's pre-residual-hook
variant per plan open Q #5.
"""

from cogflow.transformers.ntkmirror_export.exporter import controller_to_lora

__all__ = ["controller_to_lora"]
