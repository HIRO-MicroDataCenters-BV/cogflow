"""
ntkmirror → PEFT LoRA conversion.

Public entry point: :func:`controller_to_lora`. Converts a trained
ntkmirror ``SignedLogMaskState`` controller into a standard PEFT
``adapter_model.safetensors`` + ``adapter_config.json`` pair that vLLM's
``--enable-lora`` path serves natively.

The conversion is the **post-residual approximation**: for each gated
(layer i, channel c) pair with effective gate
``g = exp(max_log_gate · tanh(raw))``, scale row c of layer i's
``o_proj`` and ``mlp.down_proj`` by ``g - 1``. The residual-stream term
``(g − 1) ⊙ h_{i−1}`` is ignored, which makes the conversion an
approximation rather than an exact reconstruction. The empirical
magnitude of that approximation error is being measured downstream
(NTK feature plan, §E benchmark — Qwen2.5-0.5B + GSM8K, ~2-3% NLL
delta threshold). If a future iteration adopts a pre-residual-hook
variant of ntkmirror that admits exact conversion, the math fits
behind the same public surface and ``controller_to_lora`` can be
swapped without breaking consumers — but no such variant is
implemented here today.
"""

from cogflow.transformers.ntkmirror_export.exporter import controller_to_lora

__all__ = ["controller_to_lora"]
