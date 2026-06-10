"""
NTK controller → PEFT LoRA adapter exporter (post-residual approximation).

The math: an ntkmirror gate fires on the **post-residual** output of
decoder layer ``i``, multiplying channel ``c`` of ``h_i`` by

    g = exp(max_log_gate * tanh(raw))

The strict equivalence has a residual-stream term ``(g − 1) · h_{i-1}``
that no static weight modification can express (LayerNorm is non-linear
in the activation variance). The Phase 1 approximation ignores the
residual contribution and absorbs the gate into the **producing
modules' rows**:

    ΔW_o[c, :] = (g - 1) · W_o[c, :]      # rows of attention's o_proj
    ΔW_d[c, :] = (g - 1) · W_d[c, :]      # rows of FFN's down_proj

This factors as a rank-K_layer LoRA:

    A (K × d_in)  = stack of (g_k - 1) · W[c_k, :]
    B (d_out × K) = identity selector with B[c_k, k] = 1

so that ``B @ A`` exactly rebuilds the per-row scaling. PEFT's standard
``W' = W + (alpha/r) · B @ A`` is preserved by emitting
``lora_alpha = r``.

The output is written in HuggingFace PEFT's on-disk format
(``adapter_config.json`` + ``adapter_model.safetensors``) so the LoRA
loads on vLLM via ``--enable-lora`` with no platform-side awareness.

Heavy deps (``torch``, ``safetensors``) are lazy-imported inside the
function so ``import cogflow`` stays cheap for users that don't need
fine-tune export.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

DEFAULT_TARGET_MODULES: list[str] = ["o_proj", "down_proj"]


def _effective_gate(raw: float, max_log_gate: float) -> float:
    """ntkmirror's clamping: g = exp(max_log_gate * tanh(raw))."""
    return math.exp(max_log_gate * math.tanh(raw))


def _group_gates_by_layer(
    layer_indices: Sequence[int],
    channel_indices: Sequence[int],
    raws: Sequence[float],
    max_log_gate: float,
) -> dict[int, list[tuple[int, float]]]:
    """Return {layer_i: [(channel_c, g_minus_1), ...]} for each layer that
    has at least one gate. Skips channels where ``g`` is exactly 1 (raw=0
    yields ``g=1`` so the row's contribution is zero) — keeps the LoRA
    rank tight."""
    grouped: dict[int, list[tuple[int, float]]] = {}
    for layer_i, channel_c, raw in zip(
        layer_indices, channel_indices, raws, strict=True
    ):
        g = _effective_gate(float(raw), max_log_gate)
        delta = g - 1.0
        if delta == 0.0:
            continue
        grouped.setdefault(int(layer_i), []).append((int(channel_c), delta))
    return grouped


def _build_layer_lora(
    gated_channels: list[tuple[int, float]],
    weight,  # torch.Tensor (d_out × d_in)
    rank: int,
    torch_module,
):
    """Build A (rank × d_in) and B (d_out × rank) for one layer / module.

    For each gated channel ``c_k`` with scale ``delta_k``:
        A[k, :] = delta_k * weight[c_k, :]
        B[c_k, k] = 1.0

    Unused rows of A (when len(gated_channels) < rank) stay zero —
    PEFT consumes them as no-op rank slots.
    """
    d_out, d_in = weight.shape
    dtype = weight.dtype
    device = weight.device

    lora_a = torch_module.zeros(rank, d_in, dtype=dtype, device=device)
    lora_b = torch_module.zeros(d_out, rank, dtype=dtype, device=device)
    for slot, (channel_c, delta) in enumerate(gated_channels):
        lora_a[slot, :] = weight[channel_c, :] * delta
        lora_b[channel_c, slot] = 1.0
    return lora_a, lora_b


def controller_to_lora(
    *,
    controller_state: dict[str, Any],
    weight_lookup: Callable[[int, str], Any],
    output_dir: str | Path,
    base_model_name_or_path: str,
    target_modules: Iterable[str] | None = None,
    rank: int | None = None,
    save_safetensors: bool = True,
) -> Path:
    """Materialise a PEFT LoRA adapter from an ntkmirror controller.

    Args:
        controller_state: ``SignedLogMaskState`` as a dict — must carry
            ``layer_indices`` (list[int]), ``channel_indices``
            (list[int]), ``raw`` (list[float]), ``max_log_gate``
            (float). Compatible with ``SignedLogMaskState.to_dict()``
            from ntkmirror and with the JSON form
            ``SignedLogMaskState.save`` writes when ``safetensors``
            isn't available.
        weight_lookup: Callable ``(layer_i, module_name) -> Tensor``
            returning the base model's weight matrix for the requested
            module on the requested layer. The exporter doesn't load
            the base model itself — the caller (e.g. the kfp component)
            already holds the loaded weights and passes a thin closure.
            Module names match ``target_modules`` entries.
        output_dir: Directory to write ``adapter_model.safetensors`` and
            ``adapter_config.json`` into. Created if missing.
        base_model_name_or_path: HF id or local path of the base model,
            recorded in ``adapter_config.json`` so PEFT can find the
            base at load time.
        target_modules: PEFT target module names. Defaults to
            ``["o_proj", "down_proj"]`` (Llama / Qwen / Mistral
            conventions — both modules feed the residual stream).
        rank: LoRA rank to emit. Defaults to the max gates-per-layer
            across all gated layers. Layers with fewer gates get
            zero-padded A/B slots so every layer carries the same shape
            (PEFT's standard expectation).
        save_safetensors: When True (default) write
            ``adapter_model.safetensors`` (preferred). When False, write
            ``adapter_model.bin`` via ``torch.save``.

    Returns:
        Path to the written ``output_dir`` (containing both files).
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "controller_to_lora requires torch. Install with `pip install torch` or `pip install cogflow[ntkmirror]`."
        ) from exc

    target_module_list: list[str] = list(target_modules or DEFAULT_TARGET_MODULES)
    layer_indices = controller_state["layer_indices"]
    channel_indices = controller_state["channel_indices"]
    raws = controller_state["raw"]
    max_log_gate = float(controller_state["max_log_gate"])

    grouped = _group_gates_by_layer(
        layer_indices=layer_indices,
        channel_indices=channel_indices,
        raws=raws,
        max_log_gate=max_log_gate,
    )

    if not grouped:
        raise ValueError(
            "Controller has no effective gates (all raw values produce g=1). Refusing to write an empty LoRA adapter."
        )

    if rank is None:
        rank = max(len(channels) for channels in grouped.values())

    state_dict: dict[str, Any] = {}
    layers_to_transform: list[int] = sorted(grouped.keys())

    for layer_i, gated_channels in grouped.items():
        if len(gated_channels) > rank:
            raise ValueError(
                f"Layer {layer_i} has {len(gated_channels)} gated channels but rank={rank}; rebuild with a higher rank."
            )
        for module_name in target_module_list:
            weight = weight_lookup(layer_i, module_name)
            lora_a, lora_b = _build_layer_lora(
                gated_channels=gated_channels,
                weight=weight,
                rank=rank,
                torch_module=torch,
            )
            # PEFT v0.7+ key convention for Causal LM models. Older
            # versions consumed slightly different prefixes; this layout
            # matches what vLLM's LoRA loader expects.
            prefix = f"base_model.model.model.layers.{layer_i}.{_module_to_path(module_name)}"
            state_dict[f"{prefix}.lora_A.weight"] = lora_a
            state_dict[f"{prefix}.lora_B.weight"] = lora_b

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "auto_mapping": None,
        "base_model_name_or_path": base_model_name_or_path,
        "revision": None,
        "inference_mode": True,
        "r": rank,
        # lora_alpha = r preserves ΔW = B @ A exactly (PEFT applies
        # (alpha/r) scaling; alpha/r = 1 here).
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "fan_in_fan_out": False,
        "bias": "none",
        "modules_to_save": None,
        "target_modules": target_module_list,
        "layers_to_transform": layers_to_transform,
        "layers_pattern": None,
        "init_lora_weights": False,
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2))

    if save_safetensors:
        try:
            from safetensors.torch import save_file  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "save_safetensors=True requires the safetensors package. Install with `pip install safetensors`."
            ) from exc
        save_file(state_dict, str(out_dir / "adapter_model.safetensors"))
    else:
        torch.save(state_dict, out_dir / "adapter_model.bin")

    return out_dir


def _module_to_path(module_name: str) -> str:
    """Map a target-module short name onto the HF Llama/Qwen submodule path."""
    if module_name in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return f"self_attn.{module_name}"
    if module_name in {"gate_proj", "up_proj", "down_proj"}:
        return f"mlp.{module_name}"
    # Unknown module — emit verbatim and let PEFT validate.
    return module_name
