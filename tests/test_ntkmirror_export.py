"""
Tests for ``cogflow.transformers.ntkmirror_export``.

Math-only helpers (``_effective_gate``, ``_group_gates_by_layer``) are
testable without torch. The full ``controller_to_lora`` path needs
torch + safetensors; those tests skip when the deps aren't installed.
"""

from __future__ import annotations

import json
import math

import pytest

from cogflow.transformers.ntkmirror_export.exporter import (
    DEFAULT_TARGET_MODULES,
    _effective_gate,
    _group_gates_by_layer,
    _module_to_path,
    controller_to_lora,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_controller(
    *,
    layer_indices: list[int],
    channel_indices: list[int],
    raws: list[float],
    max_log_gate: float = 0.05,
) -> dict:
    return {
        "layer_indices": layer_indices,
        "channel_indices": channel_indices,
        "raw": raws,
        "max_log_gate": max_log_gate,
    }


# ---------------------------------------------------------------------------
# Math helpers — no torch dep
# ---------------------------------------------------------------------------


def test_effective_gate_at_raw_zero_is_one():
    """raw=0 → tanh(0)=0 → g=exp(0)=1 → gate is a no-op."""
    assert _effective_gate(0.0, 0.05) == 1.0


def test_effective_gate_is_within_max_log_gate_bound():
    """For any finite raw, |log(g)| ≤ max_log_gate (the controller's
    clamping invariant). Sample a range."""
    for raw in [-1e6, -2.0, -0.1, 0.0, 0.1, 2.0, 1e6]:
        g = _effective_gate(raw, 0.05)
        assert math.exp(-0.05) <= g <= math.exp(0.05)


def test_group_gates_collapses_by_layer_and_skips_unity_gates():
    """raw=0 gates produce g=1 → delta=0 → skipped (no LoRA contribution)."""
    grouped = _group_gates_by_layer(
        layer_indices=[0, 0, 1, 2, 2],
        channel_indices=[5, 7, 3, 1, 9],
        raws=[1.0, 0.0, 0.5, -0.5, 0.0],  # idx 1 and 4 are zero → skipped
        max_log_gate=0.05,
    )
    assert sorted(grouped.keys()) == [0, 1, 2]
    assert [c for c, _ in grouped[0]] == [5]  # idx 1 skipped
    assert [c for c, _ in grouped[1]] == [3]
    assert [c for c, _ in grouped[2]] == [1]  # idx 4 skipped


def test_group_gates_returns_signed_deltas():
    """Positive raw → g > 1 → delta > 0. Negative raw → g < 1 → delta < 0."""
    grouped = _group_gates_by_layer(
        layer_indices=[0, 0],
        channel_indices=[1, 2],
        raws=[1.0, -1.0],
        max_log_gate=0.05,
    )
    positive_delta = grouped[0][0][1]
    negative_delta = grouped[0][1][1]
    assert positive_delta > 0
    assert negative_delta < 0


def test_module_to_path_attention_and_ffn():
    assert _module_to_path("o_proj") == "self_attn.o_proj"
    assert _module_to_path("q_proj") == "self_attn.q_proj"
    assert _module_to_path("down_proj") == "mlp.down_proj"
    assert _module_to_path("gate_proj") == "mlp.gate_proj"
    # Unknown name flows through verbatim.
    assert _module_to_path("custom_lora") == "custom_lora"


def test_default_target_modules_match_post_residual_choice():
    """Both Attn and FFN contribute to the residual; gating one without
    the other would silently change behavior."""
    assert DEFAULT_TARGET_MODULES == ["o_proj", "down_proj"]


# ---------------------------------------------------------------------------
# controller_to_lora — needs torch + safetensors
# ---------------------------------------------------------------------------


@pytest.fixture
def torch_mod():
    return pytest.importorskip("torch")


@pytest.fixture
def safetensors_mod():
    return pytest.importorskip("safetensors")


def _weight_lookup_factory(torch_module, d_out=4, d_in=4):
    """Return a callable that synthesizes a deterministic weight per
    (layer, module) pair. Stable across calls so tests can reason
    about what should land in the LoRA matrices."""

    def _lookup(layer_i: int, module_name: str):
        # Deterministic per (layer, module): seed = layer_i * 100 +
        # (1 for o_proj, 2 for any other module). Pure invented
        # numbering — the only requirement is collision-free across
        # (layer, module) pairs and reproducible so test assertions
        # can recompute the synthesised weights.
        seed = layer_i * 100 + (1 if module_name == "o_proj" else 2)
        gen = torch_module.Generator().manual_seed(seed)
        return torch_module.randn(d_out, d_in, generator=gen, dtype=torch_module.float32)

    return _lookup


def test_controller_to_lora_writes_peft_files(torch_mod, safetensors_mod, tmp_path):
    """Smoke: with valid inputs, both adapter_config.json and
    adapter_model.safetensors land in output_dir."""
    controller = _make_controller(
        layer_indices=[0, 0, 1],
        channel_indices=[1, 2, 0],
        raws=[1.0, -1.0, 0.5],
    )
    out = controller_to_lora(
        controller_state=controller,
        weight_lookup=_weight_lookup_factory(torch_mod),
        output_dir=tmp_path,
        base_model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
    )
    assert out == tmp_path
    assert (tmp_path / "adapter_config.json").exists()
    assert (tmp_path / "adapter_model.safetensors").exists()


def test_controller_to_lora_config_records_layers_and_rank(torch_mod, safetensors_mod, tmp_path):
    """adapter_config.json should reflect target_modules, lora rank,
    and the set of touched layers so PEFT skips ungated layers."""
    controller = _make_controller(
        layer_indices=[0, 0, 1, 1, 1],
        channel_indices=[1, 2, 0, 3, 7],
        raws=[1.0, -1.0, 0.5, -0.5, 1.5],
    )
    controller_to_lora(
        controller_state=controller,
        weight_lookup=_weight_lookup_factory(torch_mod, d_out=8, d_in=8),
        output_dir=tmp_path,
        base_model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
    )

    cfg = json.loads((tmp_path / "adapter_config.json").read_text())
    assert cfg["peft_type"] == "LORA"
    assert cfg["task_type"] == "CAUSAL_LM"
    assert cfg["target_modules"] == ["o_proj", "down_proj"]
    assert sorted(cfg["layers_to_transform"]) == [0, 1]
    # Rank is the max gates-per-layer (layer 1 has 3 gates).
    assert cfg["r"] == 3
    # alpha = r preserves ΔW = B @ A exactly (no PEFT internal scaling).
    assert cfg["lora_alpha"] == cfg["r"]


def test_controller_to_lora_state_dict_reconstructs_row_scaling(torch_mod, safetensors_mod, tmp_path):
    """The math invariant: B @ A == ΔW where ΔW[c_k, :] = (g_k - 1) * W[c_k, :].

    Load the written safetensors and verify B @ A matches the
    expected row-scaling for a known input.
    """
    from safetensors.torch import load_file

    layer_idx = 0
    channels = [1, 3]
    raws = [1.5, -0.7]
    max_log_gate = 0.05

    controller = _make_controller(
        layer_indices=[layer_idx, layer_idx],
        channel_indices=channels,
        raws=raws,
        max_log_gate=max_log_gate,
    )
    lookup = _weight_lookup_factory(torch_mod, d_out=4, d_in=4)
    weight_o = lookup(layer_idx, "o_proj")

    controller_to_lora(
        controller_state=controller,
        weight_lookup=lookup,
        output_dir=tmp_path,
        base_model_name_or_path="dummy/x",
    )
    loaded = load_file(str(tmp_path / "adapter_model.safetensors"))

    prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj"
    lora_a = loaded[f"{prefix}.lora_A.weight"]
    lora_b = loaded[f"{prefix}.lora_B.weight"]

    # Expected ΔW from row scaling:
    expected = torch_mod.zeros_like(weight_o)
    for c, raw in zip(channels, raws, strict=True):
        g = math.exp(max_log_gate * math.tanh(raw))
        expected[c, :] = (g - 1.0) * weight_o[c, :]

    actual = lora_b @ lora_a
    assert torch_mod.allclose(actual, expected, atol=1e-6), (
        "LoRA reconstruction does not match expected row scaling. Math invariant broken."
    )


def test_controller_to_lora_emits_both_target_modules(torch_mod, safetensors_mod, tmp_path):
    """Each gated layer should have lora_A/lora_B for o_proj AND down_proj
    — both contribute to the residual."""
    from safetensors.torch import load_file

    controller = _make_controller(
        layer_indices=[2],
        channel_indices=[1],
        raws=[1.0],
    )
    controller_to_lora(
        controller_state=controller,
        weight_lookup=_weight_lookup_factory(torch_mod),
        output_dir=tmp_path,
        base_model_name_or_path="dummy/x",
    )
    loaded = load_file(str(tmp_path / "adapter_model.safetensors"))

    expected_prefixes = {
        "base_model.model.model.layers.2.self_attn.o_proj",
        "base_model.model.model.layers.2.mlp.down_proj",
    }
    actual_prefixes = {k.rsplit(".", 2)[0] for k in loaded.keys()}
    assert actual_prefixes == expected_prefixes


def test_controller_to_lora_pads_lower_rank_layers_with_zero_slots(torch_mod, safetensors_mod, tmp_path):
    """Layer A has 2 gates, layer B has 1 → both get rank=2 matrices with
    layer B's last slot zero-padded so PEFT shape is uniform."""
    from safetensors.torch import load_file

    controller = _make_controller(
        layer_indices=[0, 0, 1],
        channel_indices=[1, 2, 0],
        raws=[1.0, -1.0, 1.0],
    )
    controller_to_lora(
        controller_state=controller,
        weight_lookup=_weight_lookup_factory(torch_mod, d_out=4, d_in=4),
        output_dir=tmp_path,
        base_model_name_or_path="dummy/x",
    )
    loaded = load_file(str(tmp_path / "adapter_model.safetensors"))

    prefix = "base_model.model.model.layers.1.self_attn.o_proj"
    lora_a = loaded[f"{prefix}.lora_A.weight"]
    lora_b = loaded[f"{prefix}.lora_B.weight"]
    # Shape uniform across layers.
    assert lora_a.shape == (2, 4)
    assert lora_b.shape == (4, 2)
    # Layer 1 only has 1 real gate — the second rank slot is zero.
    assert torch_mod.all(lora_a[1] == 0)
    assert torch_mod.all(lora_b[:, 1] == 0)


def test_controller_to_lora_raises_when_all_gates_unity(torch_mod, safetensors_mod, tmp_path):
    """No effective gates → refuse to write an empty adapter."""
    controller = _make_controller(
        layer_indices=[0, 1, 2],
        channel_indices=[1, 2, 3],
        raws=[0.0, 0.0, 0.0],
    )
    with pytest.raises(ValueError, match="no effective gates"):
        controller_to_lora(
            controller_state=controller,
            weight_lookup=_weight_lookup_factory(torch_mod),
            output_dir=tmp_path,
            base_model_name_or_path="dummy/x",
        )


def test_controller_to_lora_raises_when_layer_exceeds_explicit_rank(torch_mod, safetensors_mod, tmp_path):
    """Caller-pinned rank that's too small → explicit error, not a
    silent truncation of gates."""
    controller = _make_controller(
        layer_indices=[0, 0, 0],
        channel_indices=[1, 2, 3],
        raws=[1.0, -1.0, 0.5],
    )
    with pytest.raises(ValueError, match="rank=1"):
        controller_to_lora(
            controller_state=controller,
            weight_lookup=_weight_lookup_factory(torch_mod),
            output_dir=tmp_path,
            base_model_name_or_path="dummy/x",
            rank=1,
        )


def test_controller_to_lora_raises_when_channel_index_out_of_range(torch_mod, safetensors_mod, tmp_path):
    """Channel index that exceeds the target module's d_out → explicit
    ValueError with a debuggable message. Catches controller/base
    shape mismatch early instead of raising an opaque torch
    IndexError deep inside the slot loop."""
    controller = _make_controller(
        layer_indices=[0],
        channel_indices=[999],  # weight_lookup factory makes d_out=4
        raws=[1.0],
    )
    with pytest.raises(ValueError, match="out of range"):
        controller_to_lora(
            controller_state=controller,
            weight_lookup=_weight_lookup_factory(torch_mod),
            output_dir=tmp_path,
            base_model_name_or_path="dummy/x",
        )


def test_controller_to_lora_detaches_and_moves_to_cpu_before_save(torch_mod, safetensors_mod, tmp_path):
    """``weight_lookup`` may return tensors that carry an autograd
    graph (fine-tune callers commonly hand back ``requires_grad=True``
    weights straight off the live model). The exporter must detach
    + move to CPU before ``save_file`` — otherwise we'd persist
    autograd state and ``safetensors.torch.save_file`` would refuse
    non-CPU tensors anyway.

    We can't trivially exercise the CUDA path on a CPU-only CI box,
    so this test covers the autograd-graph half of the same fix.
    Loading the file back round-trips successfully, which it would
    not if the saved tensors had carried grad metadata."""
    from safetensors.torch import load_file

    controller = _make_controller(
        layer_indices=[0],
        channel_indices=[1],
        raws=[1.0],
    )

    def _lookup_with_grad(layer_i: int, module_name: str):
        seed = layer_i * 100 + (1 if module_name == "o_proj" else 2)
        gen = torch_mod.Generator().manual_seed(seed)
        w = torch_mod.randn(4, 4, generator=gen, dtype=torch_mod.float32)
        # Attach an autograd graph the way a fine-tune caller's loop
        # would: pass through a differentiable op so the returned
        # tensor has ``grad_fn`` non-None.
        return w * w.new_ones(1, requires_grad=True)

    controller_to_lora(
        controller_state=controller,
        weight_lookup=_lookup_with_grad,
        output_dir=tmp_path,
        base_model_name_or_path="dummy/x",
    )
    # Round-trips cleanly — autograd state would have made the file
    # non-loadable (or never written by safetensors in the first place).
    loaded = load_file(str(tmp_path / "adapter_model.safetensors"))
    sample_key = next(iter(loaded.keys()))
    assert loaded[sample_key].device.type == "cpu"
    assert loaded[sample_key].requires_grad is False


def test_controller_to_lora_torch_bin_when_safetensors_disabled(torch_mod, tmp_path):
    """save_safetensors=False falls back to torch's pickle format."""
    controller = _make_controller(
        layer_indices=[0],
        channel_indices=[1],
        raws=[1.0],
    )
    controller_to_lora(
        controller_state=controller,
        weight_lookup=_weight_lookup_factory(torch_mod),
        output_dir=tmp_path,
        base_model_name_or_path="dummy/x",
        save_safetensors=False,
    )
    assert (tmp_path / "adapter_model.bin").exists()
    assert not (tmp_path / "adapter_model.safetensors").exists()
