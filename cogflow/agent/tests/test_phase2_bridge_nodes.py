"""Tests for the 7 Phase-2 bridge node factories.

Each test exercises ``to_callable`` directly with a fabricated IR node
rather than going through a full ``StateGraph`` compile, so the factory's
runtime behaviour is verified in isolation. Integration coverage (compiling
real marketplace fixtures) is in ``test_marketplace_*`` modules.
"""

from __future__ import annotations

from typing import Any

import pytest

from cogflow.agent.ir.model import IRNode
from cogflow.agent.nodes import (
    CustomFunctionFactory,
    ExecuteFlowFactory,
    HumanInputFactory,
    RetrieverFactory,
    UnsupportedNodeError,
    custom_function,
    execute_flow,
    http,
    human_input,
    iteration,
    loop,
    retriever,
)


def _ir(node_id: str, node_type: str, config: dict[str, Any] | None = None) -> IRNode:
    return IRNode(id=node_id, type=node_type, label=node_id, config=config or {})


def _require_langgraph_symbol(name: str) -> Any:
    """Skip the calling test unless ``langgraph.types.<name>`` is importable.

    ``pytest.importorskip("langgraph.types")`` only proves the module exists;
    older versions in the supported ``>=0.2,<0.5`` range may ship the module
    without ``Send`` or ``interrupt``. Both ``IterationFactory`` and
    ``HumanInputFactory`` already degrade gracefully when their specific
    symbol is missing, so the corresponding assertion-based tests should
    skip on that exact condition rather than crash.
    """
    try:
        mod = __import__("langgraph.types", fromlist=[name])
        return getattr(mod, name)
    except (ImportError, AttributeError):
        pytest.skip(f"langgraph.types.{name} not available in this langgraph build")


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


def test_loop_increments_counter_each_invocation():
    factory = loop(target="body", max_iterations=3)
    fn = factory.to_callable(_ir("loop_0", "loop", factory.config))

    state: dict[str, Any] = {}
    for expected in (1, 2, 3):
        update = fn(state)
        state.update(update)
        assert state["_loop_count__loop_0"] == expected


def test_loop_factory_persists_target_and_max_in_config():
    """The compiler reads ``loopTarget`` / ``loopMaxIterations`` from IR config."""
    factory = loop(target="body", max_iterations=7)
    assert factory.config.get("loopTarget") == "body"
    assert factory.config.get("loopMaxIterations") == 7


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------


def test_iteration_emits_send_per_item():
    _require_langgraph_symbol("Send")
    factory = iteration(over="items", body="worker")
    fn = factory.to_callable(_ir("iter_0", "iteration", factory.config))
    sends = fn({"items": ["a", "b", "c"]})
    assert isinstance(sends, list)
    assert len(sends) == 3
    # Each Send object has a ``node`` attribute (LangGraph's Send API).
    for send, expected_item in zip(sends, ["a", "b", "c"], strict=True):
        assert send.node == "worker"
        assert send.arg["_iteration_item"] == expected_item


def test_iteration_no_body_node_is_noop():
    factory = iteration(over="items")  # no body
    fn = factory.to_callable(_ir("iter_1", "iteration", factory.config))
    assert fn({"items": [1, 2]}) == {}


def test_iteration_body_id_not_in_runtime_ids_is_noop():
    """Compile-time ctx contract: body id must be in runtime_node_ids."""
    _require_langgraph_symbol("Send")
    factory = iteration(over="items", body="worker_missing")
    fn = factory.to_callable(
        _ir("iter_dangling", "iteration", factory.config),
        ctx={"runtime_node_ids": {"some_other_node"}},
    )
    # body id is unknown to the compiler → must NOT emit Send to a missing node.
    assert fn({"items": [1, 2, 3]}) == {}


def test_iteration_empty_or_missing_collection():
    # IterationFactory degrades to `{}` when ``langgraph.types.Send`` isn't
    # importable (older langgraph in the supported range), so gate the
    # ``== []`` assertion on Send actually being available.
    _require_langgraph_symbol("Send")
    factory = iteration(over="items", body="worker")
    fn = factory.to_callable(_ir("iter_2", "iteration", factory.config))
    assert fn({}) == []
    assert fn({"items": []}) == []


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class _StubResponse:
    def __init__(self, status: int, body: Any, raise_on_json: bool = False) -> None:
        self.status_code = status
        self._body = body
        self._raise = raise_on_json
        self.text = body if isinstance(body, str) else str(body)

    def json(self) -> Any:
        if self._raise:
            raise ValueError("not JSON")
        return self._body


class _StubHttpx:
    def __init__(self, response: _StubResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> _StubResponse:
        self.calls.append((method, url, kwargs))
        return self.response


def test_http_node_passes_through_response_json():
    stub = _StubHttpx(_StubResponse(200, {"ok": True}))
    factory = http(method="POST", url="https://example.test/{path}", body={"q": 1}, output_key="resp")
    fn = factory.to_callable(_ir("http_0", "http", factory.config), ctx={"httpx": stub})

    result = fn({"path": "search"})
    assert result == {"resp": {"status": 200, "body": {"ok": True}}}
    assert stub.calls == [("POST", "https://example.test/search", {"headers": {}, "timeout": 30.0, "json": {"q": 1}})]


def test_http_node_falls_back_to_text_when_not_json():
    stub = _StubHttpx(_StubResponse(204, "no content", raise_on_json=True))
    factory = http(method="GET", url="https://example.test")
    fn = factory.to_callable(_ir("http_1", "http", factory.config), ctx={"httpx": stub})
    result = fn({})
    assert result["http_response"]["status"] == 204
    assert result["http_response"]["body"] == "no content"


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class _StubRetriever:
    def __init__(self, docs: list[str]) -> None:
        self.docs = docs
        self.queries: list[str] = []

    def invoke(self, query: str) -> list[str]:
        self.queries.append(query)
        return self.docs


def test_retriever_node_returns_first_k_docs():
    stub = _StubRetriever(docs=["d1", "d2", "d3", "d4", "d5"])
    factory = retriever(retriever=stub, k=2, output_key="docs")
    fn = factory.to_callable(_ir("r_0", "retriever", factory.config))
    result = fn({"query": "hi"})
    assert result == {"docs": ["d1", "d2"]}
    assert stub.queries == ["hi"]


def test_retriever_no_live_retriever_is_passthrough():
    factory = retriever()  # nothing injected
    fn = factory.to_callable(_ir("r_1", "retriever", factory.config))
    assert fn({"query": "hi"}) == {}


def test_retriever_resolves_from_ctx_when_factory_is_bare():
    """JSON-loaded retriever nodes get their live object via ctx."""
    stub = _StubRetriever(docs=["doc"])
    bare = RetrieverFactory.from_ir(_ir("r_2", "retriever", {"retrieverQueryKey": "query"}))
    fn = bare.to_callable(_ir("r_2", "retriever", bare.config), ctx={"retriever": stub})
    assert fn({"query": "x"}) == {"documents": ["doc"]}


# ---------------------------------------------------------------------------
# CustomFunction
# ---------------------------------------------------------------------------


def test_custom_function_python_fn_runs():
    factory = custom_function(fn=lambda state: state["x"] * 2, output_key="out")
    fn = factory.to_callable(_ir("cf_0", "custom_function", factory.config))
    assert fn({"x": 5}) == {"out": 10}


def test_custom_function_fn_does_not_leak_name_into_python_body():
    """Regression: passing ``fn=`` must not stash fn.__name__ into the code field.

    Previously ``customFunctionPython`` was set to the function's name, which
    then got copied into ``_body`` on re-import. With ``allow_custom_code=True``
    that would ``exec`` the bare identifier and raise NameError (or worse,
    silently execute it).
    """

    def my_named_function(state):
        return state["x"] + 1

    factory = custom_function(fn=my_named_function)
    # Config must not carry the function name as executable code.
    assert factory.config.get("customFunctionPython") == ""
    assert factory.config.get("customFunctionName") == "my_named_function"

    # Re-hydrate via from_ir as if loaded from JSON, then verify the gated
    # exec path doesn't accidentally try to run the function name.
    bare = CustomFunctionFactory.from_ir(_ir("cf_name", "custom_function", dict(factory.config)))
    fn = bare.to_callable(_ir("cf_name", "custom_function", bare.config), ctx={"allow_custom_code": True})
    # Empty body + no live fn -> passthrough, NOT NameError.
    assert fn({"x": 5}) == {}


def test_custom_function_javascript_body_rejected():
    with pytest.raises(UnsupportedNodeError):
        custom_function(body="return 1", language="javascript")


def test_custom_function_body_is_gated_off_by_default():
    factory = custom_function(body="result = state['x'] + 1")
    fn = factory.to_callable(_ir("cf_1", "custom_function", factory.config))
    # Without allow_custom_code=True we don't exec the body.
    assert fn({"x": 5}) == {}


def test_custom_function_body_runs_when_gate_is_open():
    factory = custom_function(body="result = state['x'] + 1")
    fn = factory.to_callable(
        _ir("cf_2", "custom_function", factory.config),
        ctx={"allow_custom_code": True},
    )
    assert fn({"x": 5}) == {"custom_output": 6}


def test_custom_function_restricted_builtins_block_dangerous_names():
    factory = custom_function(body="result = open('/etc/passwd')")
    with pytest.raises(NameError):
        factory.to_callable(
            _ir("cf_3", "custom_function", factory.config),
            ctx={"allow_custom_code": True},
        )({"x": 0})


# ---------------------------------------------------------------------------
# HumanInput
# ---------------------------------------------------------------------------


def test_human_input_calls_interrupt(monkeypatch):
    _require_langgraph_symbol("interrupt")
    import langgraph.types as lgt

    captured: dict[str, Any] = {}

    def _fake_interrupt(payload: Any) -> str:
        captured["payload"] = payload
        return "user said yes"

    monkeypatch.setattr(lgt, "interrupt", _fake_interrupt)
    factory = human_input(prompt="Approve {action}?", output_key="reply")
    fn = factory.to_callable(_ir("hi_0", "human_input", factory.config))
    result = fn({"action": "delete"})
    assert result == {"reply": "user said yes"}
    assert captured["payload"] == {"prompt": "Approve delete?", "node": "hi_0"}


# ---------------------------------------------------------------------------
# ExecuteFlow
# ---------------------------------------------------------------------------


class _StubCompiledGraph:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        return {"echoed": payload.get("x")}


def test_execute_flow_invokes_supplied_graph():
    stub = _StubCompiledGraph()
    factory = execute_flow(graph=stub, input_keys=["x"], output_key="out")
    fn = factory.to_callable(_ir("ef_0", "execute_flow", factory.config))
    assert fn({"x": 42, "y": "ignored"}) == {"out": {"echoed": 42}}
    assert stub.calls == [{"x": 42}]


def test_execute_flow_resolves_from_ctx_when_factory_is_bare():
    """JSON-loaded ExecuteFlow nodes pick up their compiled child from ctx."""
    stub = _StubCompiledGraph()
    bare = ExecuteFlowFactory.from_ir(_ir("ef_1", "execute_flow", {"executeFlowId": "child"}))
    result = bare.to_callable(_ir("ef_1", "execute_flow", bare.config), ctx={"flows": {"child": stub}})({"x": 1})

    # With no explicit ``executeFlowInputKeys`` filter, the child sees the
    # caller's state dict verbatim and returns the echo-with-x payload.
    assert len(stub.calls) == 1
    assert stub.calls[0] == {"x": 1}
    assert result == {"subflow_result": {"echoed": 1}}


def test_execute_flow_no_live_graph_is_passthrough():
    factory = execute_flow(flow_id="missing")
    fn = factory.to_callable(_ir("ef_2", "execute_flow", factory.config))
    assert fn({"x": 1}) == {}


# ---------------------------------------------------------------------------
# Factory registry
# ---------------------------------------------------------------------------


def test_factory_registry_has_all_15_node_types():
    from cogflow.agent.nodes import FACTORY_BY_IR_TYPE

    expected = {
        "start",
        "llm",
        "agent",
        "tool",
        "condition",
        "condition_agent",
        "direct_reply",
        "loop",
        "iteration",
        "http",
        "retriever",
        "custom_function",
        "human_input",
        "execute_flow",
        "sticky_note",
    }
    assert set(FACTORY_BY_IR_TYPE.keys()) == expected


def test_custom_function_from_ir_rejects_js_body():
    """JSON-loaded CustomFunction with language=javascript must be refused."""
    from cogflow.agent.nodes import CustomFunctionFactory, UnsupportedNodeError

    ir_node = _ir("cf_js", "custom_function", {"customFunctionLanguage": "javascript"})
    with pytest.raises(UnsupportedNodeError):
        CustomFunctionFactory.from_ir(ir_node)


def test_human_input_reads_flowise_description_key(monkeypatch):
    """JSON-loaded HumanInput uses ``humanInputDescription``, not ``humanInputPrompt``."""

    _require_langgraph_symbol("interrupt")
    bare = HumanInputFactory.from_ir(_ir("hi_flowise", "human_input", {"humanInputDescription": "Approve?"}))
    fn = bare.to_callable(_ir("hi_flowise", "human_input", bare.config))

    # Use monkeypatch so the patch is undone after the test — direct
    # assignment to lgt.interrupt would leak across tests and could cause
    # order-dependent failures in test_human_input_calls_interrupt et al.
    import langgraph.types as lgt

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        lgt,
        "interrupt",
        lambda payload: captured.setdefault("payload", payload) or "ok",
    )
    fn({})
    assert captured["payload"]["prompt"] == "Approve?"


def test_http_url_format_falls_back_on_missing_state_key():
    stub = _StubHttpx(_StubResponse(200, {}))
    factory = http(method="GET", url="https://example.test/{missing}")
    fn = factory.to_callable(_ir("http_fb", "http", factory.config), ctx={"httpx": stub})
    fn({})  # No 'missing' key — must not raise
    assert stub.calls[0][1] == "https://example.test/{missing}"


def test_human_input_prompt_format_falls_back_on_missing_state_key(monkeypatch):
    _require_langgraph_symbol("interrupt")
    import langgraph.types as lgt

    captured: dict[str, Any] = {}
    monkeypatch.setattr(lgt, "interrupt", lambda p: captured.setdefault("p", p) or "x")
    factory = human_input(prompt="Hello {who}")
    fn = factory.to_callable(_ir("hi_fb", "human_input", factory.config))
    fn({})  # missing 'who'
    assert captured["p"]["prompt"] == "Hello {who}"


def test_from_ir_hydration_works_for_all_phase2_factories():
    """JSON-loaded graphs must hydrate without crashing for every bridge node."""
    for ir_type in ("loop", "iteration", "http", "retriever", "custom_function", "human_input", "execute_flow"):
        from cogflow.agent.nodes import FACTORY_BY_IR_TYPE

        ir_node = _ir(f"{ir_type}_0", ir_type)
        instance = FACTORY_BY_IR_TYPE[ir_type].from_ir(ir_node)
        callable_ = instance.to_callable(ir_node)
        assert callable(callable_)
