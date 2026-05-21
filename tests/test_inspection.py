"""
Unit tests for cogflow.core.pipelines.inspection.

Mocks KFP client and Kubernetes CoreV1Api to verify:
- Pagination behavior
- Pipeline/run filtering
- Error handling
- Workflow node parsing
"""

import json
from unittest.mock import MagicMock

import pytest

import cogflow.core.pipelines.inspection as inspection
from cogflow.utils.exceptions import CogflowConnectionError

# ---------------------------------------------------------------------
# HELPERS / FAKES
# ---------------------------------------------------------------------


class FakeKey:
    def __init__(self, type_, id_):
        self.type = type_
        self.id = id_


class FakeRef:
    def __init__(self, key):
        self.key = key


class FakeRun:
    def __init__(self, name, run_id, status="Succeeded", experiment_id="exp-1"):
        self.name = name
        self.id = run_id
        self.status = status
        self.created_at = None
        self.finished_at = None
        self.resource_references = [FakeRef(FakeKey("EXPERIMENT", experiment_id))]


class FakeRunsResponse:
    def __init__(self, runs, next_token=None):
        self.runs = runs
        self.next_page_token = next_token


def _make_workflow_manifest(nodes_dict, namespace="ns1"):
    """Build a workflow manifest JSON string with the given nodes."""
    return json.dumps(
        {
            "metadata": {"namespace": namespace, "name": "wf-name"},
            "status": {"nodes": nodes_dict},
        }
    )


# ---------------------------------------------------------------------
# _parse_run / _extract_experiment_id
# ---------------------------------------------------------------------


def test_extract_experiment_id_from_resource_references():
    """experiment_id should be picked from resource_references with type=EXPERIMENT."""
    run = FakeRun("r1", "id-1", experiment_id="exp-42")
    assert inspection._extract_experiment_id(run) == "exp-42"


def test_extract_experiment_id_no_references_returns_empty():
    """No resource_references → empty string."""
    run = FakeRun("r1", "id-1")
    run.resource_references = []
    assert inspection._extract_experiment_id(run) == ""


def test_extract_experiment_id_skips_non_experiment_refs():
    """Non-EXPERIMENT refs should be ignored."""
    run = FakeRun("r1", "id-1")
    run.resource_references = [FakeRef(FakeKey("PIPELINE", "p-1"))]
    assert inspection._extract_experiment_id(run) == ""


def test_parse_run_basic():
    """_parse_run should extract name, id, status, experiment_id."""
    run = FakeRun("my-run", "run-123", status="Running", experiment_id="exp-9")
    parsed = inspection._parse_run(run)
    assert parsed["run_name"] == "my-run"
    assert parsed["run_id"] == "run-123"
    assert parsed["status"] == "Running"
    assert parsed["experiment_id"] == "exp-9"


# ---------------------------------------------------------------------
# _traverse_workflow_nodes
# ---------------------------------------------------------------------


def test_traverse_workflow_nodes_finds_root_dag():
    """Should locate the DAG root and traverse children."""
    nodes = {
        "root-1": {
            "type": "DAG",
            "displayName": "my-pipeline",
            "phase": "Succeeded",
            "children": ["task-1"],
        },
        "task-1": {
            "type": "Pod",
            "displayName": "step-one",
            "phase": "Succeeded",
            "children": [],
        },
    }
    name, structure = inspection._traverse_workflow_nodes(nodes, "ns1")
    assert name == "my-pipeline"
    assert "root-1" in structure
    assert structure["root-1"]["name"] == "my-pipeline"
    assert len(structure["root-1"]["children"]) == 1
    assert structure["root-1"]["children"][0]["name"] == "step-one"


def test_traverse_workflow_nodes_raises_when_no_dag():
    """Should raise ValueError when no DAG node is present."""
    nodes = {
        "task-1": {
            "type": "Pod",
            "displayName": "step-one",
            "phase": "Succeeded",
            "children": [],
        }
    }
    with pytest.raises(ValueError, match="Root DAG node not found"):
        inspection._traverse_workflow_nodes(nodes, "ns1")


# ---------------------------------------------------------------------
# list_all_kfp_runs (pagination)
# ---------------------------------------------------------------------


def test_list_all_kfp_runs_paginates(monkeypatch):
    """Should iterate all pages until next_page_token is None."""
    page1 = FakeRunsResponse(
        runs=[FakeRun("r1", "id-1"), FakeRun("r2", "id-2")],
        next_token="token-2",
    )
    page2 = FakeRunsResponse(runs=[FakeRun("r3", "id-3")], next_token=None)

    fake_client = MagicMock()
    call_log = []

    def fake_list_runs(page_token=None):
        call_log.append(page_token)
        if page_token is None:
            return page1
        return page2

    fake_client.list_runs = fake_list_runs

    monkeypatch.setattr(inspection, "_kfp_client", lambda *a, **k: fake_client)

    runs = inspection.list_all_kfp_runs()
    assert len(runs) == 3
    assert [r["run_id"] for r in runs] == ["id-1", "id-2", "id-3"]
    assert call_log == [None, "token-2"]


# ---------------------------------------------------------------------
# Pipeline filtering — get_pipeline_task_sequence_by_pipeline_id
# ---------------------------------------------------------------------


def test_get_pipeline_task_sequence_by_pipeline_id_filters_by_pipeline(monkeypatch):
    """list_runs should be called with a pipeline_id filter."""
    fake_client = MagicMock()

    def fake_list_runs(page_size=None, filter=None, **_):
        # Verify filter contains the pipeline_id
        assert filter is not None
        parsed = json.loads(filter)
        assert parsed["predicates"][0]["key"] == "pipeline_id"
        assert parsed["predicates"][0]["string_value"] == "pid-42"
        return FakeRunsResponse(runs=[FakeRun("r1", "run-99")])

    fake_client.list_runs = fake_list_runs

    fake_run_details = MagicMock()
    fake_run_details.pipeline_runtime.workflow_manifest = _make_workflow_manifest(
        {
            "root": {
                "type": "DAG",
                "displayName": "p1",
                "phase": "Succeeded",
                "children": [],
            }
        }
    )
    fake_client.get_run = MagicMock(return_value=fake_run_details)

    monkeypatch.setattr(inspection, "_kfp_client", lambda *a, **k: fake_client)

    result = inspection.get_pipeline_task_sequence_by_pipeline_id("pid-42")
    assert result["pipeline_id"] == "pid-42"
    assert result["pipeline_workflow_name"] == "p1"


def test_get_pipeline_task_sequence_by_pipeline_id_raises_when_no_runs(monkeypatch):
    """Should raise ValueError when no runs found for the pipeline."""
    fake_client = MagicMock()
    fake_client.list_runs = MagicMock(return_value=FakeRunsResponse(runs=[]))
    monkeypatch.setattr(inspection, "_kfp_client", lambda *a, **k: fake_client)

    with pytest.raises(ValueError, match="No runs found"):
        inspection.get_pipeline_task_sequence_by_pipeline_id("pid-x")


# ---------------------------------------------------------------------
# _list_runs_by_pipeline_id (pipeline filter + pagination)
# ---------------------------------------------------------------------


def test_list_runs_by_pipeline_id_uses_filter():
    """The internal helper should pass the pipeline_id filter to list_runs."""
    fake_client = MagicMock()
    captured_filters = []

    def fake_list_runs(page_token=None, filter=None):
        captured_filters.append(filter)
        return FakeRunsResponse(runs=[FakeRun("r1", "id-1")])

    fake_client.list_runs = fake_list_runs

    runs = inspection._list_runs_by_pipeline_id(fake_client, "pid-99")
    assert len(runs) == 1
    assert captured_filters[0] is not None
    parsed = json.loads(captured_filters[0])
    assert parsed["predicates"][0]["key"] == "pipeline_id"
    assert parsed["predicates"][0]["string_value"] == "pid-99"


# ---------------------------------------------------------------------
# Pod inspection — error handling
# ---------------------------------------------------------------------


def test_get_pod_logs_raises_cogflow_connection_error_on_api_failure(monkeypatch):
    """get_pod_logs should wrap ApiException in CogflowConnectionError."""
    # Build a fake k8s_client that raises ApiException
    fake_k8s = MagicMock()

    class FakeApiException(Exception):
        pass

    fake_k8s.exceptions.ApiException = FakeApiException

    fake_v1 = MagicMock()
    fake_v1.read_namespaced_pod_log = MagicMock(side_effect=FakeApiException("boom"))
    fake_k8s.CoreV1Api = MagicMock(return_value=fake_v1)

    monkeypatch.setattr(inspection, "_load_k8s", lambda: fake_k8s)
    monkeypatch.setattr(inspection.common, "get_namespace", lambda: "test-ns", raising=True)

    with pytest.raises(CogflowConnectionError, match="Failed to fetch pod logs"):
        inspection.get_pod_logs("my-pod")


def test_get_pod_definition_returns_json(monkeypatch):
    """get_pod_definition should return JSON string of the pod spec."""
    fake_k8s = MagicMock()
    fake_v1 = MagicMock()

    class FakePod:
        def to_dict(self):
            return {"metadata": {"name": "p1"}, "spec": {"containers": []}}

    fake_v1.read_namespaced_pod = MagicMock(return_value=FakePod())
    fake_k8s.CoreV1Api = MagicMock(return_value=fake_v1)

    monkeypatch.setattr(inspection, "_load_k8s", lambda: fake_k8s)
    monkeypatch.setattr(inspection.common, "get_namespace", lambda: "test-ns", raising=True)

    result = inspection.get_pod_definition("p1")
    parsed = json.loads(result)
    assert parsed["metadata"]["name"] == "p1"


def test_get_pod_events_filters_by_pod_name(monkeypatch):
    """get_pod_events should only return events whose involvedObject matches podname."""
    fake_k8s = MagicMock()
    fake_v1 = MagicMock()

    class FakeInvolved:
        def __init__(self, name, kind="Pod"):
            self.name = name
            self.kind = kind

    class FakeEvent:
        def __init__(self, podname, reason="Started"):
            self.involved_object = FakeInvolved(podname)
            self.type = "Normal"
            self.reason = reason
            self.message = "msg"
            self.count = 1
            self.first_timestamp = None
            self.last_timestamp = None
            self.event_time = None
            self.metadata = MagicMock(creation_timestamp=None)
            self.reporting_component = None
            self.reporting_instance = None
            self.source = MagicMock(component=None)

    fake_events = MagicMock()
    fake_events.items = [
        FakeEvent("my-pod", "Started"),
        FakeEvent("other-pod", "Failed"),
        FakeEvent("my-pod", "Pulled"),
    ]
    fake_v1.list_namespaced_event = MagicMock(return_value=fake_events)
    fake_k8s.CoreV1Api = MagicMock(return_value=fake_v1)
    fake_k8s.exceptions.ApiException = Exception

    monkeypatch.setattr(inspection, "_load_k8s", lambda: fake_k8s)
    monkeypatch.setattr(inspection.common, "get_namespace", lambda: "test-ns", raising=True)

    result = inspection.get_pod_events("my-pod")
    assert result["count"] == 2
    assert result["podname"] == "my-pod"
    reasons = [e["reason"] for e in result["events"]]
    assert "Started" in reasons
    assert "Pulled" in reasons
    assert "Failed" not in reasons
