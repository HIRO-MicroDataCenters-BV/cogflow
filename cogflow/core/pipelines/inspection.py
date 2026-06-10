"""
CogFlow - Pipeline & Pod Inspection
------------------------------------

Provides pipeline querying (task sequences, runs) and Kubernetes pod
inspection (logs, events, definitions) functionality.

Replaces the 1.x NotebookPlugin with clean module-level functions.
Offers both sync and async variants for K8s operations.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from ...utils import common
from ...utils.exceptions import CogflowConnectionError
from ...utils.logging import get_logger

logger = get_logger(__name__)


# ================================================================
# LAZY LOADERS
# ================================================================


def _load_k8s():
    """Lazy load sync Kubernetes client."""
    from kubernetes import client

    common.load_k8s_config()
    return client


def _load_k8s_async():
    """Lazy load async Kubernetes client."""
    from kubernetes_asyncio import client, config

    return client, config


def _kfp_client(
    api_url: str | None = None,
    skip_tls_verify: bool = False,
    session_cookies: str | None = None,
    namespace: str | None = None,
):
    """Create a KFP client using the orchestration module."""
    from . import orchestration

    return orchestration.client(
        api_url=api_url,
        skip_tls_verify=skip_tls_verify,
        session_cookies=session_cookies,
        namespace=namespace,
    )


def _convert_datetime(obj):
    """Recursively convert datetime objects to ISO strings."""
    if isinstance(obj, dict):
        return {k: _convert_datetime(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_datetime(i) for i in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def _parse_run(run) -> dict:
    """Parse a single KFP run into a dict."""
    duration = "N/A"
    if hasattr(run, "finished_at") and hasattr(run, "created_at"):
        if run.finished_at and run.created_at:
            try:
                delta = run.finished_at - run.created_at
                duration = str(delta).split(".")[0]
            except Exception:
                pass

    start_time = None
    if hasattr(run, "created_at") and run.created_at:
        try:
            start_time = run.created_at.isoformat() if isinstance(run.created_at, datetime) else str(run.created_at)
        except Exception:
            start_time = "N/A"

    return {
        "run_name": getattr(run, "name", getattr(run, "display_name", "Unknown")),
        "run_id": getattr(run, "id", getattr(run, "run_id", "Unknown")),
        "status": getattr(run, "status", "Unknown"),
        "duration": duration,
        "experiment_id": _extract_experiment_id(run),
        "start_time": start_time,
    }


def _traverse_workflow_nodes(nodes: dict, namespace: str) -> tuple:
    """
    Parse workflow nodes to extract pipeline name and task structure.

    Returns:
        tuple: (pipeline_workflow_name, task_structure_dict)
    """
    pipeline_workflow_name = None
    root_node_id = None

    for node_id, node_data in nodes.items():
        if node_data.get("type") == "DAG":
            pipeline_workflow_name = node_data.get("displayName")
            root_node_id = node_id
            break

    if not root_node_id:
        raise ValueError("Root DAG node not found in the pipeline run.")

    task_structure = {}

    def traverse(node_id, parent=None):
        node = nodes[node_id]
        task_info = {
            "id": node_id,
            "namespace": namespace,
            "podName": node_id,
            "name": node.get("displayName", ""),
            "inputs": node.get("inputs", {}).get("parameters", []),
            "outputs": node.get("outputs", []),
            "status": node.get("phase", "unknown"),
            "startedAt": node.get("startedAt", "unknown"),
            "finishedAt": node.get("finishedAt", "unknown"),
            "resourcesDuration": node.get("resourcesDuration", {}),
            "children": [],
        }

        if parent is None:
            task_structure[node_id] = task_info
        else:
            parent["children"].append(task_info)

        for child_id in node.get("children", []):
            traverse(child_id, task_info)

    traverse(root_node_id)
    return pipeline_workflow_name, task_structure


# ================================================================
# PIPELINE QUERY FUNCTIONS (sync)
# ================================================================


def list_all_kfp_runs(
    api_url: str | None = None,
    skip_tls_verify: bool = False,
    session_cookies: str | None = None,
    namespace: str | None = None,
) -> list[dict]:
    """
    List all KFP pipeline runs, handling pagination.

    Returns:
        list[dict]: Parsed run entries with run_name, run_id, status, etc.
    """
    parsed_runs = []
    next_page_token = None
    kfp_client_instance = _kfp_client(api_url, skip_tls_verify, session_cookies, namespace)

    while True:
        runs_response = kfp_client_instance.list_runs(page_token=next_page_token)
        if runs_response and runs_response.runs:
            for run in runs_response.runs:
                parsed_runs.append(_parse_run(run))
        next_page_token = getattr(runs_response, "next_page_token", None)
        if not next_page_token:
            break

    return parsed_runs


def list_pipelines_by_name(
    pipeline_name: str,
    api_url: str | None = None,
    skip_tls_verify: bool = False,
    session_cookies: str | None = None,
    namespace: str | None = None,
) -> dict:
    """
    List all versions and runs of a pipeline by name.

    Returns:
        dict: {pipeline_id, versions, runs}
    """
    kfp_client_instance = _kfp_client(api_url, skip_tls_verify, session_cookies, namespace)

    # Find pipeline ID by name
    pipeline_id = _get_pipeline_id_by_name(kfp_client_instance, pipeline_name)

    # Get versions
    versions_response = kfp_client_instance.list_pipeline_versions(pipeline_id)

    # Get runs for this pipeline
    run_list = _list_runs_by_pipeline_id(kfp_client_instance, pipeline_id)

    return {
        "pipeline_id": pipeline_id,
        "versions": getattr(versions_response, "versions", []),
        "runs": run_list,
    }


def get_pipeline_task_sequence_by_run_id(
    run_id: str,
    api_url: str | None = None,
    skip_tls_verify: bool = False,
    session_cookies: str | None = None,
    namespace: str | None = None,
) -> dict:
    """
    Get pipeline workflow and task sequence for a given run ID.

    Returns:
        dict: {run_id, pipeline_workflow_name, namespace, task_structure}
    """
    kfp_client_instance = _kfp_client(api_url, skip_tls_verify, session_cookies, namespace)
    run_details = kfp_client_instance.get_run(run_id)

    workflow_graph = json.loads(run_details.pipeline_runtime.workflow_manifest)
    ns = workflow_graph["metadata"]["namespace"]
    nodes = workflow_graph["status"]["nodes"]

    pipeline_workflow_name, task_structure = _traverse_workflow_nodes(nodes, ns)

    return {
        "run_id": run_id,
        "pipeline_workflow_name": pipeline_workflow_name,
        "namespace": ns,
        "task_structure": task_structure,
    }


def get_pipeline_task_sequence_by_run_name(
    run_name: str,
    api_url: str | None = None,
    skip_tls_verify: bool = False,
    session_cookies: str | None = None,
    namespace: str | None = None,
) -> dict:
    """
    Get pipeline task sequence by run name (resolves to run_id first).

    Returns:
        dict: {run_id, pipeline_workflow_name, namespace, task_structure}
    """
    kfp_client_instance = _kfp_client(api_url, skip_tls_verify, session_cookies, namespace)
    run_id = _get_run_id_by_name(kfp_client_instance, run_name)
    if not run_id:
        raise ValueError(f"Pipeline run with name '{run_name}' not found.")

    return get_pipeline_task_sequence_by_run_id(run_id, api_url, skip_tls_verify, session_cookies, namespace)


def get_pipeline_task_sequence_by_pipeline_id(
    pipeline_id: str,
    api_url: str | None = None,
    skip_tls_verify: bool = False,
    session_cookies: str | None = None,
    namespace: str | None = None,
) -> dict:
    """
    Get task sequence for the latest run of a pipeline by pipeline ID.

    Returns:
        dict: {pipeline_id, pipeline_workflow_name, namespace, task_structure}
    """
    kfp_client_instance = _kfp_client(api_url, skip_tls_verify, session_cookies, namespace)

    # Get latest run for this specific pipeline
    filter_str = json.dumps({"predicates": [{"key": "pipeline_id", "op": "EQUALS", "string_value": pipeline_id}]})
    runs = kfp_client_instance.list_runs(page_size=1, filter=filter_str)
    if not runs or not runs.runs:
        raise ValueError(f"No runs found for pipeline ID '{pipeline_id}'.")

    run_id = runs.runs[0].id
    result = get_pipeline_task_sequence_by_run_id(run_id, api_url, skip_tls_verify, session_cookies, namespace)
    result["pipeline_id"] = pipeline_id
    return result


def get_pipeline_task_sequence(
    pipeline_name: str | None = None,
    pipeline_workflow_name: str | None = None,
    api_url: str | None = None,
    skip_tls_verify: bool = False,
    session_cookies: str | None = None,
    namespace: str | None = None,
) -> dict:
    """
    Get pipeline task sequence by pipeline name or workflow name.

    Returns:
        dict: {pipeline_workflow_name, namespace, task_structure}
    """
    kfp_client_instance = _kfp_client(api_url, skip_tls_verify, session_cookies, namespace)

    if pipeline_name:
        pipeline_id = _get_pipeline_id_by_name(kfp_client_instance, pipeline_name)
        return get_pipeline_task_sequence_by_pipeline_id(
            pipeline_id, api_url, skip_tls_verify, session_cookies, namespace
        )

    if pipeline_workflow_name:
        # Find run by workflow name
        run_id = _get_run_id_by_workflow_name(kfp_client_instance, pipeline_workflow_name)
        if not run_id:
            raise ValueError(f"No run found for workflow name '{pipeline_workflow_name}'.")
        return get_pipeline_task_sequence_by_run_id(run_id, api_url, skip_tls_verify, session_cookies, namespace)

    raise ValueError("Either pipeline_name or pipeline_workflow_name must be provided.")


def get_task_structure_by_task_id(
    task_id: str,
    run_id: str,
    api_url: str | None = None,
    skip_tls_verify: bool = False,
    session_cookies: str | None = None,
    namespace: str | None = None,
) -> dict:
    """
    Get the task structure for a specific task within a run.

    Returns:
        dict: Task info with id, podName, name, inputs, outputs, status, etc.
    """
    kfp_client_instance = _kfp_client(api_url, skip_tls_verify, session_cookies, namespace)
    run_details = kfp_client_instance.get_run(run_id)

    workflow_graph = json.loads(run_details.pipeline_runtime.workflow_manifest)
    nodes = workflow_graph["status"]["nodes"]
    ns = workflow_graph["metadata"]["namespace"]

    if task_id not in nodes:
        raise ValueError(f"Task '{task_id}' not found in run '{run_id}'.")

    node = nodes[task_id]
    return {
        "id": task_id,
        "namespace": ns,
        "podName": task_id,
        "name": node.get("displayName", ""),
        "inputs": node.get("inputs", {}).get("parameters", []),
        "outputs": node.get("outputs", []),
        "status": node.get("phase", "unknown"),
        "startedAt": node.get("startedAt", "unknown"),
        "finishedAt": node.get("finishedAt", "unknown"),
        "resourcesDuration": node.get("resourcesDuration", {}),
    }


# ================================================================
# POD INSPECTION FUNCTIONS (sync)
# ================================================================


def get_pod_definition(podname: str, namespace: str | None = None) -> str:
    """
    Fetch pod definition as JSON string.

    Returns:
        str: JSON-formatted pod definition.
    """
    namespace = namespace or common.get_namespace()
    k8s_client = _load_k8s()
    v1 = k8s_client.CoreV1Api()

    try:
        pod = v1.read_namespaced_pod(name=podname, namespace=namespace)
        pod_dict = _convert_datetime(pod.to_dict())
        return json.dumps(pod_dict, indent=4)
    except k8s_client.exceptions.ApiException as e:
        if e.status == 404:
            return json.dumps({"error": f"Pod '{podname}' not found in namespace '{namespace}'"})
        raise


def get_pod_events(podname: str, namespace: str | None = None) -> dict:
    """
    Fetch Kubernetes events for a specific pod.

    Returns:
        dict: {podname, namespace, count, events: [...]}
    """
    namespace = namespace or common.get_namespace()
    k8s_client = _load_k8s()
    v1 = k8s_client.CoreV1Api()

    def to_iso(ts):
        return ts.isoformat() if ts else None

    try:
        events_list = v1.list_namespaced_event(namespace=namespace)
    except k8s_client.exceptions.ApiException as e:
        return {"podname": podname, "namespace": namespace, "count": 0, "events": [], "error": str(e)}

    filtered = []
    for ev in events_list.items or []:
        involved = getattr(ev, "involved_object", None)
        if not involved or involved.name != podname:
            continue

        first_ts = (
            getattr(ev, "first_timestamp", None)
            or getattr(ev, "event_time", None)
            or getattr(getattr(ev, "metadata", None), "creation_timestamp", None)
        )

        filtered.append(
            {
                "type": getattr(ev, "type", None),
                "reason": getattr(ev, "reason", None),
                "message": getattr(ev, "message", None),
                "count": getattr(ev, "count", 1),
                "firstTimestamp": to_iso(first_ts),
                "lastTimestamp": to_iso(getattr(ev, "last_timestamp", None)),
                "reportingComponent": getattr(ev, "reporting_component", None),
                "source": getattr(getattr(ev, "source", None), "component", None),
                "involvedKind": getattr(involved, "kind", None),
                "involvedName": getattr(involved, "name", None),
            }
        )

    return {"podname": podname, "namespace": namespace, "count": len(filtered), "events": filtered}


def get_pod_logs(
    pod_name: str,
    namespace: str | None = None,
    container_name: str | None = None,
) -> list[str]:
    """
    Fetch pod logs as a list of log lines.

    Returns:
        List[str]: Log lines (empty list if no logs available).
    """
    namespace = namespace or common.get_namespace()
    k8s_client = _load_k8s()
    v1 = k8s_client.CoreV1Api()

    # Auto-detect container name based on pod name patterns
    if not container_name:
        if "pipeline" in pod_name:
            container_name = "main"
        elif "predictor" in pod_name:
            container_name = "kserve-container"

    try:
        kwargs = {"name": pod_name, "namespace": namespace}
        if container_name:
            kwargs["container"] = container_name

        raw_logs = v1.read_namespaced_pod_log(**kwargs)
        return [line.strip() for line in raw_logs.split("\n") if line.strip()]
    except k8s_client.exceptions.ApiException as e:
        raise CogflowConnectionError(
            f"Failed to fetch pod logs for pod '{pod_name}' in namespace '{namespace}': {e}"
        ) from e


def get_inference_service_logs(
    inference_service_name: str,
    namespace: str | None = None,
    container_name: str = "kserve-container",
) -> list[dict[str, Any]]:
    """
    Fetch logs for all pods matching an InferenceService name.

    Returns:
        List[Dict]: One entry per pod with pod_name, namespace, and logs (list of lines).
    """
    namespace = namespace or common.get_namespace()
    k8s_client = _load_k8s()
    v1 = k8s_client.CoreV1Api()

    pods = v1.list_namespaced_pod(namespace=namespace)
    isvc_pods = [p for p in pods.items if inference_service_name in p.metadata.name]

    log_entries = []
    for pod in isvc_pods:
        pod_logs = get_pod_logs(
            pod_name=pod.metadata.name,
            namespace=namespace,
            container_name=container_name,
        )
        log_entries.append(
            {
                "pod_name": pod.metadata.name,
                "namespace": pod.metadata.namespace,
                "logs": pod_logs,
            }
        )

    return log_entries


# ================================================================
# ASYNC POD INSPECTION FUNCTIONS
# ================================================================

_async_k8s_loaded = False


async def _ensure_async_k8s():
    global _async_k8s_loaded
    if not _async_k8s_loaded:
        from kubernetes_asyncio import config as async_config

        try:
            async_config.load_incluster_config()
        except Exception:
            await async_config.load_kube_config()
        _async_k8s_loaded = True


async def async_get_pod_definition(podname: str, namespace: str | None = None) -> str:
    """Fetch pod definition (async)."""
    namespace = namespace or common.get_namespace()
    await _ensure_async_k8s()
    from kubernetes_asyncio import client as async_client

    v1 = async_client.CoreV1Api()
    try:
        pod = await v1.read_namespaced_pod(name=podname, namespace=namespace)
        pod_dict = _convert_datetime(pod.to_dict())
        return json.dumps(pod_dict, indent=4)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch pod: {e}"})


async def async_get_pod_events(podname: str, namespace: str | None = None) -> dict:
    """Fetch pod events (async)."""
    namespace = namespace or common.get_namespace()
    await _ensure_async_k8s()
    from kubernetes_asyncio import client as async_client

    v1 = async_client.CoreV1Api()

    def to_iso(ts):
        return ts.isoformat() if ts else None

    try:
        events_list = await v1.list_namespaced_event(namespace=namespace)
    except Exception as e:
        return {"podname": podname, "namespace": namespace, "count": 0, "events": [], "error": str(e)}

    filtered = []
    for ev in events_list.items or []:
        involved = getattr(ev, "involved_object", None)
        if not involved or involved.name != podname:
            continue
        first_ts = getattr(ev, "first_timestamp", None) or getattr(ev, "event_time", None)
        filtered.append(
            {
                "type": getattr(ev, "type", None),
                "reason": getattr(ev, "reason", None),
                "message": getattr(ev, "message", None),
                "count": getattr(ev, "count", 1),
                "firstTimestamp": to_iso(first_ts),
                "lastTimestamp": to_iso(getattr(ev, "last_timestamp", None)),
                "source": getattr(getattr(ev, "source", None), "component", None),
            }
        )

    return {"podname": podname, "namespace": namespace, "count": len(filtered), "events": filtered}


async def async_get_pod_logs(
    pod_name: str,
    namespace: str | None = None,
    container_name: str | None = None,
) -> list[str]:
    """Fetch pod logs as list of lines (async)."""
    namespace = namespace or common.get_namespace()
    await _ensure_async_k8s()
    from kubernetes_asyncio import client as async_client

    v1 = async_client.CoreV1Api()

    if not container_name:
        if "pipeline" in pod_name:
            container_name = "main"
        elif "predictor" in pod_name:
            container_name = "kserve-container"

    kwargs = {"name": pod_name, "namespace": namespace}
    if container_name:
        kwargs["container"] = container_name

    raw_logs = await v1.read_namespaced_pod_log(**kwargs)
    return [line.strip() for line in raw_logs.split("\n") if line.strip()]


async def async_get_inference_service_logs(
    inference_service_name: str,
    namespace: str | None = None,
    container_name: str = "kserve-container",
) -> list[dict[str, Any]]:
    """Fetch ISVC logs as structured list (async)."""
    namespace = namespace or common.get_namespace()
    await _ensure_async_k8s()
    from kubernetes_asyncio import client as async_client

    v1 = async_client.CoreV1Api()

    pods = await v1.list_namespaced_pod(namespace=namespace)
    isvc_pods = [p for p in pods.items if inference_service_name in p.metadata.name]

    log_entries = []
    for pod in isvc_pods:
        pod_logs = await async_get_pod_logs(
            pod_name=pod.metadata.name,
            namespace=namespace,
            container_name=container_name,
        )
        log_entries.append(
            {
                "pod_name": pod.metadata.name,
                "namespace": pod.metadata.namespace,
                "logs": pod_logs,
            }
        )

    return log_entries


# ================================================================
# INTERNAL HELPERS
# ================================================================


def _extract_experiment_id(run) -> str:
    """Extract experiment_id from a KFP run object."""
    refs = getattr(run, "resource_references", None)
    if refs:
        for ref in refs:
            key = getattr(ref, "key", None)
            if key and getattr(key, "type", None) == "EXPERIMENT":
                return getattr(key, "id", "")
    return ""


def _get_pipeline_id_by_name(kfp_client_instance, pipeline_name: str) -> str:
    """Find pipeline ID by name."""
    next_page_token = None
    while True:
        pipelines = kfp_client_instance.list_pipelines(page_size=100, page_token=next_page_token)
        if pipelines and pipelines.pipelines:
            for p in pipelines.pipelines:
                if p.name == pipeline_name:
                    return p.id
        next_page_token = getattr(pipelines, "next_page_token", None)
        if not next_page_token:
            break
    raise ValueError(f"Pipeline '{pipeline_name}' not found.")


def _list_runs_by_pipeline_id(kfp_client_instance, pipeline_id: str) -> list:
    """List all runs for a given pipeline ID."""
    parsed_runs = []
    next_page_token = None
    filter_str = json.dumps({"predicates": [{"key": "pipeline_id", "op": "EQUALS", "string_value": pipeline_id}]})
    while True:
        runs = kfp_client_instance.list_runs(page_token=next_page_token, filter=filter_str)
        if runs and runs.runs:
            for run in runs.runs:
                parsed_runs.append(_parse_run(run))
        next_page_token = getattr(runs, "next_page_token", None)
        if not next_page_token:
            break
    return parsed_runs


def _get_run_id_by_name(kfp_client_instance, run_name: str) -> str | None:
    """Find run ID by run name."""
    next_page_token = None
    while True:
        runs = kfp_client_instance.list_runs(page_size=100, page_token=next_page_token)
        if runs and runs.runs:
            for run in runs.runs:
                if run.name == run_name:
                    return run.id
        next_page_token = getattr(runs, "next_page_token", None)
        if not next_page_token:
            break
    return None


def _get_run_id_by_workflow_name(kfp_client_instance, workflow_name: str) -> str | None:
    """Find run ID by workflow name."""
    next_page_token = None
    while True:
        runs = kfp_client_instance.list_runs(page_size=100, page_token=next_page_token)
        if runs and runs.runs:
            for run in runs.runs:
                run_details = kfp_client_instance.get_run(run.id)
                try:
                    manifest = json.loads(run_details.pipeline_runtime.workflow_manifest)
                    if manifest.get("metadata", {}).get("name") == workflow_name:
                        return run.id
                except Exception:
                    continue
        next_page_token = getattr(runs, "next_page_token", None)
        if not next_page_token:
            break
    return None
