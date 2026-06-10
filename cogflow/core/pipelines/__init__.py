"""
Unified public API for CogFlow Pipelines.

This aggregates the user-facing functions from:
- pipelines.components
- pipelines.orchestration
"""

from uuid import UUID

# Only import submodules — NOT functions
from . import components, inspection, orchestration

# ===========================================================
# Public API (manually exposed)
# ===========================================================

# Pull functions we want to expose (namespaced, not imported)
cogcomponent = components.cogcomponent
register_component = components.register_component

pipeline = orchestration.pipeline
create_component_from_func = orchestration.create_component_from_func
client = orchestration.client
create_fl_pipeline = orchestration.create_fl_pipeline
create_fl_pipeline_dataspace = orchestration.create_fl_pipeline_dataspace
create_run_from_pipeline_func = orchestration.create_run_from_pipeline_func
kfp = orchestration.kfp

# Pipeline inspection (sync)
list_all_kfp_runs = inspection.list_all_kfp_runs
list_pipelines_by_name = inspection.list_pipelines_by_name
get_pipeline_task_sequence = inspection.get_pipeline_task_sequence
get_pipeline_task_sequence_by_run_id = inspection.get_pipeline_task_sequence_by_run_id
get_pipeline_task_sequence_by_run_name = inspection.get_pipeline_task_sequence_by_run_name
get_pipeline_task_sequence_by_pipeline_id = inspection.get_pipeline_task_sequence_by_pipeline_id
get_task_structure_by_task_id = inspection.get_task_structure_by_task_id

# Pod/K8s inspection (sync)
get_pod_definition = inspection.get_pod_definition
get_pod_events = inspection.get_pod_events
get_pod_logs = inspection.get_pod_logs
get_inference_service_logs = inspection.get_inference_service_logs

# Pod/K8s inspection (async)
async_get_pod_definition = inspection.async_get_pod_definition
async_get_pod_events = inspection.async_get_pod_events
async_get_pod_logs = inspection.async_get_pod_logs
async_get_inference_service_logs = inspection.async_get_inference_service_logs


# ===========================================================
# Unified load_component() API
# ===========================================================


def load_component(
    file_path: str = None,
    url: str = None,
    text: str = None,
    id: UUID = None,  # pylint: disable=redefined-builtin
):
    """Unified public component loader."""
    sources = [file_path, url, text, id]
    if sum(v is not None for v in sources) != 1:
        raise ValueError("Exactly one of file_path, url, text, or id must be provided.")

    # use submodule methods — NOT directly imported functions
    if file_path:
        base = orchestration.load_component_from_file(file_path)
    elif url:
        base = orchestration.load_component_from_url(url)
    elif text:
        base = orchestration.load_component_from_text(text)
    elif id:
        base = components.load_component_from_id(id)
        print(base)

    def wrapped(*args, **kwargs):
        op = base(*args, **kwargs)
        # Inject env only if this is a real container-backed op
        if hasattr(op, "add_env_variable"):
            orchestration._inject_env_into_container_op(op)

    wrapped.__signature__ = getattr(base, "__signature__", None)
    wrapped.component_spec = getattr(base, "component_spec", None)
    wrapped.__name__ = getattr(base, "__name__", "wrapped_component")

    return wrapped
