"""
Unified public API for CogFlow Pipelines.

This aggregates the user-facing functions from:
- pipelines.components
- pipelines.orchestration
"""

from uuid import UUID

# Only import submodules — NOT functions
from . import components
from . import orchestration

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
