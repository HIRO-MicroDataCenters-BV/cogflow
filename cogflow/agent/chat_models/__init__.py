"""Chat-model re-exports, one submodule per provider.

Named ``chat_models`` rather than ``models`` to avoid colliding with
cogflow's top-level ``cogflow.models`` namespace, which already exists
and serves the MLflow-tracked ML model registry. Two different concepts
("ML model" vs "LLM chat client") deserve two different names.

Each provider module lives behind an optional install extra so the
base ``cogflow[agent]`` install doesn't drag in every LangChain
provider package. See ``pyproject.toml`` for the available extras.
"""

__all__: list[str] = []
