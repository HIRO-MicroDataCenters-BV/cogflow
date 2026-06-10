"""
Kubeflow Pipeline Orchestration for CogFlow SDK
------------------------------------------------
This module provides:
    - lazy-loaded KFP interface
    - safe exception wrapping
    - component creation
    - pipeline decorator
    - FL pipeline builder (connectors + dataspace)
    - K8s service creation/deletion
    - CogContainer environment injection

No plugin structure. No circular imports. SDK-style API.
"""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Mapping

from ...utils.exceptions import (
    CogflowConnectionError,
    CogflowErrorHandler,
    CogflowPipelineError,
)
from ...utils.logging import get_logger

logger = get_logger(__name__)


# ================================================================
# LAZY LOADERS
# ================================================================


def _load_kfp():
    """Lazy import full KFP; returns (kfp, dsl, ContainerOp)."""
    import kfp
    from kfp import dsl

    return kfp, dsl


def kfp():
    """Return the lazy-loaded kfp module."""
    kfp_module, _ = _load_kfp()
    return kfp_module


def _load_kfp_components():
    import kfp

    return kfp.components


def _load_k8s_client():
    """Lazy load Kubernetes Core client + ApiException."""
    from kubernetes import client
    from kubernetes.client import ApiException

    return client, ApiException


def _load_k8s_models():
    """Lazy load Kubernetes typed models."""
    from kubernetes.client import ApiException, V1EnvVar

    return ApiException, V1EnvVar


def _load_config():
    from ...config import config

    return config


# ================================================================
# SAFE WRAPPER
# ================================================================


def _safe_kfp_call(fn, context: str):
    """Wrap KFP calls into CogFlow unified exceptions."""
    api_exception, _ = _load_k8s_models()

    try:
        return fn()

    except api_exception as exc:
        CogflowErrorHandler.handle_exception(
            exc,
            context=context,
            raise_as=CogflowConnectionError,
            re_raise=True,
        )

    except Exception as exc:
        CogflowErrorHandler.handle_exception(
            exc,
            context=context,
            raise_as=CogflowPipelineError,
            re_raise=True,
        )


# ================================================================
# COG CONTAINER – supports env injection in-place
# ================================================================


def _inject_env_into_container_op(op):
    """Inject required CogFlow runtime environment variables."""
    api_exception, v1_env_var = _load_k8s_models()

    env_keys = [
        "DB_HOST",
        "DB_PORT",
        "DB_USER",
        "DB_PASSWORD",
        "DB_NAME",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "MINIO_BUCKET_NAME",
        "API_PATH",
        "MLFLOW_TRACKING_URI",
        "KF_PIPELINES_SA_TOKEN_PATH",
        "MINIO_ENDPOINT_URL",
        "MLFLOW_S3_ENDPOINT_URL",
    ]

    for key in env_keys:
        val = os.environ.get(key)
        if val:
            try:
                op.add_env_variable(v1_env_var(name=key, value=val))
                # logger.debug("Injected env var %s=%s", key, val)
            except Exception as e:
                logger.warning("Skipping env var injection for %s: %s", key, e)
        else:
            logger.debug("Env var not set: %s", key)

    return op


# ================================================================
# KFP CLIENT
# ================================================================


def client(
    api_url: str | None = None,
    skip_tls_verify: bool = True,
    session_cookies: str | None = None,
    namespace: str | None = None,
):
    """Create a Kubeflow Pipelines client with lazy loading."""
    logger.info("Creating KFP client api_url=%s namespace=%s", api_url, namespace)

    def _inner():
        kfp, _ = _load_kfp()

        # Internal cluster
        if not api_url and not session_cookies:
            return kfp.Client()

        # External cluster with cookies
        if session_cookies:
            client_cls = kfp.Client
            original_loader = client_cls._load_config

            def patched_loader(self, *args, **kwargs):
                cfg = original_loader(self, *args, **kwargs)
                cfg.verify_ssl = not skip_tls_verify
                return cfg

            client_cls._load_config = patched_loader

            return client_cls(
                host=api_url,
                cookies=session_cookies,
                namespace=namespace,
            )

        # External without cookies
        return kfp.Client(host=api_url, namespace=namespace)

    return _safe_kfp_call(_inner, "Create KFP client")


# ================================================================
# PIPELINE DECORATOR
# ================================================================


def pipeline(name: str = None, description: str = None):
    """Return the KFP dsl.pipeline decorator (lazy-loaded)."""

    logger.debug("Creating pipeline decorator name=%s", name)

    _, dsl = _load_kfp()

    def _inner():
        return dsl.pipeline(name=name, description=description)

    return _safe_kfp_call(_inner, "Create KFP pipeline decorator")


# ================================================================
# CREATE COMPONENT FROM FUNCTION
# ================================================================


def create_component_from_func(
    func,
    output_component_file: str | None = None,
    base_image: str | None = None,
    packages_to_install: list[str] | None = None,
    annotations: Mapping[str, str] | None = None,
):
    """Convert python function into a KFP component with env injection."""

    logger.info("Creating component from func=%s", func.__name__)
    cfg = _load_config()

    def _inner():
        components = _load_kfp_components()
        # Use provided base_image or fall back to global default
        resolved_image = base_image or cfg.COMP_BASE_IMAGE

        # Create raw comp
        comp = components.create_component_from_func(
            func=func,
            output_component_file=output_component_file,
            base_image=resolved_image,
            packages_to_install=packages_to_install,
            annotations=annotations,
        )

        # Wrapped component (runtime op)
        def wrapped(*args, **kwargs):
            op = comp(*args, **kwargs)

            # Only inject env vars if op is a real ContainerOp
            if hasattr(op, "add_env_variable"):
                _inject_env_into_container_op(op)

            return op

        wrapped.__signature__ = inspect.signature(comp)
        wrapped.component_spec = comp.component_spec
        return wrapped

    return _safe_kfp_call(_inner, "Create component from function")


# ================================================================
# LOAD COMPONENT
# ================================================================


def load_component_from_url(url: str):
    """ "
    Load a KFP component from a remote URL.
    """
    logger.info("Loading component from URL=%s", url)

    def _inner():
        components = _load_kfp_components()
        return components.load_component_from_url(url)

    return _safe_kfp_call(_inner, "Load KFP component from URL")


def load_component_from_file(path: str):
    """ "
    Load a KFP component from a local file."""
    logger.info("Loading component from file=%s", path)

    def _inner():
        components = _load_kfp_components()
        return components.load_component_from_file(path)

    return _safe_kfp_call(_inner, "Load KFP component from file")


def load_component_from_text(text: str):
    """ "
    Load a KFP component from raw YAML text."""
    logger.info("Loading component from text len=%s", len(text))

    def _inner():
        components = _load_kfp_components()
        return components.load_component_from_text(text)

    return _safe_kfp_call(_inner, "Load KFP component from text")


# ================================================================
# RUN MANAGEMENT
# ================================================================


def create_run_from_pipeline_func(
    pipeline_func,
    arguments: dict | None = None,
    run_name: str | None = None,
    experiment_name: str | None = None,
    namespace: str | None = None,
    pipeline_root: str | None = None,
    enable_caching: bool | None = None,
    service_account: str | None = None,
):
    """Create a KFP run safely."""

    logger.info("Creating run from pipeline=%s", pipeline_func.__name__)

    def _inner():
        return client().create_run_from_pipeline_func(
            pipeline_func=pipeline_func,
            arguments=arguments,
            run_name=run_name,
            experiment_name=experiment_name,
            namespace=namespace,
            pipeline_root=pipeline_root,
            enable_caching=enable_caching,
            service_account=service_account,
        )

    return _safe_kfp_call(_inner, "Create KFP run")


def delete_pipeline(pipeline_id: str, **client_kwargs):
    """Delete a pipeline by ID."""

    logger.info("Deleting pipeline_id=%s", pipeline_id)

    def _inner():
        return client(**client_kwargs).delete_pipeline(pipeline_id=pipeline_id)

    return _safe_kfp_call(_inner, "Delete pipeline")


# ================================================================
# KUBERNETES SERVICE MANAGEMENT
# ================================================================


def create_service(name: str) -> str:
    """Create a simple ClusterIP service with selector app=<name>."""
    logger.info("Creating Kubernetes service=%s", name)

    client, api_exception = _load_k8s_client()
    from ...utils.common import get_namespace, load_k8s_config

    load_k8s_config()
    namespace = get_namespace()

    svc = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(
            name=name,
            annotations={"service.alpha.kubernetes.io/app-protocols": '{"grpc":"HTTP2"}'},
        ),
        spec=client.V1ServiceSpec(
            selector={"app": name},
            ports=[client.V1ServicePort(protocol="TCP", port=8080, target_port=8080)],
            type="ClusterIP",
        ),
    )

    def _inner():
        api = client.CoreV1Api()
        return api.create_namespaced_service(namespace=namespace, body=svc)

    _safe_kfp_call(_inner, f"Create service {name}")
    return name


def delete_service(name: str):
    """Delete service by name."""
    logger.info("Deleting Kubernetes service=%s", name)

    client, api_exception = _load_k8s_client()
    from ...utils.common import get_namespace, load_k8s_config

    load_k8s_config()
    namespace = get_namespace()

    def _inner():
        api = client.CoreV1Api()
        return api.delete_namespaced_service(name=name, namespace=namespace)

    return _safe_kfp_call(_inner, f"Delete service {name}")


def _create_service_component(name: str) -> str:
    create_service(name)
    return name


def _delete_service_component(name: str) -> str:
    delete_service(name)
    return name


# ================================================================
# PARAMETER UTIL
# ================================================================


def _valid_param_names(sig):
    """Return only real parameters, no *args/**kwargs."""
    return [
        name
        for name, p in sig.parameters.items()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]


# ================================================================
# FEDERATED LEARNING PIPELINE
# ================================================================


def create_fl_pipeline(
    fl_client,
    fl_server,
    connectors: list,
    node_enforce: bool = True,
    pipeline_name: str | None = None,
    description: str | None = None,
):
    """
    Auto-generate a Federated Learning pipeline for standard connectors.

    Args:
        fl_client:   KFP component for FL client
        fl_server:   KFP component for FL server
        connectors:  List of connector objects with 'link' and 'region' attributes
        node_enforce (bool): Enforce node selector for region.
        pipeline_name (str, optional): Custom pipeline name.
        description (str, optional): Pipeline description text.
    Returns:
        Callable KFP pipeline function decorated with @dsl.pipeline
    """

    # Lazy load KFP
    kfp, dsl = _load_kfp()
    cfg = _load_config()

    # Provide sane defaults
    pipeline_name = pipeline_name or "Federated Learning Pipeline"
    description = description or "Auto-generated FL pipeline"

    # -------------------------------------------------------------------
    # 1) Inspect the function signatures
    # -------------------------------------------------------------------
    client_sig = inspect.signature(fl_client)
    server_sig = inspect.signature(fl_server)

    client_req = {"server_address", "local_data_connector"}
    server_req = {"number_of_iterations"}

    client_params = _valid_param_names(client_sig)
    server_params = _valid_param_names(server_sig)

    client_extra = [p for p in client_params if p not in client_req]
    server_extra = [p for p in server_params if p not in server_req]
    extra_params = list(dict.fromkeys(client_extra + server_extra))  # preserve order

    # -------------------------------------------------------------------
    # 2) Build the dynamic pipeline signature
    # -------------------------------------------------------------------
    sig_params = [
        inspect.Parameter(
            name="number_of_iterations",
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=int,
        )
    ]

    for name in extra_params:
        param = client_sig.parameters.get(name, server_sig.parameters.get(name))
        default = param.default if param.default is not inspect._empty else inspect._empty
        ann = param.annotation if param.annotation is not inspect._empty else None

        sig_params.append(
            inspect.Parameter(
                name=name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=ann,
                default=default,
            )
        )

    pipeline_sig = inspect.Signature(parameters=sig_params)

    # -------------------------------------------------------------------
    # 3) Convert service create/delete to reusable components
    # -------------------------------------------------------------------
    setup_links = create_component_from_func(_create_service_component, base_image=cfg.FL_LINKS_BASE_IMAGE)

    release_links = create_component_from_func(_delete_service_component, base_image=cfg.FL_LINKS_BASE_IMAGE)

    # -------------------------------------------------------------------
    # 4) Actual pipeline implementation
    # -------------------------------------------------------------------
    def fl_pipeline_func(*args, _node_enforce=node_enforce, **kwargs):
        # Bind incoming args to our generated signature
        bound = fl_pipeline_func.__signature__.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        args_map = bound.arguments

        number_of_iterations = args_map["number_of_iterations"]

        server_kwargs = {k: args_map[k] for k in server_extra}
        client_kwargs = {k: args_map[k] for k in client_extra}

        # Unique per-run service name
        srv_name = "flserver-" + "{{workflow.uid}}"

        # Create and delete service
        setup_task = setup_links(name=srv_name)
        cleanup_task = release_links(name=srv_name)

        # ------------------- Pipeline steps -------------------------
        with dsl.ExitHandler(cleanup_task):
            # FL server
            server_task = fl_server(
                number_of_iterations=number_of_iterations,
                **server_kwargs,
            ).after(setup_task)

            server_task.add_pod_label(name="app", value=srv_name)

            # FL clients fan-out
            for conn in connectors:
                region = getattr(conn, "region", "")
                link = getattr(conn, "link", None)

                if not link:
                    raise ValueError(f"Connector '{conn}' missing 'link' attribute")

                op = fl_client(
                    server_address=setup_task.output,
                    local_data_connector=link,
                    **client_kwargs,
                ).after(setup_task)

                if _node_enforce:
                    op.add_node_selector_constraint("region", region)

                op.set_display_name(f"client:{region}")

    # Attach dynamic signature
    fl_pipeline_func.__signature__ = pipeline_sig

    # -------------------------------------------------------------------
    # 5) Return the decorated KFP pipeline
    # -------------------------------------------------------------------
    return dsl.pipeline(
        name=pipeline_name,
        description=description,
    )(fl_pipeline_func)


# ================================================================
# FEDERATED LEARNING PIPELINE (DATASPACE)
# ================================================================


def _normalize_data_products(data_input) -> list[dict]:
    """
    Normalize dataspace data-product metadata into a flat list of
    ``{"region": ..., "access_url": ...}`` dicts.

    Accepts any of:
      - a JSON string,
      - a single dict or a list of dicts,
      - items already flattened to ``{"region", "access_url"}``, or
      - raw catalog items carrying ``region`` plus a ``distribution`` list
        whose entries hold ``accessURL``.

    Already-flat items pass through unchanged, so the function is idempotent
    and safe to call regardless of which shape the caller supplies.
    """
    if isinstance(data_input, str):
        try:
            data = json.loads(data_input)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON string for data products: {exc}") from exc
    else:
        data = data_input

    if data is None:
        return []
    if isinstance(data, Mapping):
        data = [data]
    elif not isinstance(data, (list, tuple)):
        raise ValueError(
            "Data products must be a mapping, a list/tuple of mappings, or a JSON string of the same."
        )
    normalized: list[dict] = []
    for item in data:
        if not isinstance(item, Mapping):
            continue
        region = item.get("region", "")
        access_url = item.get("access_url")
        if access_url:
            # Already-flat form — pass through unchanged.
            normalized.append({"region": region, "access_url": access_url})
            continue
        # Raw catalog form — pull URLs out of ``distribution``.
        for dist in item.get("distribution", []) or []:
            dist_url = dist.get("accessURL") if isinstance(dist, Mapping) else None
            if region and dist_url:
                normalized.append({"region": region, "access_url": dist_url})

    return normalized


def create_fl_pipeline_dataspace(
    fl_client,
    fl_server,
    data_products: list,
    node_enforce=True,
    pipeline_name: str | None = None,
    description: str | None = None,
):
    """
    Auto-generate a Federated Learning pipeline for dataspace integrations.

    Args:
        fl_client:   KFP component for FL client
        fl_server:   KFP component for FL server
        data_products: Either already-flat dicts
                       ``{ "access_url": "<url>", "region": "<region>" }`` or
                       raw catalog items (``region`` + ``distribution[].accessURL``),
                       or a JSON string of the same. Normalized internally.
        node_enforce (bool): Enforce node selector for region.
        pipeline_name (str, optional): Custom pipeline name.
        description (str, optional): Pipeline description text.

    Returns:
        Callable KFP pipeline function decorated with @dsl.pipeline
    """
    kfp, dsl = _load_kfp()
    cfg = _load_config()

    # Accept raw catalog payloads (region + distribution[].accessURL) as well
    # as already-flat {region, access_url} lists, so callers that forward the
    # dataspace checkout response verbatim keep working.
    data_products = _normalize_data_products(data_products)

    # Default names
    pipeline_name = pipeline_name or "Federated Learning Pipeline (Dataspace)"
    description = description or "Auto-generated FL pipeline for dataspace environments"

    client_sig = inspect.signature(fl_client)
    server_sig = inspect.signature(fl_server)
    print(server_sig)

    client_req = {"server_address", "local_data_connector"}
    server_req = {"number_of_iterations"}

    client_params = _valid_param_names(client_sig)
    server_params = _valid_param_names(server_sig)

    client_extra = [p for p in client_params if p not in client_req]
    server_extra = [p for p in server_params if p not in server_req]
    extra_params = list(dict.fromkeys(client_extra + server_extra))

    sig_params = [
        inspect.Parameter(
            name="number_of_iterations",
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=int,
        )
    ]

    for name in extra_params:
        param = client_sig.parameters.get(name, server_sig.parameters.get(name))
        default = param.default if param.default is not inspect._empty else inspect._empty
        ann = param.annotation if param.annotation is not inspect._empty else None
        sig_params.append(
            inspect.Parameter(
                name=name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=ann,
            )
        )

    pipeline_sig = inspect.Signature(parameters=sig_params)

    setup_links = create_component_from_func(_create_service_component, base_image=cfg.FL_LINKS_BASE_IMAGE)

    release_links = create_component_from_func(_delete_service_component, base_image=cfg.FL_LINKS_BASE_IMAGE)

    def fl_pipeline_func(*args, _node_enforce=node_enforce, **kwargs):
        bound = fl_pipeline_func.__signature__.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        args_map = bound.arguments

        number_of_iterations = args_map["number_of_iterations"]
        server_kwargs = {k: args_map[k] for k in server_extra}
        client_kwargs = {k: args_map[k] for k in client_extra}

        srv_name = "flserver-" + "{{workflow.uid}}"

        setup_task = setup_links(name=srv_name)
        cleanup_task = release_links(name=srv_name)

        with dsl.ExitHandler(cleanup_task):
            server_task = fl_server(number_of_iterations=number_of_iterations, **server_kwargs).after(setup_task)
            server_task.add_pod_label(name="app", value=srv_name)

            for dp in data_products:
                access_url = dp.get("access_url")
                region = dp.get("region", "")

                op = fl_client(
                    server_address=setup_task.output,
                    local_data_connector=access_url,
                    **client_kwargs,
                ).after(setup_task)

                if _node_enforce:
                    op.add_node_selector_constraint("region", region)

                op.set_display_name(f"client:{region}")

    fl_pipeline_func.__signature__ = pipeline_sig

    return dsl.pipeline(
        name=pipeline_name,
        description=description,
    )(fl_pipeline_func)
