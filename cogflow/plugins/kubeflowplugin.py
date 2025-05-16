"""
This module provides functionality related to Kubeflow Pipelines.
"""

import functools
import inspect
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any, Mapping, Callable
from inspect import Signature, signature, Parameter
import kfp
from kfp import dsl
from kserve import (
    KServeClient,
    V1beta1InferenceService,
    V1beta1InferenceServiceSpec,
    V1beta1ModelFormat,
    V1beta1ModelSpec,
    V1beta1PredictorSpec,
    V1beta1SKLearnSpec,
    constants,
    utils,
)
from kubernetes import client, config
from kubernetes.client import V1ObjectMeta, V1ContainerPort
from kubernetes.client.models import V1EnvVar
from kubernetes.config import ConfigException
from tenacity import retry, wait_exponential, stop_after_attempt

from .. import plugin_config
from ..pluginmanager import PluginManager


class CogContainer(kfp.dsl._container_op.Container):
    """
    Subclass of Container to add model access environment variables.
    """

    def __init__(self, name=None, image=None, command=None, args=None, **kwargs):
        """
        Initializes the CogContainer class.
        """
        super().__init__(name=name, image=image, command=command, args=args, **kwargs)

    def add_model_access(self):
        """
        Adds model access environment variables to the container.

        Returns:
            CogContainer: Container instance with added environment variables.
        """
        # Verify plugin activation
        PluginManager().verify_activation(KubeflowPlugin().section)

        env_vars = [
            "DB_HOST",
            "DB_PORT",
            "DB_USER",
            "DB_PASSWORD",
            "DB_NAME",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "MINIO_BUCKET_NAME",
            "BASE_PATH",
            "MLFLOW_TRACKING_URI",
            "KF_PIPELINES_SA_TOKEN_PATH",
            "MINIO_ENDPOINT_URL",
            "MLFLOW_S3_ENDPOINT_URL",
        ]

        # Adding only environment variables present in the image
        for key in env_vars:
            value = os.environ.get(key)
            if value:
                self.add_env_variable(V1EnvVar(name=key, value=value))

        return self


class KubeflowPlugin:
    """
    Class for defining reusable components.
    """

    def __init__(self, image=None, command=None, args=None):
        """
        Initializes the KubeflowPlugin class.
        """
        self.kfp = kfp
        self.kfp.dsl._container_op.Container.AddModelAccess = (
            CogContainer.add_model_access
        )
        self.kfp.dsl._container_op.ContainerOp.AddModelAccess = (
            CogContainer.add_model_access
        )
        self.config_file_path = os.getenv(plugin_config.COGFLOW_CONFIG_FILE_PATH)
        self.v2 = kfp.v2
        self.section = "kubeflow_plugin"

    @staticmethod
    def pipeline(name=None, description=None):
        """
        Decorator function to define Kubeflow Pipelines.

        Args:
            name (str, optional): Name of the pipeline. Defaults to None.
            description (str, optional): Description of the pipeline. Defaults to None.

        Returns:
            Callable: Decorator for defining Kubeflow Pipelines.
        """
        # Verify plugin activation
        PluginManager().verify_activation(KubeflowPlugin().section)

        return dsl.pipeline(name=name, description=description)

    @staticmethod
    def create_component_from_func(
        func,
        output_component_file=None,
        base_image=None,
        packages_to_install=None,
        annotations: Optional[Mapping[str, str]] = None,
    ):
        """
        Create a component from a Python function.

        Args:
            func (Callable): Python function to convert into a component.
            output_component_file (str, optional): Path to save the component YAML file. Defaults
            to None.
            base_image (str, optional): Base Docker image for the component. Defaults to None.
            packages_to_install (List[str], optional): List of additional Python packages
            to install in the component.
            Defaults to None.
            annotations: Optional. Allows adding arbitrary key-value data to the component specification.
        Returns:
            kfp.components.ComponentSpec: Component specification.
        """
        # Verify plugin activation
        PluginManager().verify_activation(KubeflowPlugin().section)

        training_var = kfp.components.create_component_from_func(
            func=func,
            output_component_file=output_component_file,
            base_image=base_image,
            packages_to_install=packages_to_install,
            annotations=annotations,
        )

        def wrapped_component(*args, **kwargs):
            component_op = training_var(*args, **kwargs)
            component_op = CogContainer.add_model_access(component_op)
            return component_op

        wrapped_component.component_spec = training_var.component_spec
        return wrapped_component

    @staticmethod
    def client():
        """
        Get the Kubeflow Pipeline client.

        Returns:
            kfp.Client: Kubeflow Pipeline client instance.
        """
        # Verify plugin activation
        PluginManager().verify_activation(KubeflowPlugin().section)

        return kfp.Client()

    @staticmethod
    def load_component_from_url(url):
        """
        Load a component from a URL.

        Args:
            url (str): URL to load the component from.

        Returns:
            kfp.components.ComponentSpec: Loaded component specification.
        """
        # Verify plugin activation
        PluginManager().verify_activation(KubeflowPlugin().section)

        return kfp.components.load_component_from_url(url)

    @staticmethod
    def serve_model_v2(model_uri: str, name: str = None):
        """
        Create a kserve instance.

        Args:
            model_uri (str): URI of the model.
            name (str, optional): Name of the kserve instance. If not provided,
            a default name will be generated.

        Returns:
            None
        """
        # Verify plugin activation
        PluginManager().verify_activation(KubeflowPlugin().section)

        namespace = utils.get_default_target_namespace()
        if name is None:
            now = datetime.now()
            date = now.strftime("%d%M")
            name = f"predictormodel{date}"
        isvc_name = name
        predictor = V1beta1PredictorSpec(
            service_account_name="kserve-controller-s3",
            min_replicas=1,
            model=V1beta1ModelSpec(
                model_format=V1beta1ModelFormat(
                    name=plugin_config.ML_TOOL,
                ),
                storage_uri=model_uri,
                protocol_version="v2",
            ),
        )

        isvc = V1beta1InferenceService(
            api_version=constants.KSERVE_V1BETA1,
            kind=constants.KSERVE_KIND,
            metadata=client.V1ObjectMeta(
                name=isvc_name,
                namespace=namespace,
                annotations={"sidecar.istio.io/inject": "false"},
            ),
            spec=V1beta1InferenceServiceSpec(predictor=predictor),
        )
        kserve = KServeClient()
        kserve.create(isvc)
        time.sleep(plugin_config.TIMER_IN_SEC)

    @staticmethod
    def serve_model_v1(model_uri: str, name: str = None):
        """
        Create a kserve instance version1.

        Args:
            model_uri (str): URI of the model.
            name (str, optional): Name of the kserve instance. If not provided,
            a default name will be generated.

        Returns:
            None
        """
        # Verify plugin activation
        PluginManager().verify_activation(KubeflowPlugin().section)

        isvc_name = name
        namespace = utils.get_default_target_namespace()
        isvc = V1beta1InferenceService(
            api_version=constants.KSERVE_V1BETA1,
            kind=constants.KSERVE_KIND,
            metadata=V1ObjectMeta(
                name=isvc_name,
                namespace=namespace,
                annotations={"sidecar.istio.io/inject": "false"},
            ),
            spec=V1beta1InferenceServiceSpec(
                predictor=V1beta1PredictorSpec(
                    service_account_name="kserve-controller-s3",
                    sklearn=V1beta1SKLearnSpec(storage_uri=model_uri),
                )
            ),
        )

        kclient = KServeClient()
        kclient.create(isvc)
        time.sleep(plugin_config.TIMER_IN_SEC)

    @staticmethod
    def get_served_model_url(isvc_name: str):
        """
        Retrieve the URL of a deployed model.

        Args:
            isvc_name (str): Name of the deployed model.

        Returns:
            str: URL of the deployed model.
        """
        # Verify plugin activation
        PluginManager().verify_activation(KubeflowPlugin().section)

        kclient = KServeClient()

        @retry(
            wait=wait_exponential(multiplier=2, min=1, max=10),
            stop=stop_after_attempt(30),
            reraise=True,
        )
        def assert_isvc_created(kserve_client, isvc_name):
            """Wait for the Inference Service to be created successfully."""
            assert kserve_client.is_isvc_ready(
                isvc_name
            ), f"Failed to create Inference Service {isvc_name}."

        assert_isvc_created(kclient, isvc_name)

        isvc_resp = kclient.get(isvc_name)
        isvc_url = isvc_resp["status"]["address"]["url"]
        return isvc_url

    @staticmethod
    def delete_served_model(isvc_name: str):
        """
        Delete a deployed model by its ISVC name.

        Args:
            isvc_name (str): Name of the deployed model.

        Returns:
            None
        """
        # Verify plugin activation
        PluginManager().verify_activation(KubeflowPlugin().section)

        # if (
        #         response.get("status", {}).get("conditions", [{}])[0].get("type")
        #         == "IngressReady"
        # ):

        try:
            KServeClient().delete(isvc_name)
            print("Inference Service has been deleted successfully.")
        except Exception as exp:
            raise Exception(f"Failed to delete Inference Service: {exp}")

    @staticmethod
    def load_component_from_file(file_path):
        """
        Load a component from a File.

        Args:
            file_path (str): file_path to load the component from file.

        Returns:
            kfp.components.ComponentSpec: Loaded component specification.
        """
        # Verify plugin activation
        PluginManager().verify_activation(KubeflowPlugin().section)
        return kfp.components.load_component_from_file(file_path)

    @staticmethod
    def load_component_from_text(text):
        """
        Load a component from the text.

        Args:
            text (str):  load the component from text.

        Returns:
            kfp.components.ComponentSpec: Loaded component specification.
        """
        # Verify plugin activation
        PluginManager().verify_activation(KubeflowPlugin().section)
        return kfp.components.load_component_from_text(text)

    def create_run_from_pipeline_func(
        self,
        pipeline_func,
        arguments: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        namespace: Optional[str] = None,
        pipeline_root: Optional[str] = None,
        enable_caching: Optional[bool] = None,
        service_account: Optional[str] = None,
    ):
        """
            method to create a run from pipeline function
        :param pipeline_func:
        :param arguments:
        :param run_name:
        :param experiment_name:
        :param namespace:
        :param pipeline_root:
        :param enable_caching:
        :param service_account:
        :param experiment_id:
        :return:
        """
        run_details = self.client().create_run_from_pipeline_func(
            pipeline_func,
            arguments,
            run_name,
            experiment_name,
            namespace,
            pipeline_root,
            enable_caching,
            service_account,
        )
        return run_details

    def is_run_finished(self, run_id):
        """
            method to check if the run is finished
        :param run_id: run_id of the run
        :return: boolean
        """
        status = self.client().get_run(run_id).run.status
        return status in ["Succeeded", "Failed", "Skipped", "Error"]

    def get_run_status(self, run_id):
        """
        method return the status of run
        :param run_id: run_id of the run
        :return: status of the run
        """
        return self.client().get_run(run_id).run.status

    @staticmethod
    def delete_pipeline(pipeline_id):
        """
        method deletes the pipeline
        :param pipeline_id: pipeline id
        :return:
        """
        KubeflowPlugin.client().delete_pipeline(pipeline_id=pipeline_id)

    @staticmethod
    def list_pipeline_versions(pipeline_id):
        """
         method to list the pipeline based on pipeline_id
        :param pipeline_id: pipeline id
        :return:
        """
        response = KubeflowPlugin.client().list_pipeline_versions(
            pipeline_id=pipeline_id
        )
        return response

    @staticmethod
    def delete_pipeline_version(version_id):
        """
        method to list the pipeline based on version_id
        :param version_id: pipeline id
        :return:
        """
        KubeflowPlugin.client().delete_pipeline_version(version_id=version_id)

    @staticmethod
    def delete_runs(run_ids):
        """
        delete the pipeline runs
        :param run_ids: list of runs
        :return: successful deletion runs or 404 error
        """
        for run in run_ids:
            KubeflowPlugin.client().runs.delete_run(id=run)

    @staticmethod
    def get_default_namespace() -> str:
        """
        Retrieve the default namespace from the current Kubernetes configuration.
        Returns:
            str: The default namespace.
        """
        try:
            config.load_incluster_config()
            with open(
                "/var/run/secrets/kubernetes.io/serviceaccount/namespace",
                "r",
                encoding="utf-8",
            ) as f:
                return f.read().strip()
        except (FileNotFoundError, ConfigException):
            try:
                config.load_kube_config()
                current_context = config.list_kube_config_contexts()[1]
                return current_context["context"].get("namespace", "default")
            except ConfigException:
                return "default"

    @staticmethod
    def create_service(name: str) -> str:
        """
        Create a Kubernetes service for the component in the default namespace.
        Args:
            name (str): Name of the service to be created.
        Returns:
            str: Name of the created service.
        """
        namespace = KubeflowPlugin().get_default_namespace()
        srvname = name

        print(f"Creating service in namespace '{namespace}'...")

        # Define the service
        service_spec = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name=srvname,
                annotations={
                    "service.alpha.kubernetes.io/app-protocols": '{"grpc":"HTTP2"}'
                },
            ),
            spec=client.V1ServiceSpec(
                selector={"app": name},
                ports=[
                    client.V1ServicePort(
                        protocol="TCP", port=8080, name="grpc", target_port=8080
                    )
                ],
                type="ClusterIP",
            ),
        )

        # Create the Kubernetes API client
        api_instance = client.CoreV1Api()

        try:
            # Create the service
            api_instance.create_namespaced_service(
                namespace=namespace, body=service_spec
            )
            print(
                f"Service '{srvname}' created successfully in namespace '{namespace}'."
            )
        except client.exceptions.ApiException as e:
            raise RuntimeError(f"Exception when creating service: {e}")

        return srvname

    @staticmethod
    def delete_service(name: str):
        """
        Delete a Kubernetes service by name in the default namespace.
        Args:
            name (str): Name of the service to be deleted.
        """
        namespace = KubeflowPlugin().get_default_namespace()
        srvname = name
        print(f"Deleting service '{srvname}' from namespace '{namespace}'...")

        api_instance = client.CoreV1Api()

        try:
            api_instance.delete_namespaced_service(name=srvname, namespace=namespace)
            print(
                f"Service '{srvname}' deleted successfully from namespace '{namespace}'."
            )
        except client.exceptions.ApiException as e:
            print(f"Exception when deleting service: {e}")

    @staticmethod
    def create_fl_pipeline(fl_client, fl_server):
        """
        Returns a KFP pipeline function that wires up:
        setup_links → fl_server → many fl_client → release_links

        fl_client must accept at minimum:
        - server_address: str
        - local_data_connector

        fl_server must accept at minimum:
        - number_of_iterations: int

        Any other parameters that fl_client/ fl_server declare will automatically
        become pipeline inputs and be forwarded along.
        """

        def setup_links_func(name: str) -> str:
            """
            Set up a service in the default namespace with the given name.
            Args:
                name (str): Name of the service to be created.
            Returns:
                str: Name of the created service.
            """
            from cogflow import KubeflowPlugin

            KubeflowPlugin().create_service(name=name)
            return name

        def release_links_func(name: str):
            """
            Release a service created by `setup_links_func`.
            Deletes a previously created service by name in the default namespace.
            Args:
                name (str): Name of the service to be deleted.
            Returns:
                str: Result message of service deletion.
            """
            from cogflow import KubeflowPlugin

            KubeflowPlugin().delete_service(name=name)

        # Introspect client/server signatures
        client_sig = inspect.signature(fl_client)
        server_sig = inspect.signature(fl_server)

        # Mandatory params
        client_req = {"server_address", "local_data_connector"}
        server_req = {"number_of_iterations"}

        # ← CHANGE: only consider real kw/positional params, skip VAR_POSITIONAL and VAR_KEYWORD
        def _valid_param_names(sig):
            return [
                name
                for name, p in sig.parameters.items()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            ]

        client_params = _valid_param_names(client_sig)  # ← CHANGE
        server_params = _valid_param_names(server_sig)  # ← CHANGE

        # Find any extra parameters
        client_extra = [p for p in client_params if p not in client_req]
        server_extra = [p for p in server_params if p not in server_req]
        extra_params = list(dict.fromkeys(client_extra + server_extra))

        # Build a list of inspect.Parameter for the pipeline signature
        sig_params = []
        # 1) local_data_connectors
        sig_params.append(
            inspect.Parameter(
                name="local_data_connectors",
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=list,
            )
        )
        # 2) number_of_iterations
        sig_params.append(
            Parameter(
                name="number_of_iterations",
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                annotation=int,
            )
        )
        # 3) extras, preserving defaults & annotations
        for name in extra_params:
            # pick whichever component declares it
            param = client_sig.parameters.get(name, server_sig.parameters.get(name))
            default = (
                param.default if param.default is not inspect._empty else inspect._empty
            )
            ann = param.annotation if param.annotation is not inspect._empty else None
            sig_params.append(
                Parameter(
                    name=name,
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=ann,
                    default=default,
                )
            )
        pipeline_sig = Signature(parameters=sig_params)

        # create compoent from func
        setup_links = KubeflowPlugin.create_component_from_func(
            setup_links_func, base_image="hiroregistry/cogflow:latest"
        )
        release_links = KubeflowPlugin.create_component_from_func(
            release_links_func, base_image="hiroregistry/cogflow:latest"
        )

        def fl_pipeline_func(*args, **kwargs):
            # 2) bind positional → named arguments per our explicit signature
            bound = fl_pipeline_func.__signature__.bind_partial(
                *args, **kwargs
            )  # ← CHANGE
            bound.apply_defaults()  # ← CHANGE
            args_map = bound.arguments  # ← CHANGE
            # extract required inputs
            local_data_connectors = args_map["local_data_connectors"]
            number_of_iterations = args_map["number_of_iterations"]

            # split extras for client & server
            server_kwargs = {k: args_map[k] for k in server_extra}
            client_kwargs = {k: args_map[k] for k in client_extra}

            # generate service name with run id later at runtime it will replaced by run id
            srv_name = "flserver-" + "{{workflow.uid}}"
            # 1. create the k8s Service
            setup_task = setup_links(name=srv_name)
            # 1.1. tear down once the server is done
            cleanup_task = release_links(name=srv_name)
            # 2. start the FL server
            with dsl.ExitHandler(cleanup_task):
                server_task = fl_server(
                    number_of_iterations=number_of_iterations, **server_kwargs
                ).after(setup_task)
                server_task.add_pod_label(name="app", value=srv_name)

                # 3. fan-out clients in parallel
                with dsl.ParallelFor(local_data_connectors) as connector:
                    fl_client(
                        server_address=setup_task.output,
                        local_data_connector=connector,
                        **client_kwargs,
                    ).after(setup_task)


            # Attach the explicit signature so KFP can see all inputs

        fl_pipeline_func.__signature__ = pipeline_sig

        # Decorate as a pipeline
        flpipeline = dsl.pipeline(
            name="Federated Learning Pipeline", description="Auto-generated FL pipeline"
        )(fl_pipeline_func)
        return flpipeline

    @staticmethod
    def create_fl_component_from_func(
        func,
        output_component_file=None,
        base_image=None,
        packages_to_install=None,
        annotations: Optional[Mapping[str, str]] = None,
        container_port=8080,
        pod_label_name="app",
    ):
        """
        Create a component from a Python function with additional configurations
        for ports and pod labels using Pod UID to ensure unique run_id.
        """

        # Verify plugin activation
        PluginManager().verify_activation(KubeflowPlugin().section)

        def wrap_function_with_service(func):
            """
            Wraps a function to ensure service creation and deletion
            tied to the pod unique ID.
            """
            sig = inspect.signature(func)
            run_id_param = inspect.Parameter(
                name="srv_name",
                kind=inspect.Parameter.KEYWORD_ONLY,
                default="{{workflow.uid}}",
            )
            new_sig = sig.replace(
                parameters=list(sig.parameters.values()) + [run_id_param]
            )

            # @functools.wraps(func)
            def wrapped_func(*args, srv_name="{{workflow.uid}}", **kwargs):
                from kubernetes import client

                # srv_name = f"flserver-{srv_name}"
                # KubeflowPlugin().create_service(name=srv_name)
                srvname = f"flserver-{srv_name}"
                name = srvname
                import os
                from kubernetes import client, config

                # Load in-cluster config
                config.load_incluster_config()
                # _, active_context = config.list_kube_config_contexts()
                # print(active_context["context"])
                ns_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
                default_namespace = ""
                if os.path.exists(ns_path):
                    with open(ns_path, encoding="utf-8") as f:
                        default_namespace = f.read().strip()

                print(f"Crating endpoint...{default_namespace}")
                # Define the service
                service_spec = client.V1Service(
                    api_version="v1",
                    kind="Service",
                    metadata=client.V1ObjectMeta(
                        name=srvname,
                        annotations={
                            "service.alpha.kubernetes.io/app-protocols": '{"grpc":"HTTP2"}'
                        },
                    ),
                    spec=client.V1ServiceSpec(
                        selector={"app": name},
                        ports=[
                            client.V1ServicePort(
                                protocol="TCP", port=8080, name="grpc", target_port=8080
                            )
                        ],
                        type="ClusterIP",
                    ),
                )
                # Create the Kubernetes API client
                api_instance = client.CoreV1Api()
                try:
                    # Create the service
                    api_instance.create_namespaced_service(
                        namespace=default_namespace, body=service_spec
                    )
                    import time

                    time.sleep(30)
                    return f"Service '{srvname}' created successfully."
                except client.exceptions.ApiException as e:
                    return f"Exception when creating service: {e}"
                # try:
                #     return func(*args, **kwargs)
                # finally:
                #     KubeflowPlugin().delete_service(name=srv_name)

        wrapped_func.__signature__ = new_sig

        # Create the initial KFP component
        training_var = kfp.components.create_component_from_func(
            func=wrapped_func,
            output_component_file=output_component_file,
            base_image=base_image,
            packages_to_install=packages_to_install,
            annotations=annotations,
        )

        def wrapped_fl_component(*args, **kwargs):
            run_id = "{{workflow.uid}}"

            component_op = training_var(*args, **kwargs)

            # Add container port and pod labels
            component_op.container.add_port(
                V1ContainerPort(container_port=container_port)
            )
            component_op.add_pod_label(name=pod_label_name, value=run_id)

            # Add model access configurations
            component_op = CogContainer.add_model_access(component_op)

            # Add run_id and pod label
            run_id = f"flserver-{uuid.uuid4()}"
            component_op.add_env_variable(V1EnvVar("POD_LABEL", value=run_id))
            component_op.add_pod_label(name="app", value=run_id)

            return component_op

        wrapped_fl_component.component_spec = training_var.component_spec
        return wrapped_fl_component

    @staticmethod
    def create_fl_client_component(
        func,
        annotations: Optional[Mapping[str, str]] = None,
        output_component_file=None,
        base_image=None,
        packages_to_install=None,
    ) -> Callable:
        """
        Decorator to mark and execute an FL client function.

        Args:
            annotations (dict, optional): Arbitrary metadata to tag the component.
            func : Wraps a function
            output_component_file (str, optional): The output file for the component.
            base_image (str, optional): The base image to use. Defaults to
            "hiroregistry/cogflow:dev".
            packages_to_install (list, optional): List of packages to install.

        Returns:
            Callable: The original function, executed when called.
        """
        # 1) Build a signature that adds `run_id="{{workflow.uid}}"` as a kw-only arg
        sig = inspect.signature(func)
        run_id_param = inspect.Parameter(
            name="run_id",
            kind=inspect.Parameter.KEYWORD_ONLY,
            default="{{workflow.uid}}",
        )
        new_sig = sig.replace(parameters=list(sig.parameters.values()) + [run_id_param])

        # 2) Wrap the user's func so it *accepts* run_id (but just ignores it)
        # @functools.wraps(func)
        def func_with_run_id(*args, run_id="{{workflow.uid}}", **kwargs):
            kwargs.update({"serveraddress": f"flserver-{run_id}"})
            return func(*args, **kwargs)

        func_with_run_id.__signature__ = new_sig

        # 3) Create the initial KFP component
        training_var = kfp.components.create_component_from_func(
            func=func_with_run_id,
            output_component_file=output_component_file,
            base_image=base_image,
            packages_to_install=packages_to_install,
            annotations=annotations,
        )

        def wrapped_fl_client_component(*args, run_id="{{workflow.uid}}", **kwargs):
            component_op = training_var(*args, run_id=run_id, **kwargs)

            # Add model access configurations
            component_op = CogContainer.add_model_access(component_op)
            return component_op

        wrapped_fl_client_component.component_spec = training_var.component_spec
        return wrapped_fl_client_component

    @staticmethod
    def create_fl_base_component(
        func,
        annotations: Optional[Mapping[str, str]] = None,
        output_component_file=None,
        base_image=None,
        packages_to_install=None,
    ) -> Callable:
        """
        Decorator to mark and execute an FL client function.

        Args:
            annotations (dict, optional): Arbitrary metadata to tag the component.
            func : Wraps a function
            output_component_file (str, optional): The output file for the component.
            base_image (str, optional): The base image to use. Defaults to
            "hiroregistry/cogflow:dev".
            packages_to_install (list, optional): List of packages to install.

        Returns:
            Callable: The original function, executed when called.
        """

        # 3) Create the initial KFP component
        training_var = kfp.components.create_component_from_func(
            func=func,
            output_component_file=output_component_file,
            base_image=base_image,
            packages_to_install=packages_to_install,
            annotations=annotations,
        )

        def wrapped_fl_client_component(*args, run_id="{{workflow.uid}}", **kwargs):
            component_op = training_var(*args, run_id=run_id, **kwargs)

            # Add model access configurations
            component_op = CogContainer.add_model_access(component_op)
            return component_op

        wrapped_fl_client_component.component_spec = training_var.component_spec
        return wrapped_fl_client_component
