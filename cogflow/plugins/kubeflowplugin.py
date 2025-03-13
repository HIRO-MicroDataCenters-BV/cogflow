"""
This module provides functionality related to Kubeflow Pipelines.
"""
import functools
import inspect
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any, Mapping
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
    def create_fl_component_from_func(
        func,
        output_component_file=None,
        base_image=None,
        packages_to_install=None,
        annotations: Optional[Mapping[str, str]] = None,
        container_port=8080,
        pod_label_name="app",
        pod_label_value="flserver",
    ):
        """
        Create a component from a Python function with additional configurations
        for ports and pod labels.
        Args:
            func (Callable): Python function to convert into a component.
            output_component_file (str, optional): Path to save the component YAML file. Defaults to None.
            base_image (str, optional): Base Docker image for the component. Defaults to None.
            packages_to_install (List[str], optional): List of additional Python packages to install.
            annotations: Optional. Adds arbitrary key-value data to the component specification.
            container_port (int, optional): Container port to expose. Defaults to 8080.
            pod_label_name (str, optional): Name of the pod label. Defaults to "app".
            pod_label_value (str, optional): Value of the pod label. Defaults to "flserver".
        Returns:
            kfp.components.ComponentSpec: Component specification.
        """
        # Verify plugin activation
        PluginManager().verify_activation(KubeflowPlugin().section)

        def wrap_function_with_service(func):
            """Wraps a function to ensure create_service is called before, and delete_service is called after execution."""
            sig = inspect.signature(func)

            @functools.wraps(func)
            def wrapped_func(*args, **kwargs):
                """
                Wrapper function to create and delete a service for the component.
                """
                KubeflowPlugin().create_service(name=pod_label_value)

                try:
                    return func(*args, **kwargs)
                finally:
                    KubeflowPlugin().delete_service(name=pod_label_value)

            wrapped_func.__signature__ = sig

            return wrapped_func

        # Create the initial component specification
        training_var = kfp.components.create_component_from_func(
            func=wrap_function_with_service(func),
            output_component_file=output_component_file,
            base_image=base_image,
            packages_to_install=packages_to_install,
            annotations=annotations,
        )

        def wrapped_fl_component(*args, **kwargs):
            # Create the service
            KubeflowPlugin().create_service(name=pod_label_value)

            try:
                # Execute the component operation
                component_op = training_var(*args, **kwargs)

                # Add container port and pod labels
                component_op.container.add_port(
                    V1ContainerPort(container_port=container_port)
                )
                component_op.add_pod_label(name=pod_label_name, value=pod_label_value)

                # Add model access configurations
                component_op = CogContainer.add_model_access(component_op)
                return component_op
            finally:
                # Ensure service is deleted after execution
                KubeflowPlugin().delete_service(name=pod_label_value)

        wrapped_fl_component.component_spec = training_var.component_spec
        return wrapped_fl_component
