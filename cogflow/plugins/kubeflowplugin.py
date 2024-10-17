"""
This module provides functionality related to Kubeflow Pipelines.
"""

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
from kubernetes import client
from kubernetes.client import V1ObjectMeta
from kubernetes.client.models import V1EnvVar
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

        # Adding environment variables
        for key, value in os.environ.items():
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
