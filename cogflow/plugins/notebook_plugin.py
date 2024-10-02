"""
This module provides functionality related to Model actions via plugin.
"""

import json
import os
import subprocess
import sys
from datetime import datetime

# from typing import Optional, Mapping
import joblib
from .mlflowplugin import MlflowPlugin
from .. import plugin_config
from ..pluginmanager import PluginManager
from ..util import custom_serializer
from ..util import make_post_request, make_delete_request, make_get_request
from .kubeflowplugin import KubeflowPlugin


class NotebookPlugin:
    """
    Class for defining reusable components.
    """

    def __init__(self):
        """
        Initializes the ModelPlugin class.
        """
        self.section = "notebook_plugin"

    @staticmethod
    def link_model_to_dataset(dataset_id, model_id):
        """
        Links a model to a dataset using the provided API endpoint.

        This method sends a POST request to the API to associate a specified model
        with a given dataset. It uses the user's ID defined in the plugin configuration.

        Args:
            dataset_id (str): The ID of the dataset to link to the model.
            model_id (str): The ID of the model to be linked to the dataset.

        Returns:
            Response: The response object from the API call.

        Raises:
            requests.exceptions.RequestException: An error occurred when making the POST request.
        """
        # Verify plugin activation
        PluginManager().verify_activation(NotebookPlugin().section)

        PluginManager().load_config()

        data = {
            "user_id": plugin_config.JUPYTER_USER_ID,
            "model_id": model_id,
            "dataset_id": dataset_id,
        }
        # call the api
        url = os.getenv(plugin_config.API_BASEPATH) + PluginManager().load_path(
            "link_dataset_model"
        )
        return make_post_request(url, data=data)

    def save_model_details_to_db(self, registered_model_name):
        """
        store model details in database
        :param registered_model_name: name of the registered model
        :return: id of model
        """
        # Verify plugin activation
        PluginManager().verify_activation(NotebookPlugin().section)

        PluginManager().load_config()

        data = {
            "name": registered_model_name,
            "version": self.get_model_latest_version(registered_model_name),
            "type": "sklearn",
            "user_id": plugin_config.JUPYTER_USER_ID,
            "description": f"{registered_model_name} model",
        }

        # call the api to register model
        url = os.getenv(plugin_config.API_BASEPATH) + PluginManager().load_path(
            "models"
        )
        return make_post_request(url, data=data)

    @staticmethod
    def get_model_latest_version(registered_model_name: str):
        """
        return the latest version of registered model
        :param registered_model_name: model name to get the versions
        :return: latest version
        """
        # Verify plugin activation
        PluginManager().verify_activation(NotebookPlugin().section)

        latest_version_info = MlflowPlugin().search_model_versions(
            filter_string=f"name='{registered_model_name}'"
        )
        sorted_model_versions = sorted(
            latest_version_info, key=lambda x: int(x.version), reverse=True
        )

        if sorted_model_versions:
            latest_version = sorted_model_versions[0]
            # print("Latest Version:", latest_version.version)
            # print("Status:", latest_version.status)
            # print("Stage:", latest_version.current_stage)
            # print("Description:", latest_version.description)
            # print("Last Updated:", latest_version.last_updated_timestamp)
            return latest_version.version

        # print(f"No model versions found for {registered_model_name}")
        return 1

    @staticmethod
    def save_model_uri_to_db(model_id, model_uri):
        """
            method to call the api to save model uri
        :param model_id: model id of the model
        :param model_uri: model uri
        :return: API response
        """
        # Verify plugin activation
        PluginManager().verify_activation(NotebookPlugin().section)

        PluginManager().load_config()

        # call the api for saving model_uri
        data = {
            "user_id": plugin_config.JUPYTER_USER_ID,
            "model_id": model_id,
            "uri": model_uri,
            "description": f"model uri of model id :{model_id}",
        }
        url = os.getenv(plugin_config.API_BASEPATH) + PluginManager().load_path(
            "models_uri"
        )
        return make_post_request(url, data=data)

    @staticmethod
    def delete_pipeline_details_from_db(pipeline_id):
        """
        delete the pipeline details
        :param pipeline_id: pipeline id
        :return:
        """
        # Verify plugin activation
        PluginManager().verify_activation(NotebookPlugin().section)

        PluginManager().load_config()

        url = os.getenv(plugin_config.API_BASEPATH) + PluginManager().load_path(
            "pipeline"
        )
        return make_delete_request(url=url, path_params=pipeline_id)

    @staticmethod
    def list_runs_by_pipeline_id(pipeline_id):
        """
        list the pipeline run details
        :param pipeline_id: pipeline_id
        :return: list of run details
        """
        # Verify plugin activation
        PluginManager().verify_activation(NotebookPlugin().section)

        PluginManager().load_config()

        url = os.getenv(plugin_config.API_BASEPATH) + PluginManager().load_path(
            "pipeline_runs"
        )
        response = make_get_request(url=url, path_params=pipeline_id)
        return response['data']

    @staticmethod
    def delete_run_details_from_db(pipeline_id):
        """
         delete the pipeline details
        :param pipeline_id: pipeline_id
        :return: successful deletion message or 404 error
        """
        # Verify plugin activation
        PluginManager().verify_activation(NotebookPlugin().section)

        PluginManager().load_config()

        url = os.getenv(plugin_config.API_BASEPATH) + PluginManager().load_path(
            "pipeline_runs"
        )
        return make_delete_request(url=url, path_params=pipeline_id)

    @staticmethod
    def get_pipeline_id_by_name(pipeline_name):
        """
        Retrieves the pipeline ID for a given pipeline name.

        Args:
            pipeline_name (str): The name of the pipeline to fetch the ID for.

        Returns:
            str: The ID of the specified pipeline if found.

        Raises:
            ValueError: If no pipeline with the given name is found.

        Example:
            pipeline_id = NotebookPlugin.get_pipeline_id_by_name("Example Pipeline")
        """
        kfp = KubeflowPlugin()
        pipelines_response = kfp.client().list_pipelines()
        pipeline_id = None
        if pipelines_response.pipelines:
            for pipeline in pipelines_response.pipelines:
                if pipeline.name == pipeline_name:
                    pipeline_id = pipeline.id
                    return pipeline_id

        if not pipeline_id:
            print(f"No pipeline found with the name '{pipeline_name}'")

    @staticmethod
    def list_pipelines_by_name(pipeline_name):
        """
        Lists all versions and runs of the specified pipeline by name.

        Args:
            pipeline_name (str): The name of the pipeline to fetch details for.

        Returns:
            dict: A dictionary containing the pipeline ID, versions,
             and runs of the specified pipeline.

        Raises:
            ValueError: If the pipeline name is invalid or not found.
            Exception: For any other issues encountered during the fetch operations.
        """
        # Fetch all versions of the specified pipeline
        kfp = KubeflowPlugin()

        pipeline_id = NotebookPlugin.get_pipeline_id_by_name(pipeline_name)
        versions_response = kfp.list_pipeline_versions(pipeline_id=pipeline_id)
        run_list = NotebookPlugin.list_runs_by_pipeline_id(pipeline_id=pipeline_id)
        result_dict = {
            "pipeline_id": pipeline_id,
            "versions": versions_response.versions,
            "runs": run_list,
        }
        return result_dict

    @staticmethod
    def save_pipeline_details_to_db(details):
        """
            save the details related to pipeline to the database
        :param details: dictionary with all the details of pipeline,run_details,task_details,experiments
        :return:
        """
        # Verify plugin activation
        PluginManager().verify_activation(NotebookPlugin().section)

        PluginManager().load_config()

        data = json.dumps(details, default=custom_serializer, indent=4)
        url = os.getenv(plugin_config.API_BASEPATH) + PluginManager().load_path(
            "pipeline_add"
        )
        make_post_request(url=url, data=data)

    def log_model_by_model_file(self, model_file_path, model_name):
        """
            log_model in cogflow with the model_file
        :param model_file_path: file_path of model
        :param model_name: name of the model
        :return:
            data = {
                "artifact_uri" : 'artifact_uri of the model',
                "version" : "model version"
            }
        """
        model = self.load_pkl(model_file_path)
        mfp = MlflowPlugin()
        mfp.mlflow.set_experiment(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_exp")
        with mfp.mlflow.start_run() as run:
            run_id = run.info.run_id
            mfp.mlflow.sklearn.log_model(
                model, "model", registered_model_name=model_name
            )
            print("Artifact_path", run.info.artifact_uri)
            print("run_id", run_id)
            print("model_name", model_name)
            latest_version = self.get_model_latest_version(model_name)
            data = {
                "artifact_uri": f"{run.info.artifact_uri}/model",
                "version": latest_version,
            }
            return data

    def install_and_import(self, package):
        """
            install and import the given package
        :param package: package to be installed
        :return:
        """
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            __import__(package)

    def load_pkl(self, file_path):
        """
            load the pkl file to joblib
        :param file_path: path of .pkl file
        :return:
        """
        try:
            with open(file_path, "rb") as file:
                return joblib.load(file)
        except ModuleNotFoundError as e:
            missing_module = str(e).split("'")[1]
            print(f"Module {missing_module} not found. Installing...")
            self.install_and_import(missing_module)
            print(f"Module {missing_module} installed. Trying to load the file again.")
            return self.load_pkl(file_path)

    def deploy_model(self, model_name, model_version, isvc_name):
        """

        :param model_name: name of the model
        :param model_version: version of the model
        :param isvc_name: service name to be created for the deployed model
        :return:
        """
        try:
            PluginManager().verify_activation(NotebookPlugin().section)
            mfp = MlflowPlugin()
            model_uri = mfp.get_model_uri(model_name=model_name, version=model_version)
            kfp = KubeflowPlugin()
            kfp.serve_model_v1(model_uri, name=isvc_name)
            return {
                "status": True,
                "msg": f"Model {model_name} deployed with service {isvc_name}",
            }
        except Exception as exp:
            raise exp

    @staticmethod
    def run_kubectl_command(command):
        """
        Run a kubectl command and return the output.

        Args:
            command (list): The kubectl command to run.

        Returns:
            str: The output from the kubectl command, or None if the command failed.
        """
        try:
            print(f"Running command: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print(f"Command output: {result.stdout}")
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Command '{' '.join(command)}' failed with error: {e.stderr}")
            return None
        except Exception as e:
            print(f"Unexpected error running command '{' '.join(command)}': {str(e)}")
            return None

    @staticmethod
    def get_pods_in_namespace(namespace):
        """
        Get the list of pods in a given namespace.

        Args:
            namespace (str): The Kubernetes namespace to query.

        Returns:
            list: A list of pod items in the given namespace, or an empty list if none are found.
        """
        command = ["kubectl", "get", "pods", "-n", namespace, "-o", "json"]
        output = NotebookPlugin.run_kubectl_command(command)
        if output:
            pod_info = json.loads(output)
            return pod_info["items"]
        return []

    @staticmethod
    def get_pod_logs(namespace, pod_name, container_name=None):
        """
        Get the logs for a specific pod.

        Args:
            namespace (str): The Kubernetes namespace.
            pod_name (str): The name of the pod.
            container_name (str, optional): The name of the container in the pod. Defaults to None.

        Returns:
            str: The logs from the pod, or None if the command failed.
        """
        command = ["kubectl", "logs", "-n", namespace, pod_name]
        if container_name:
            command.append(container_name)
        logs = NotebookPlugin.run_kubectl_command(command)
        return logs

    @staticmethod
    def get_pod_prefix(inference_service_name):
        """
        Get the pod prefix for a given inference service.

        Args:
            inference_service_name (str): The name of the inference service.

        Returns:
            str: The latest ready revision of the inference service, or None if not found.
        """
        command = [
            "kubectl",
            "get",
            "inferenceservice",
            inference_service_name,
            "-o",
            "json",
        ]
        output = NotebookPlugin.run_kubectl_command(command)
        if output:
            isvc_info = json.loads(output)
            # logger.info(f"InferenceService JSON: {json.dumps(isvc_info, indent=2)}")
            # Adjust the key based on the actual JSON structure
            if (
                "status" in isvc_info
                and "components" in isvc_info["status"]
                and "predictor" in isvc_info["status"]["components"]
            ):
                predictor_info = isvc_info["status"]["components"]["predictor"]
                if "latestReadyRevision" in predictor_info:
                    latest_ready_revision = predictor_info["latestReadyRevision"]
                    return latest_ready_revision
            return "Key 'latestReadyRevision' not found in InferenceService status"
        return None

    @staticmethod
    def get_logs_for_inference_service(namespace, inference_service_name):
        """
        Get the logs for all pods related to a specific inference service.

        Args:
            namespace (str): The Kubernetes namespace.
            inference_service_name (str): The name of the inference service.

        Returns:
            dict: A dictionary containing the logs for each related pod, or None if an error occurred.
        """
        pod_prefix = NotebookPlugin.get_pod_prefix(inference_service_name)
        if not pod_prefix:
            return f"Failed to retrieve pod prefix for inference service '{inference_service_name}'"
        pods = NotebookPlugin.get_pods_in_namespace(namespace)
        if not pods:
            return f"No pods found in namespace '{namespace}'"
        related_pods = [pod for pod in pods if pod_prefix in pod["metadata"]["name"]]
        if not related_pods:
            return f"No pods found for inference service '{inference_service_name}' with prefix '{pod_prefix}'"
        logs_output = {}
        for pod in related_pods:
            pod_name = pod["metadata"]["name"]
            print(f"Retrieving logs for pod: {pod_name} in namespace: {namespace}")
            logs = NotebookPlugin.get_pod_logs(namespace, pod_name)
            if logs:
                logs_output[pod_name] = logs
            else:
                print(f"Failed to get logs for pod  '{pod_name}'")
        return logs_output

    @staticmethod
    def serialize_artifacts(artifacts):
        """
        Converts the artifacts dictionary into a JSON serializable format.
        Each artifact object is converted to its URI string representation.

        Args:
            artifacts (dict): The original artifacts dictionary.

        Returns:
            dict: A dictionary with JSON serializable artifact data.
        """
        serialized_artifacts = {}

        for key, artifact in artifacts.items():
            # Convert artifact objects (like ImageEvaluationArtifact) to their URI string representation
            if hasattr(artifact, "uri"):
                serialized_artifacts[key] = artifact.uri
            else:
                serialized_artifacts[key] = str(
                    artifact
                )  # Fallback conversion to string

        return {"validation_artifacts": serialized_artifacts}
