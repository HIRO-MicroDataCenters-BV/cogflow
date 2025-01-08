"""
This module provides functionality related to Model actions via plugin.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Any
import joblib
from kubernetes import client, config
from kubernetes.client.rest import ApiException
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
            "file_type": plugin_config.FILE_TYPE,
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
        return response["data"]

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
            "pipeline"
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
                serialized_artifacts[key] = str(artifact)

        return {"validation_artifacts": serialized_artifacts}

    @staticmethod
    def model_recommender(model_name=None, classification_score=None):
        """
        Calls the model recommender API and returns the response.

        Args:
        - model_name (str): The name of the model to recommend (optional).
        - classification_score (list): A list of classification scores to consider(e.g., accuracy_score, f1_score,
         recall_score, log_loss, roc_auc, precision_score, example_count, score.). (optional).

        Returns:
        - dict: The response from the model recommender API.
        """
        # Verify plugin activation
        PluginManager().verify_activation(NotebookPlugin().section)

        PluginManager().load_config()

        # call the api for model recommend
        data = {"model_name": model_name, "classification_score": classification_score}
        url = os.getenv(plugin_config.API_BASEPATH) + PluginManager().load_path(
            "model_recommend"
        )
        return make_get_request(url, query_params=data)

    @staticmethod
    def get_pipeline_task_sequence_by_run_id(run_id):
        """
        Fetches the pipeline workflow and task sequence for a given run in Kubeflow.

        Args:
            run_id (str): The ID of the pipeline run to fetch details for.

        Returns:
            tuple: A tuple containing:
                - pipeline_workflow_name (str): The name of the pipeline's workflow (root node of the DAG).
                - task_structure (dict): A dictionary representing the task structure of the pipeline, with each node
                                         containing information such as task ID, pod name, status, inputs, outputs,
                                         and resource duration.

        The task structure contains the following fields for each node:
            - id (str): The unique ID of the task (node).
            - podName (str): The name of the pod associated with the task.
            - name (str): The display name of the task.
            - inputs (list): A list of input parameters for the task.
            - outputs (list): A list of outputs produced by the task.
            - status (str): The phase/status of the task (e.g., 'Succeeded', 'Failed').
            - startedAt (str): The timestamp when the task started.
            - finishedAt (str): The timestamp when the task finished.
            - resourcesDuration (dict): A dictionary representing the resources used (e.g., CPU, memory).
            - children (list): A list of child tasks (if any) in the DAG.

        Example:
            >>> run_id = "afcf98bb-a9af-4a34-a512-1236110150ae"
            >>> pipeline_name, task_structure = get_pipeline_task_sequence_by_run_id(run_id)
            >>> print(f"Pipeline Workflow Name: {pipeline_name}")
            >>> print("Task Structure:", task_structure)

        Raises:
            ValueError: If the root node (DAG) is not found in the pipeline.
        """

        # Initialize the Kubeflow client
        kfp_client = KubeflowPlugin().client()

        # Get the details of the specified run using the run ID
        run_details = kfp_client.get_run(run_id)

        # Parse the workflow manifest from the pipeline runtime
        workflow_graph = json.loads(run_details.pipeline_runtime.workflow_manifest)
        namespace = workflow_graph["metadata"]["namespace"]

        # Access the nodes in the pipeline graph
        nodes = workflow_graph["status"]["nodes"]

        # Initialize variables for the pipeline name and root node ID
        pipeline_workflow_name = None
        root_node_id = None

        # Iterate through nodes to find the DAG root (pipeline root node)
        for node_id, node_data in nodes.items():
            if node_data["type"] == "DAG":
                pipeline_workflow_name = node_data["displayName"]
                root_node_id = node_id
                break

        if not root_node_id:
            raise ValueError("Root DAG node not found in the pipeline run.")

        # Task structure to store the task details
        task_structure = {}

        # Recursive function to traverse the graph and build the task structure
        def traverse(node_id, parent=None):
            node = nodes[node_id]

            # Extract inputs, outputs, phase (status), and other information
            inputs = node.get("inputs", {}).get("parameters", [])
            outputs = node.get("outputs", [])
            phase = node.get("phase", "unknown")
            started_at = node.get("startedAt", "unknown")
            finished_at = node.get("finishedAt", "unknown")
            resources_duration = node.get("resourcesDuration", {})

            # Task information dictionary for the current node
            task_info = {
                "id": node_id,
                "namespace": namespace,
                "podName": node_id,  # Assuming podName is the same as node_id
                "name": node["displayName"],
                "inputs": inputs,
                "outputs": outputs,
                "status": phase,
                "startedAt": started_at,
                "finishedAt": finished_at,
                "resourcesDuration": resources_duration,
                "children": [],
            }

            # Add the task to the parent's children or to the task structure if it's the root
            if parent is None:
                task_structure[node_id] = task_info
            else:
                parent["children"].append(task_info)

            # Recursively traverse and process child nodes
            if "children" in node and node["children"]:
                for child_id in node["children"]:
                    traverse(child_id, task_info)

        # Begin traversal starting from the root node of the pipeline
        if root_node_id:
            traverse(root_node_id)

        response = {
            "run_id": run_id,
            "pipeline_workflow_name": pipeline_workflow_name,
            "namespace": namespace,
            "task_structure": task_structure,
        }

        return response

    @staticmethod
    def get_run_id_by_run_name(run_name):
        """
        Fetches the run_id of a pipeline run by its name, traversing all pages if necessary.

        Args:
            run_name (str): The name of the pipeline run to search for.

        Returns:
            str: The run_id if found, otherwise None.
        """
        next_page_token = None
        page_size = 100  # Set page size (adjust if needed)
        kfp_client = KubeflowPlugin().client()

        # Traverse through pages to find the matching run name
        while True:
            # Fetch the list of runs, providing the next_page_token to continue from the last point
            runs_list = kfp_client.list_runs(
                page_size=page_size, page_token=next_page_token
            )

            # Check the current page for the run with the specified name
            for run in runs_list.runs:
                if run.name == run_name:
                    return run.id

            # Check if there are more pages
            next_page_token = runs_list.next_page_token
            if not next_page_token:
                # No more pages, the run was not found
                break

        return None

    @staticmethod
    def get_pipeline_task_sequence_by_run_name(run_name):
        """
        Fetches the task structure of a pipeline run based on its name.

        Args:
            run_name (str): The name of the pipeline run to fetch task structure for.

        Returns:
            tuple: (pipeline_workflow_name, task_structure)
        Example:
            >>>run_name = "Run of test_pipeline (ad001)"
            >>>pipeline_name, task_structure = get_pipeline_task_sequence_by_run_name(run_name)
            >>>print(f'Pipeline Workflow Name: {pipeline_name}')
            >>>print("Task Structure:")
            >>>print(json.dumps(task_structure, indent=4))
        """
        kfp_client = KubeflowPlugin().client()

        # Fetch the run_id using the run_name
        run_id = NotebookPlugin().get_run_id_by_run_name(run_name)

        if not run_id:
            raise ValueError(f"No run found with name: {run_name}")

        # Get the details of the specified run by run_id
        run_details = kfp_client.get_run(run_id)

        # Parse the workflow manifest
        workflow_graph = json.loads(run_details.pipeline_runtime.workflow_manifest)
        namespace = workflow_graph["metadata"]["namespace"]

        # Access the nodes in the graph
        nodes = workflow_graph["status"]["nodes"]

        # Store the pipeline name and root node
        pipeline_workflow_name = None
        root_node_id = None

        for node_id, node_data in nodes.items():
            if node_data["type"] == "DAG":
                pipeline_workflow_name = node_data["displayName"]
                root_node_id = node_id
                break

        # Create a task representation structure
        task_structure = {}

        # Function to traverse the graph and build the task structure
        def traverse(node_id, parent=None):
            node = nodes[node_id]

            # Extract inputs, outputs, and additional information
            inputs = node.get("inputs", {}).get("parameters", [])
            outputs = node.get("outputs", [])
            phase = node.get("phase", "unknown")
            started_at = node.get("startedAt", "unknown")
            finished_at = node.get("finishedAt", "unknown")
            resources_duration = node.get("resourcesDuration", {})

            # Prepare the task information
            task_info = {
                "id": node_id,
                "namespace": namespace,
                "podName": node_id,
                "name": node["displayName"],
                "inputs": inputs,  # Include inputs
                "outputs": outputs,  # Include outputs
                "status": phase,
                "startedAt": started_at,
                "finishedAt": finished_at,
                "resourcesDuration": resources_duration,
                "children": [],
            }

            # Add task to the parent
            if parent is None:
                task_structure[node_id] = task_info
            else:
                parent["children"].append(task_info)

            # Recursively traverse child nodes
            if "children" in node and node["children"]:
                for child_id in node["children"]:
                    traverse(child_id, task_info)

        # Start traversing from the root node
        if root_node_id:
            traverse(root_node_id)

        response = {
            "run_id": run_id,
            "pipeline_workflow_name": pipeline_workflow_name,
            "namespace": namespace,
            "task_structure": task_structure,
        }

        return response

    @staticmethod
    def get_run_ids_by_pipeline_id(pipeline_id):
        """
        Fetches all run_ids for a given pipeline ID.

        Args:
            pipeline_id (str): The ID of the pipeline to search for.

        Returns:
            list: A list of run_ids for the matching pipeline ID.
        """
        run_ids = []
        next_page_token = None
        kfp_client = KubeflowPlugin().client()
        while True:
            runs_list = kfp_client.list_runs(page_size=100, page_token=next_page_token)
            for run in runs_list.runs:
                # Check if the run's pipeline_id matches the provided pipeline_id
                if run.pipeline_spec.pipeline_id == pipeline_id:
                    run_ids.append(run.id)

            # Check if there is a next page
            next_page_token = runs_list.next_page_token
            if not next_page_token:
                break  # Exit if there are no more pages

        return run_ids

    @staticmethod
    def get_pipeline_task_sequence_by_pipeline_id(pipeline_id):
        """
        Fetches the task structures of all pipeline runs based on the provided pipeline_id.

        Args:
            pipeline_id (str): The ID of the pipeline to fetch task structures for.

        Returns:
            list: A list of dictionaries containing pipeline workflow names and task structures for each run.
        Example:
            >>>pipeline_id = "1000537e-b101-4432-a779-768ec479c2b0"  # Replace with your actual pipeline_id
            >>>all_task_structures = get_pipeline_task_sequence_by_pipeline_id(pipeline_id)
            >>>for details in all_task_structures:
                >>>print(f'Run ID: {details["run_id"]}')
                >>>print(f'Pipeline Workflow Name: {details["pipeline_workflow_name"]}')
                >>>print("Task Structure:")
                >>>print(json.dumps(details["task_structure"], indent=4))
        """
        kfp_client = KubeflowPlugin().client()

        # Fetch all run_ids using the pipeline_id
        run_ids = NotebookPlugin().get_run_ids_by_pipeline_id(pipeline_id)

        if not run_ids:
            raise ValueError(f"No runs found for pipeline_id: {pipeline_id}")

        # Create a list to hold task structures for each run
        task_structures = []

        for run_id in run_ids:
            # Get the details of the specified run by run_id
            run_details = kfp_client.get_run(run_id)

            # Parse the workflow manifest
            workflow_graph = json.loads(run_details.pipeline_runtime.workflow_manifest)
            namespace = workflow_graph["metadata"]["namespace"]

            # Access the nodes in the graph
            nodes = workflow_graph["status"]["nodes"]

            # Store the pipeline name and root node
            pipeline_workflow_name = None
            root_node_id = None

            for node_id, node_data in nodes.items():
                if node_data["type"] == "DAG":
                    pipeline_workflow_name = node_data["displayName"]
                    root_node_id = node_id
                    break

            # Create a task representation structure
            task_structure = {}

            # Function to traverse the graph and build the task structure
            def traverse(node_id, parent=None):
                node = nodes[node_id]

                # Extract inputs, outputs, and additional information
                inputs = node.get("inputs", {}).get("parameters", [])
                outputs = node.get("outputs", [])
                phase = node.get("phase", "unknown")
                started_at = node.get("startedAt", "unknown")
                finished_at = node.get("finishedAt", "unknown")
                resources_duration = node.get("resourcesDuration", {})

                # Prepare the task information
                task_info = {
                    "id": node_id,
                    "namespace": namespace,
                    "podName": node_id,
                    "name": node["displayName"],
                    "inputs": inputs,  # Include inputs
                    "outputs": outputs,  # Include outputs
                    "status": phase,
                    "startedAt": started_at,
                    "finishedAt": finished_at,
                    "resourcesDuration": resources_duration,
                    "children": [],
                }

                # Add task to the parent
                if parent is None:
                    task_structure[node_id] = task_info
                else:
                    parent["children"].append(task_info)

                # Recursively traverse child nodes
                if "children" in node and node["children"]:
                    for child_id in node["children"]:
                        traverse(child_id, task_info)

            # Start traversing from the root node
            if root_node_id:
                traverse(root_node_id)

            # Append the task structure and workflow name for the current run_id
            task_structures.append(
                {
                    "run_id": run_id,
                    "pipeline_workflow_name": pipeline_workflow_name,
                    "namsepace": namespace,
                    "task_structure": task_structure,
                }
            )

        return task_structures

    @staticmethod
    def list_all_pipelines():
        """
        Lists all pipelines along with their IDs, handling pagination.

        Returns:
            list: A list of tuples containing (pipeline_name, pipeline_id).
        """
        kfp_client = KubeflowPlugin().client()

        pipelines_info = []
        next_page_token = None
        page_size = 100  # You can adjust this as needed

        while True:
            # Fetch all pipelines with pagination
            pipelines_list = kfp_client.list_pipelines(
                page_size=page_size, page_token=next_page_token
            )

            # Add the pipelines to the list
            for pipeline in pipelines_list.pipelines:
                pipelines_info.append((pipeline.name, pipeline.id))

            # Check if there is a next page
            next_page_token = pipelines_list.next_page_token
            if not next_page_token:
                break  # Exit the loop if there are no more pages

        return pipelines_info

    @staticmethod
    def get_run_ids_by_pipeline_name(pipeline_name):
        """
        Fetches all run_ids for a given pipeline name.

        Args:
            pipeline_name (str): The name of the pipeline to search for.

        Returns:
            list: A list of run_ids for the matching pipeline name.
        """
        run_ids = []
        next_page_token = None
        kfp_client = KubeflowPlugin().client()
        while True:
            runs_list = kfp_client.list_runs(page_size=100, page_token=next_page_token)
            for run in runs_list.runs:
                # Check if the run's pipeline name matches the provided pipeline name
                if run.pipeline_spec.pipeline_name == pipeline_name:
                    run_ids.append(run.id)

            # Check if there is a next page
            next_page_token = runs_list.next_page_token
            if not next_page_token:
                break  # Exit if there are no more pages

        return run_ids

    @staticmethod
    def get_all_run_ids():
        """
        Fetches all run_ids available in the system.

        Returns:
            list: A list of all run_ids.
        """
        run_ids = []
        next_page_token = None
        kfp_client = KubeflowPlugin().client()
        while True:
            runs_list = kfp_client.list_runs(page_size=100, page_token=next_page_token)
            for run in runs_list.runs:
                run_ids.append(run.id)

            next_page_token = runs_list.next_page_token
            if not next_page_token:
                break

        return run_ids

    @staticmethod
    def get_run_ids_by_name(run_name):
        """
        Fetches run_ids by run name.

        Args:
            run_name (str): The name of the run to search for.

        Returns:
            list: A list of run_ids matching the run_name.
        """
        run_ids = []
        next_page_token = None
        kfp_client = KubeflowPlugin().client()
        while True:
            runs_list = kfp_client.list_runs(page_size=100, page_token=next_page_token)
            for run in runs_list.runs:
                if run.name == run_name:
                    run_ids.append(run.id)

            next_page_token = runs_list.next_page_token
            if not next_page_token:
                break

        return run_ids

    @staticmethod
    def get_task_structure_by_task_id(task_id, run_id=None, run_name=None):
        """
        Fetches the task structure of a specific task ID, optionally filtered by run_id or run_name.

        Args:
            task_id (str): The task ID to look for.
            run_id (str, optional): The specific run ID to filter by. Defaults to None.
            run_name (str, optional): The specific run name to filter by. Defaults to None.

        Returns:
            list: A list of dictionaries containing run IDs and their corresponding task info if found.
        Example:
            >>>task_id = "test-pipeline-749dn-2534915009"
            >>>run_id = None  # "afcf98bb-a9af-4a34-a512-1236110150ae"
            >>>run_name = "Run of test_pipeline (ad001)"
            >>>get_task_structure_by_task_id(task_id, run_id, run_name)
        """
        kfp_client = KubeflowPlugin().client()

        # Fetch all run_ids available in the system
        run_ids = NotebookPlugin().get_all_run_ids()

        # If run_name is provided, filter by run_name
        if run_name:
            run_ids = NotebookPlugin().get_run_ids_by_name(run_name)

        # If run_id is provided, make it the only run to check
        if run_id:
            run_ids = [run_id] if run_id in run_ids else []

        task_structures = []

        for run_id in run_ids:
            # Get the details of the specified run by run_id
            run_details = kfp_client.get_run(run_id)

            # Parse the workflow manifest
            workflow_graph = json.loads(run_details.pipeline_runtime.workflow_manifest)
            namespace = workflow_graph["metadata"]["namespace"]

            # Access the nodes in the graph
            nodes = workflow_graph["status"]["nodes"]

            # Check if the task_id exists in the nodes
            if task_id in nodes:
                node_data = nodes[task_id]
                # Extract necessary details
                task_info = {
                    "id": task_id,
                    "namespace": namespace,
                    "name": node_data["displayName"],
                    "inputs": node_data.get("inputs", {}).get("parameters", []),
                    "outputs": node_data.get("outputs", []),
                    "status": node_data.get("phase", "unknown"),
                    "startedAt": node_data.get("startedAt", "unknown"),
                    "finishedAt": node_data.get("finishedAt", "unknown"),
                    "resourcesDuration": node_data.get("resourcesDuration", {}),
                    "run_id": run_id,
                }

                # Store the task info
                task_structures.append(task_info)
        if not task_structures:
            raise ValueError(f"No task found with ID: {task_id}.")
        return task_structures

    @staticmethod
    def load_k8s_config() -> None:
        """
        Loads the Kubernetes configuration.

        This method tries to load the in-cluster configuration if the code is running inside a pod.
        If it fails (e.g., if the code is running outside the cluster), it loads the kubeconfig file
        from the default location.

        Raises:
            config.config_exception.ConfigException: If the configuration could not be loaded.
        """
        try:
            # Load in-cluster config if running in a pod
            config.load_incluster_config()
        except config.config_exception.ConfigException:
            # If not running in a pod, load the kubeconfig file
            config.load_kube_config()

    @staticmethod
    def get_pods(
        namespace=KubeflowPlugin().get_default_namespace(),
    ) -> Any:
        """
        Retrieves the list of pods in a specified namespace with a given label selector.

        Args:
            namespace (str): The namespace to list pods from.

        Returns:
            list: A list of V1Pod objects representing the pods that match the criteria.

        Raises:
            ApiException: If an error occurs while trying to list the pods.
        """
        NotebookPlugin().load_k8s_config()
        v_1 = client.CoreV1Api()
        try:
            pods = v_1.list_namespaced_pod(namespace=namespace)
            return pods
        except ApiException as exp:
            raise Exception(
                f"Exception when calling CoreV1Api->list_namespaced_pod: {str(exp)}"
            )

    @staticmethod
    def get_inference_service_logs(
        inference_service_name,
        namespace=KubeflowPlugin().get_default_namespace(),
        container_name="kserve-container",
    ) -> Any:
        """
        Retrieve logs for all pods matching the given label selector in a specified namespace.
        Args:
            namespace (str, optional): The Kubernetes namespace where the pods are located.
            inference_service_name (str): A inference_service to filter pods by their inference service.

            container_name (str, optional): The name of the container to fetch logs from.
                                            If not specified, logs are fetched from the default kserve-container.

        Returns:
            str: A JSON-formatted string containing the logs for each pod. Each log entry includes:
                 - "pod_name": The name of the pod.
                 - "namespace": The namespace of the pod.
                 - "logs": The logs for the pod, or a message indicating that no logs are available.
        """
        NotebookPlugin().load_k8s_config()

        pods = NotebookPlugin().get_pods(namespace)
        inference_pods = [
            pod for pod in pods.items if inference_service_name in pod.metadata.name
        ]
        log_entries = []
        for pod in inference_pods:
            pod_logs = NotebookPlugin().get_pod_logs(
                namespace, pod.metadata.name, container_name
            )
            log_entry = {
                "pod_name": pod.metadata.name,
                "namespace": pod.metadata.namespace,
                "logs": pod_logs if pod_logs else "No logs available.",
            }
            log_entries.append(log_entry)
        # Convert log entries list to JSON string
        json_logs = json.dumps(log_entries, indent=4)
        return {"logs": json_logs}

    @staticmethod
    def get_pod_logs(
        pod_name,
        namespace=KubeflowPlugin().get_default_namespace(),
        container_name=None,
    ):
        """
        Retrieves the logs of a specified pod.

        Args:
            namespace (str): The namespace of the pod.
            pod_name (str): The name of the pod.
            container_name (str, optional): The name of the container within the pod.
            If not specified, logs of the first container in the pod are returned.

        Returns:
            str: The logs of the specified pod.

        Raises:
            ApiException: If an error occurs while trying to retrieve the pod logs.
        """
        NotebookPlugin().load_k8s_config()

        v_1 = client.CoreV1Api()
        try:
            if container_name:
                raw_logs = v_1.read_namespaced_pod_log(
                    name=pod_name, namespace=namespace, container=container_name
                )
            elif "pipeline" in pod_name:
                container_name = "main"  # Container name is 'main' for pipeline pods
                raw_logs = v_1.read_namespaced_pod_log(
                    name=pod_name, namespace=namespace, container=container_name
                )
            elif "postgres" in pod_name:
                container_name = (
                    "postgres"  # Container name is 'main' for pipeline pods
                )
                raw_logs = v_1.read_namespaced_pod_log(
                    name=pod_name, namespace=namespace, container=container_name
                )
            elif "predictor" in pod_name:
                container_name = (
                    "kserve-container"  # Container name is 'main' for pipeline pods
                )
                raw_logs = v_1.read_namespaced_pod_log(
                    name=pod_name, namespace=namespace, container=container_name
                )

            else:
                raw_logs = v_1.read_namespaced_pod_log(
                    name=pod_name, namespace=namespace
                )

            parse_logs = [line.strip() for line in raw_logs.split("\n") if line.strip()]

            logs = json.dumps(parse_logs, indent=4)
            return logs
        except ApiException as exp:
            raise Exception(
                f"Exception when calling CoreV1Api->read_namespace_pod_log: {str(exp)}"
            )

    @staticmethod
    def get_run_ids_by_pipeline_workflow_name(pipeline_workflow_name):
        """
        Fetches all run IDs associated with a given pipeline workflow name.

        Args:
            pipeline_workflow_name (str): The workflow name of the pipeline.

        Returns:
            list: A list of run IDs associated with the provided workflow name.

        Example:
            >>> workflow_name = "pipeline-vzn7z"
            >>> run_ids = get_run_ids_by_pipeline_workflow_name(workflow_name)
            >>> print(run_ids)
        """
        kfp_client = KubeflowPlugin().client()

        # Initialize variables for pagination
        run_ids = []
        page_token = None

        while True:
            try:
                # Fetch a page of runs
                response = kfp_client.list_runs(page_token=page_token)
            except Exception as e:
                raise Exception(f"Error fetching runs from Kubeflow: {str(e)}")

            if not response or not response.runs:
                break

            # Process runs in the current page
            for run in response.runs:
                try:
                    # Fetch detailed run information using run ID
                    run_details = client.get_run(run.id)
                    # print(run_details)
                    # Extract workflow manifest from run details
                    workflow_manifest = json.loads(
                        run_details.pipeline_runtime.workflow_manifest
                    )

                    # Check if the workflow name matches the given pipeline_workflow_name
                    if pipeline_workflow_name in workflow_manifest["metadata"]["name"]:
                        # print(run.id)
                        run_ids.append(run.id)
                except Exception as e:
                    print(f"Error processing run {run.id}: {str(e)}")

            # Update the page token to fetch the next page
            page_token = response.next_page_token

            # Break the loop if there are no more pages
            if not page_token:
                break

        return run_ids

    @staticmethod
    def get_pipeline_task_sequence(pipeline_name=None, pipeline_workflow_name=None):
        """
        Fetches the task structures of all pipeline runs based on the provided pipeline name or pipeline workflow name.

        Args:
            pipeline_name (str, optional): The name of the pipeline to fetch task structures for.
            pipeline_workflow_name (str, optional): The workflow name of the pipeline to fetch task structures for.

        Returns:
            list: A list with details of task structures for each run.
        Example:
            >>> pipeline_workflow_name = "pipeline-vzn7z"
            >>> all_task_structures = get_pipeline_task_sequence(pipeline_workflow_name=pipeline_workflow_name)
            >>> for details in all_task_structures:
                    >>> print(f'Run ID: {details["run_id"]}')
                    >>> print(f'Pipeline Workflow Name: {details["pipeline_workflow_name"]}')
                    >>> print("Task Structure:")
                    >>> print(json.dumps(details["task_structure"], indent=4))

        Raises:
            ValueError: If neither pipeline_name nor pipeline_workflow_name is provided.
        """
        kfp_client = KubeflowPlugin().client()

        # Fetch all run_ids based on pipeline_name or pipeline_workflow_name
        if pipeline_name:
            run_ids = NotebookPlugin().get_run_ids_by_pipeline_name(pipeline_name)
        elif pipeline_workflow_name:
            run_ids = NotebookPlugin().get_run_ids_by_pipeline_workflow_name(
                pipeline_workflow_name
            )
        else:
            raise ValueError(
                "Either pipeline_name or pipeline_workflow_name must be provided."
            )

        if not run_ids:
            identifier = pipeline_name or pipeline_workflow_name
            raise ValueError(f"No runs found for identifier: {identifier}")

        # Create a dictionary to hold task structures for each run
        task_structures = {}
        output_details = []  # List to hold details to return

        for run_id in run_ids:
            # Get the details of the specified run by run_id
            run_details = kfp_client.get_run(run_id)

            # Parse the workflow manifest
            workflow_graph = json.loads(run_details.pipeline_runtime.workflow_manifest)
            namespace = workflow_graph["metadata"]["namespace"]

            # Access the nodes in the graph
            nodes = workflow_graph["status"]["nodes"]

            # Store the pipeline name and root node
            pipeline_workflow_name = None
            root_node_id = None

            for node_id, node_data in nodes.items():
                if node_data["type"] == "DAG":
                    pipeline_workflow_name = node_data["displayName"]
                    root_node_id = node_id
                    break

            # Create a task representation structure
            task_structure = {}

            # Function to traverse the graph and build the task structure
            def traverse(node_id, parent=None):
                node = nodes[node_id]

                # Extract inputs, outputs, and additional information
                inputs = node.get("inputs", {}).get("parameters", [])
                outputs = node.get("outputs", [])
                phase = node.get("phase", "unknown")
                started_at = node.get("startedAt", "unknown")
                finished_at = node.get("finishedAt", "unknown")
                resources_duration = node.get("resourcesDuration", {})

                # Prepare the task information
                task_info = {
                    "id": node_id,
                    "namespace": namespace,
                    "podName": node_id,
                    "name": node["displayName"],
                    "inputs": inputs,  # Include inputs
                    "outputs": outputs,  # Include outputs
                    "status": phase,
                    "startedAt": started_at,
                    "finishedAt": finished_at,
                    "resourcesDuration": resources_duration,
                    "children": [],
                }

                # Add task to the parent
                if parent is None:
                    task_structure[node_id] = task_info
                else:
                    parent["children"].append(task_info)

                # Recursively traverse child nodes
                if "children" in node and node["children"]:
                    for child_id in node["children"]:
                        traverse(child_id, task_info)

            # Start traversing from the root node
            if root_node_id:
                traverse(root_node_id)

            # Store the task structure for the current run_id
            task_structures[run_id] = {
                "pipeline_workflow_name": pipeline_workflow_name,
                "task_structure": task_structure,
            }

            # Append details to the output list
            output_details.append(
                {
                    "run_id": run_id,
                    "pipeline_workflow_name": pipeline_workflow_name,
                    "namespace": namespace,
                    "task_structure": task_structure,
                }
            )

        return output_details

    @staticmethod
    def get_pod_events(podname, namespace=KubeflowPlugin().get_default_namespace()):
        """
        Fetches Kubernetes events for a specific pod in a namespace.

        Args:
            podname (str): The name of the pod for which events are being fetched.
            namespace (str): The namespace of the pod.

        Returns:
            dict: A dictionary with event details related to the specified pod.
        """
        NotebookPlugin().load_k8s_config()

        # Initialize the CoreV1Api client
        v1 = client.CoreV1Api()
        event = {}
        try:
            # Fetch all events in the namespace
            events = v1.list_namespaced_event(namespace=namespace)

            event["kind"] = events.kind
            event["apiVersion"] = events.api_version
            event["metadata_resource_version"] = events.metadata.resource_version
            event["items"] = events.items
            return event
        except client.exceptions.ApiException as e:
            return {
                "error": f"Failed to fetch events: {str(e)}",
                "podname": podname,
                "namespace": namespace,
            }

    @staticmethod
    def convert_datetime(obj):
        """Recursively converts datetime objects to string."""
        if isinstance(obj, dict):
            return {k: NotebookPlugin().convert_datetime(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [NotebookPlugin().convert_datetime(i) for i in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()  # Convert datetime to string in ISO format
        else:
            return obj

    @staticmethod
    def get_pod_definition(podname, namespace=KubeflowPlugin().get_default_namespace()):
        """
        Fetches the pod definition for a specific pod in a specific namespace.

        Args:
            podname (str): The name of the pod for which the definition is being fetched.
            namespace (str): The namespace of the pod.

        Returns:
            dict: A dictionary with the full pod definition (specifications and metadata).
        """
        # Load Kubernetes configuration (use load_incluster_config() if running within a Kubernetes pod)
        NotebookPlugin().load_k8s_config()

        # Initialize the CoreV1Api client
        v1 = client.CoreV1Api()

        try:
            # Fetch the pod definition
            pod = v1.read_namespaced_pod(name=podname, namespace=namespace)

            # Convert the pod object to a dictionary and then recursively handle datetime objects
            pod_dict = pod.to_dict()
            pod_dict = NotebookPlugin().convert_datetime(
                pod_dict
            )  # Convert any datetime objects to strings
            pod_dict = json.dumps(pod_dict, indent=4)
            return pod_dict

        except client.exceptions.ApiException as e:
            return {
                "error": f"Failed to fetch pod definition: {str(e)}",
                "podname": podname,
                "namespace": namespace,
            }

    @staticmethod
    def get_deployments(namespace):
        """
        Fetches details of all InferenceServices in the given namespace and formats them.

        Args:
        - namespace (str): The Kubernetes namespace where InferenceServices are deployed. Defaults to "default".

        Returns:
        - list of dicts: A list of dictionaries with InferenceService details.
        """
        try:
            # Load Kubernetes configuration from the cluster (in-cluster configuration)
            config.load_incluster_config()  # Use this if running inside a cluster
            # config.load_kube_config()  # Uncomment this for local development with kubeconfig

            # Create the CustomObjectsApi to interact with KServe resources
            custom_api = client.CustomObjectsApi()

            # Fetch InferenceServices in the specified namespace
            kserve_deployments = custom_api.list_namespaced_custom_object(
                group="serving.kserve.io",  # KServe custom resources are in this API group
                version="v1beta1",  # KServe InferenceService typically uses v1beta1
                namespace=namespace,
                plural="inferenceservices",
            )

            inferenceservices_details = []

            # Iterate through the InferenceServices
            for isvc in kserve_deployments.get("items", []):
                # Extract relevant information from metadata and status
                name = isvc["metadata"]["name"]
                url = isvc["status"].get("url", "N/A")

                # Determine the ready status from conditions (if status is "True" for type "Ready")
                ready = (
                    "True"
                    if any(
                        condition.get("type") == "Ready"
                        and condition.get("status") == "True"
                        for condition in isvc["status"].get("conditions", [])
                    )
                    else "False"
                )

                # Get the previous, latest, and ready revisions
                latest_ready_revision = (
                    isvc["status"]
                    .get("components", {})
                    .get("predictor", {})
                    .get("latestReadyRevision", "N/A")
                )

                # Calculate the age of the InferenceService (from creationTimestamp)
                creation_timestamp = isvc["metadata"].get("creationTimestamp", None)
                if creation_timestamp:
                    creation_time = datetime.strptime(
                        creation_timestamp, "%Y-%m-%dT%H:%M:%SZ"
                    )
                    age = str(datetime.now() - creation_time).split(".", maxsplit=1)[0]

                else:
                    age = "Unknown"

                # Collect all the details into a dictionary
                inferenceservices_details.append(
                    {
                        "NAME": name,
                        "URL": url,
                        "READY": ready,
                        "LATESTREADYREVISION": latest_ready_revision,
                        "AGE": age,
                    }
                )

            return inferenceservices_details

        except ApiException as e:
            print(f"Exception when calling API: {e}")
            return []

    @staticmethod
    def calculate_duration(start_time, end_time):
        """
        Calculate the duration between two datetime objects.

        Args:
            start_time (datetime): The start time of the run.
            end_time (datetime): The end time of the run.

        Returns:
            str: Duration in the format 'HH:MM:SS'. If either start_time or end_time
                 is None, returns "N/A".
        """
        if start_time and end_time:
            duration = end_time - start_time
            return str(timedelta(seconds=duration.total_seconds()))
        return "N/A"

    @staticmethod
    def parse_runs(runs_response):
        """
        Parse the list of Kubeflow Pipeline (KFP) runs into a structured format.

        Args:
            runs_response (kfp_server_api.models.ApiListRunsResponse): The response object from
                the KFP `list_runs` API containing the runs information.

        Returns:
            list[dict]: A list of dictionaries where each dictionary contains the following:
                - "run_name" (str): The name of the run.
                - "run_id" (str): The unique ID of the run.
                - "status" (str): The current status of the run (e.g., "Succeeded", "Failed").
                - "duration" (str): The duration of the run in 'HH:MM:SS' format or "N/A".
                - "experiment_id" (str): The ID of the experiment associated with the run.
                - "start_time" (str): The start time of the run in ISO 8601 format or "N/A".
        """
        parsed_runs = []
        for run in runs_response.runs:
            run_name = run.name
            run_id = run.id
            start_time = run.created_at.isoformat() if run.created_at else "N/A"
            experiment_id = None
            for ref in run.resource_references:
                if ref.key.type == "EXPERIMENT":
                    experiment_id = ref.key.id
                    break
            status = run.status
            duration = NotebookPlugin().calculate_duration(
                run.created_at, run.finished_at
            )
            parsed_runs.append(
                {
                    "run_name": run_name,
                    "run_id": run_id,
                    "status": status,
                    "duration": duration,
                    "experiment_id": experiment_id,
                    "start_time": start_time,
                }
            )
        return parsed_runs

    @staticmethod
    def list_all_kfp_runs():
        """
        List all Kubeflow Pipeline (KFP) runs by iterating through all pages of results.

        This method retrieves and parses all available KFP runs using the KFP client API,
        handling pagination via `next_page_token`.

        Returns:
            list[dict]: A list of parsed runs, where each run is represented as a dictionary
                        containing:
                - "run_name" (str): The name of the run.
                - "run_id" (str): The unique ID of the run.
                - "status" (str): The current status of the run (e.g., "Succeeded", "Failed").
                - "duration" (str): The duration of the run in 'HH:MM:SS' format or "N/A".
                - "experiment_id" (str): The ID of the experiment associated with the run.
                - "start_time" (str): The start time of the run in ISO 8601 format or "N/A".
        """
        parsed_runs = []
        next_page_token = None
        kfp_client = KubeflowPlugin().client()
        while True:
            runs_response = kfp_client.list_runs(page_token=next_page_token)
            # print(runs_response)# Fetch KFP runs
            if runs_response and runs_response.runs:
                parsed_runs.extend(NotebookPlugin().parse_runs(runs_response))
            next_page_token = runs_response.next_page_token
            if not next_page_token:  # Stop when there are no more pages
                break

        return parsed_runs
