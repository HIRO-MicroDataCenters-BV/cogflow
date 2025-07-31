"""
knative eventing plugin
"""

import re
from kubernetes import client, dynamic
from kubernetes.dynamic import DynamicClient
from kubernetes.client.exceptions import ApiException
from kubernetes.client import api_client, ApiClient
from kubernetes.client import (
    V1ObjectMeta,
    V1Deployment,
    V1DeploymentSpec,
    V1LabelSelector,
    V1PodTemplateSpec,
    V1PodSpec,
    V1Container,
    V1VolumeMount,
    V1Volume,
    V1ConfigMapVolumeSource,
    V1ConfigMap,
)
from .. import util, plugin_config
from .kubeflowplugin import KubeflowPlugin


class KnativePlugin:
    """
    Class for defining reusable components.
    """

    def __init__(self):
        """
        Initializes the KnativePlugin class.
        """
        self.section = "knative_plugin"

    @staticmethod
    def check_dataset_exists(api_url, dataset_name, query_params=None):
        """
        Checks if a dataset with the exact name exists in the DB.

        :param api_url: URL to the dataset API endpoint.
        :param dataset_name: Name of the dataset to look for (exact match).
        :param query_params: Optional additional query parameters.
        :return: Boolean indicating existence, and matched dataset(s) if any.
        """
        if query_params is None:
            query_params = {"limit": 10}
        datasets = util.make_get_request(
            api_url, query_params=query_params, paginate=True
        )

        matching = [item for item in datasets if item["dataset_name"] == dataset_name]

        if matching:
            return True, matching
        else:
            print(f"No dataset found with name: {dataset_name}")
            return False, []

    @staticmethod
    def get_broker_and_topic_by_dataset_id(api_url, dataset_id, query_params=None):
        """
        Fetch broker_name and topic_name if the dataset_id matches.

        :param api_url: API endpoint that returns dataset message details.
        :param dataset_id: ID of the dataset to validate.
        :param query_params: Optional query parameters.
        :return: Tuple (exists_flag, broker_name, topic_name)
        """
        try:
            path_params = f"{dataset_id}/message/details"

            response = util.make_get_request(
                api_url, path_params=path_params, query_params=query_params
            )

            if not isinstance(response, dict):
                print("Unexpected response format")
                return False, None, None, None

            data = response.get("data", {})

            dataset = data.get("dataset")
            broker = data.get("broker_details")
            topic = data.get("topic_details")

            if dataset and dataset.get("id") == dataset_id:
                broker_name = broker.get("broker_name") if broker else None
                broker_port = broker.get("broker_port") if broker else None
                topic_name = topic.get("topic_name") if topic else None
                return True, broker_name, broker_port, topic_name
            else:
                print(f"No matching dataset ID: {dataset_id}")
                return False, None, None, None

        except Exception as e:
            print(f"Error occurred: {e}")
            return False, None, None, None

    @staticmethod
    def deploy_kafka_sink(name: str, namespace: str, topic: str, bootstrap_server: str):
        """
        Deploy or patch a KafkaSink in the given namespace.
        Args:
            name (str): KafkaSink name
            namespace (str): Kubernetes namespace
            topic (str): Kafka topic
            bootstrap_server (str): Kafka bootstrap server (e.g. kafka:9092)
        """

        KubeflowPlugin().load_k8s_config()
        k8s_client = client.ApiClient()
        dyn_client = DynamicClient(k8s_client)

        kafka_sink_manifest = {
            "apiVersion": "eventing.knative.dev/v1alpha1",
            "kind": "KafkaSink",
            "metadata": {"name": name, "namespace": namespace},
            "spec": {
                "topic": topic,
                "bootstrapServers": [bootstrap_server],  # list
                "contentMode": "binary",
            },
        }
        try:
            kafka_sink_resource = dyn_client.resources.get(
                api_version="eventing.knative.dev/v1alpha1", kind="KafkaSink"
            )

            # Try to get the KafkaSink
            try:
                kafka_sink_resource.get(name=name, namespace=namespace)

                print(f"KafkaSink '{name}' exists. Patching...")

                kafka_sink_resource.patch(
                    body=kafka_sink_manifest,
                    name=name,
                    namespace=namespace,
                    content_type="application/merge-patch+json",
                )

            except ApiException as e:
                if e.status == 404:

                    print(f"KafkaSink '{name}' not found. Creating...")

                    kafka_sink_resource.create(
                        body=kafka_sink_manifest, namespace=namespace
                    )
                else:
                    raise

            print(
                f"KafkaSink '{name}' successfully deployed in namespace '{namespace}'."
            )

        except Exception as e:
            print(f"Failed to deploy KafkaSink: {e}")

    @staticmethod
    def deploy_kafka_source(
        namespace,
        bootstrap_server,
        topic,
        source_name=None,
        consumer_group=None,
        sequence_name=None,
    ):
        """
        Deploy a Knative KafkaSource to consume messages from a Kafka topic
        and route them to a Knative Sequence.

        This function uses the dynamic Kubernetes client to create a
        KafkaSource custom resource that listens to a specified Kafka topic
        and sends messages to a Sequence sink in the same namespace.

        Parameters:
            namespace (str): Kubernetes namespace where the KafkaSource and Sequence exist.
            bootstrap_server (str): Kafka bootstrap server address (e.g., "kafka-cluster-kafka-bootstrap.kafka:9092").
            topic (str): Kafka topic name to subscribe to.
            source_name (str, optional): Name for the KafkaSource resource. Defaults to None.
            consumer_group (str, optional): Kafka consumer group ID. Defaults to None.
            sequence_name (str, optional): Name of the Knative Sequence to use as the sink.
            Defaults to None.

        Returns:
            None

        Raises:
            Prints errors if creation fails or if resource already exists.
        """
        # Load kubeconfig
        KubeflowPlugin().load_k8s_config()

        # Dynamic client
        dyn_client = dynamic.DynamicClient(api_client.ApiClient())

        # Load KafkaSource CRD resource
        kafka_source_api = dyn_client.resources.get(
            api_version="sources.knative.dev/v1beta1", kind="KafkaSource"
        )

        # Define KafkaSource manifest
        kafka_source_manifest = {
            "apiVersion": "sources.knative.dev/v1beta1",
            "kind": "KafkaSource",
            "metadata": {"name": source_name, "namespace": namespace},
            "spec": {
                "bootstrapServers": [bootstrap_server],
                "consumerGroup": consumer_group,
                "topics": [topic],
                "sink": {
                    "ref": {
                        "apiVersion": "flows.knative.dev/v1",
                        "kind": "Sequence",
                        "name": sequence_name,
                        "namespace": namespace,
                    }
                },
            },
        }

        try:
            # Create KafkaSource
            kafka_source_api.create(body=kafka_source_manifest, namespace=namespace)
            print(f"KafkaSource '{source_name}' created in namespace '{namespace}'")
        except Exception as e:
            if "AlreadyExists" in str(e):
                print(
                    f"KafkaSource '{source_name}' already exists. You may need to delete and recreate "
                    f"it if updating immutable fields."
                )
            else:
                print(f"Failed to create KafkaSource: {e}")

    @staticmethod
    def deploy_sequence(sequence_name, namespace, model_isvc_name, kafka_sink_name):
        """
        Deploy a Knative Sequence that sends data to a model and replies to a KafkaSink.

        Args:
            sequence_name (str): Name for the Knative Sequence.
            namespace (str): Kubernetes namespace.
            model_isvc_name (str): Name of the KServe model (InferenceService).
            kafka_sink_name (str): Name of the KafkaSink to route replies to.
        """
        try:
            # Load kubeconfig (in-cluster or local)
            KubeflowPlugin().load_k8s_config()

            # Create dynamic client from loaded config
            k8s_client = ApiClient()
            dyn_client = DynamicClient(k8s_client)

            # Define Sequence manifest
            sequence_manifest = {
                "apiVersion": "flows.knative.dev/v1",
                "kind": "Sequence",
                "metadata": {
                    "name": sequence_name,
                    "namespace": namespace,
                },
                "spec": {
                    "channelTemplate": {
                        "apiVersion": "messaging.knative.dev/v1",
                        "kind": "InMemoryChannel",
                    },
                    "steps": [
                        {
                            "uri": f"http://{model_isvc_name}-predictor.{namespace}.svc.cluster.local/v1/models"
                            f"/{model_isvc_name}:predict"
                        }
                    ],
                    "reply": {
                        "ref": {
                            "apiVersion": "eventing.knative.dev/v1alpha1",
                            "kind": "KafkaSink",
                            "name": kafka_sink_name,
                            "namespace": namespace,
                        }
                    },
                },
            }

            # Deploy
            sequence_api = dyn_client.resources.get(
                api_version="flows.knative.dev/v1", kind="Sequence"
            )

            try:
                # Check if it exists
                sequence_api.get(name=sequence_name, namespace=namespace)
                print(f"Sequence '{sequence_name}' exists. Patching...")
                response = sequence_api.patch(
                    name=sequence_name,
                    namespace=namespace,
                    body=sequence_manifest,
                    content_type="application/merge-patch+json",
                )
            except ApiException as e:
                if e.status == 404:
                    print(f"Creating Sequence '{sequence_name}'...")
                    response = sequence_api.create(
                        body=sequence_manifest, namespace=namespace
                    )
                else:
                    raise

            print(
                f"Sequence '{response.metadata.name}' deployed in namespace '{namespace}'."
            )

        except ApiException as e:
            print(f"Failed to deploy Sequence: {e.status}")
            print("Reason:", e.reason)
            print("Details:", e.body)

    @staticmethod
    def create_configmap_nats_kafka_bridge(
        namespace: str, configmap_name: str, config_data: str
    ):
        """
        Create or update a Kubernetes ConfigMap for the NATS-Kafka bridge configuration.

        If the ConfigMap already exists, this method merges new connector entries into
        the existing 'connectors.conf' while preventing duplication. If it doesn't exist,
        a new ConfigMap is created.

        Parameters:
            namespace (str): Kubernetes namespace for the ConfigMap.
            configmap_name (str): Name of the ConfigMap.
            config_data (str): YAML-like bridge config (as a string) to include.

        Returns:
            None

        Raises:
            kubernetes.client.rest.ApiException: For Kubernetes API errors other than 404.
        """
        v1 = client.CoreV1Api()
        key = "connectors.conf"

        try:
            existing_configmap = v1.read_namespaced_config_map(
                name=configmap_name, namespace=namespace
            )
            existing_data = existing_configmap.data.get(key, "")

            # Extract connect arrays from both existing and new config strings
            existing_connectors = re.findall(r"{[^{}]+}", existing_data)
            new_connectors = re.findall(r"{[^{}]+}", config_data)

            # Merge unique entries (basic string-level match)
            merged_connectors = existing_connectors.copy()
            for connector in new_connectors:
                if connector not in existing_connectors:
                    merged_connectors.append(connector)

            # Replace the connect block
            common_preamble = re.split(
                r"connect\s*:\s*\[", existing_data or config_data
            )[0].strip()
            merged_config = (
                f"{common_preamble}\nconnect: [\n  "
                + ",\n  ".join(merged_connectors)
                + "\n]"
            )

            print(f"Updating ConfigMap '{configmap_name}' with merged connectors...")
            existing_configmap.data[key] = merged_config
            v1.replace_namespaced_config_map(
                name=configmap_name, namespace=namespace, body=existing_configmap
            )

        except ApiException as e:
            if e.status == 404:
                print(f"Creating ConfigMap '{configmap_name}'...")
                config_map = V1ConfigMap(
                    metadata=V1ObjectMeta(name=configmap_name, namespace=namespace),
                    data={key: config_data},
                )
                v1.create_namespaced_config_map(namespace=namespace, body=config_map)
            else:
                raise

    @staticmethod
    def deploy_nats_kafka_bridge_deployment(
        namespace: str, deployment_name: str, configmap_name: str
    ):
        """
        Creates or updates a Kubernetes Deployment for the NATS-Kafka bridge.

        This function defines a Deployment resource for running the `natsio/nats-kafka` bridge
        container, which connects NATS subjects to Kafka topics based on configuration from a
        specified ConfigMap. It mounts the configuration file from the ConfigMap at
        `/etc/bridge/connectors.conf`.

        If the Deployment already exists in the given namespace, it will be updated.
        Otherwise, a new Deployment will be created.

        Parameters:
            namespace (str): The Kubernetes namespace where the Deployment should be created or updated.
            deployment_name (str): The name of the Deployment.
            configmap_name (str): The name of the ConfigMap containing the bridge configuration.

        Returns:
            None

        Raises:
            kubernetes.client.rest.ApiException: If the Kubernetes API call fails for reasons
            other than the Deployment not being found (404).
        """
        apps_api = client.AppsV1Api()
        labels = {"app": deployment_name}

        deployment = V1Deployment(
            metadata=V1ObjectMeta(name=deployment_name, namespace=namespace),
            spec=V1DeploymentSpec(
                replicas=1,
                selector=V1LabelSelector(match_labels=labels),
                template=V1PodTemplateSpec(
                    metadata=V1ObjectMeta(labels=labels),
                    spec=V1PodSpec(
                        containers=[
                            V1Container(
                                name="bridge",
                                image="natsio/nats-kafka:latest",
                                args=["-c", "/etc/bridge/connectors.conf"],
                                volume_mounts=[
                                    V1VolumeMount(
                                        name="config", mount_path="/etc/bridge"
                                    )
                                ],
                            )
                        ],
                        volumes=[
                            V1Volume(
                                name="config",
                                config_map=V1ConfigMapVolumeSource(name=configmap_name),
                            )
                        ],
                    ),
                ),
            ),
        )

        try:
            apps_api.read_namespaced_deployment(
                name=deployment_name, namespace=namespace
            )
            print(f"Updating Deployment '{deployment_name}'...")
            apps_api.replace_namespaced_deployment(
                name=deployment_name, namespace=namespace, body=deployment
            )
            print(
                f"Deployment '{deployment_name}' updated successfully in namespace '{namespace}'."
            )
        except ApiException as e:
            if e.status == 404:
                print(f"Creating Deployment '{deployment_name}'...")
                apps_api.create_namespaced_deployment(
                    namespace=namespace, body=deployment
                )
                print(
                    f"Deployment '{deployment_name}' created successfully in namespace '{namespace}'."
                )
            else:
                raise

    @staticmethod
    def connect(source_dataset, model_isvc, destination_dataset):
        """
        normally in cogflow you check each dataset ,
        1) all of them should be streaming type
        2) then create sink and source and sequnce for them
        3) for source and destination if the type is nats , create bridge for each of them as well
        """

        dataset_url = (
            plugin_config.API_BASEPATH + plugin_config.MESSAGE_BROKER_DATASETS_URL
        )
        namespace = KubeflowPlugin().get_default_namespace()

        # check if source_dataset exists
        source_exists, source_matched = KnativePlugin().check_dataset_exists(
            dataset_url, source_dataset
        )

        # check if destination_dataset exists
        destination_exists, destination_matched = KnativePlugin().check_dataset_exists(
            dataset_url, destination_dataset
        )

        # Ensure both exist
        if not source_exists:
            print(f"Source dataset {source_dataset} does not exist.")
            return
        if not destination_exists:
            print(f"Destination dataset {destination_dataset} does not exist.")
            return

        # Extract the first matched dataset info
        source_info = source_matched[0]
        dest_info = destination_matched[0]

        if source_info["data_source_type"] != 2 or dest_info["data_source_type"] != 2:
            print("Both datasets must be of streaming type.")
            return
        # Check if both have same data_source_type
        elif source_info["data_source_type"] != dest_info["data_source_type"]:
            print("Source and destination datasets have different data_source_type.")
            return
        # If both are valid streaming datasets
        else:
            # Check for broker and topic details
            (
                source_exists,
                source_broker_name,
                source_broker_port,
                source_topic,
            ) = KnativePlugin().get_broker_and_topic_by_dataset_id(
                api_url=dataset_url,
                dataset_id=source_info["id"],
                query_params={"limit": 10},
            )

            (
                dest_exists,
                _,
                _,
                _,
            ) = KnativePlugin().get_broker_and_topic_by_dataset_id(
                api_url=dataset_url,
                dataset_id=dest_info["id"],
                query_params={"limit": 10},
            )
            if (
                source_info["data_source_type"] == 11
                and dest_info["data_source_type"] == 11
            ):

                KnativePlugin().create_configmap_nats_kafka_bridge(
                    namespace=namespace,
                    configmap_name="nats-kafka-config",
                    config_data=plugin_config.NATS_KAFKA_CONNECTOR_JSON,
                )

                KnativePlugin().deploy_nats_kafka_bridge_deployment(
                    namespace=namespace,
                    deployment_name="nats-kafka-bridge",
                    configmap_name="nats-kafka-config",
                )
            elif source_info["data_source_type"] in [10, 11] and dest_info[
                "data_source_type"
            ] in [10, 11]:

                if source_exists and dest_exists:
                    try:
                        KnativePlugin().deploy_kafka_sink(
                            name=f"{source_topic}-sink",
                            namespace=namespace,
                            topic=f"{source_topic}",
                            bootstrap_server=f"{source_broker_name}:{source_broker_port}",
                        )
                        KnativePlugin().deploy_kafka_source(
                            namespace=namespace,
                            bootstrap_server=f"{source_broker_name}:{source_broker_port}",
                            topic=f"{source_topic}",
                            source_name=f"{source_dataset}-events-source-seq",
                            consumer_group=f"{source_dataset}-kserve-group",
                            sequence_name=f"{source_dataset}-prediction-sequence",
                        )
                        KnativePlugin().deploy_sequence(
                            sequence_name=f"{source_dataset}-prediction-sequence",
                            namespace=namespace,
                            model_isvc_name=model_isvc,
                            kafka_sink_name=f"{source_topic}-sink",
                        )
                    except Exception as e:
                        print(f"Error occurred during deployment: {e}")

                else:
                    print("No matching dataset or invalid response")
