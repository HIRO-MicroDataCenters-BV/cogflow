"""
knative eventing plugin
"""

from .. import util


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
    datasets = util.make_get_request(api_url, query_params=query_params, paginate=True)

    matching = [item for item in datasets if item["dataset_name"] == dataset_name]

    if matching:
        # print(f"Dataset '{dataset_name}' exists in the DB")
        return True, matching
    else:
        print(f"No dataset found with name: {dataset_name}")
        return False, []


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
        # print(f"Requesting: {path_params}")
        response = util.make_get_request(
            api_url, path_params=path_params, query_params=query_params
        )

        if not isinstance(response, dict):
            print("Unexpected response format")
            return False, None, None

        data = response.get("data", {})
        # print("Full data:", data)

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
