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
