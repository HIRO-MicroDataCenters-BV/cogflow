import pytest


@pytest.fixture(autouse=True)
def mock_current_user(mocker):
    """
    Always mock current user resolution for all tests.
    Prevents RuntimeError: Unable to resolve current user ID
    """
    mocker.patch(
        "cogflow.utils.common.get_current_user",
        return_value="test-user",
    )
