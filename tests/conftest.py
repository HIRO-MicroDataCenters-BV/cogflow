import pytest

# Neutralize the MLflow tracking-server health check at conftest-import
# time so tests don't depend on a reachable MLflow server.
#
# ``cogflow.core.models`` builds a module-level ``_models = ModelManager()``
# singleton, and ``ModelManager.__init__`` calls
# ``network.make_health_check_request`` — a real HTTP call to whatever
# MLflow URI is configured. An autouse fixture runs per-test, which is
# too late: the first ``import cogflow.core.models`` anywhere in the
# test session already triggered the call.
#
# Approach:
#   1. Swap ``network.make_health_check_request`` for a no-op.
#   2. Force-import ``cogflow.core.models`` now so its singleton builds
#      harmlessly against the stub.
#   3. Restore the real function so ``tests/test_network.py`` (and any
#      future direct users of the helper) still see the genuine
#      behaviour.
# Subsequent ``import cogflow.core.models`` calls are no-ops because
# it's already in ``sys.modules``.
import cogflow.utils.network as _cogflow_network  # noqa: E402

_original_health_check = _cogflow_network.make_health_check_request


def _stub_health_check(*_args, **_kwargs):  # pragma: no cover - helper
    """No-op stand-in for ``network.make_health_check_request``."""
    return None


_cogflow_network.make_health_check_request = _stub_health_check
try:
    import cogflow.core.models  # noqa: E402,F401  — builds _models safely
finally:
    _cogflow_network.make_health_check_request = _original_health_check


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
