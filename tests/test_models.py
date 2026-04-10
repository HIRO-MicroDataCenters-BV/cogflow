"""Unit tests for cogflow.core.models.ModelManager."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from cogflow.core import models as models_mod
from cogflow.utils import exceptions as exc_mod

# ---------- Helpers / Fixtures ----------


@pytest.fixture(autouse=True)
def patch_health_check(monkeypatch):
    """
    Avoid real network calls during ModelManager.__init__.
    """
    monkeypatch.setattr(
        models_mod.network,
        "make_health_check_request",
        lambda uri, timeout: None,
    )
    yield


@pytest.fixture
def manager(monkeypatch):
    """Fixture that provides a ModelManager with mocked mlflow and submodules."""
    mm = models_mod.ModelManager(strict=False)

    # Base mlflow mock
    mlflow = MagicMock()
    mlflow.start_run = MagicMock()
    mlflow.end_run = MagicMock()
    mlflow.evaluate = MagicMock()
    mlflow.get_run = MagicMock()
    mlflow.set_experiment = MagicMock()
    mlflow.autolog = MagicMock()
    mlflow.get_artifact_uri = MagicMock(return_value="artifact_base_uri")

    # sklearn stub
    class SklearnModule:
        """Sklearn module stub."""

        class BaseEstimator:
            """Dummy Sklearn BaseEstimator base class."""

            pass

        log_model = MagicMock()
        load_model = MagicMock()

    # pyfunc stub
    class PyfuncModule:
        """PyFunc module stub."""

        class PythonModel:
            """Dummy PyFunc PythonModel base class."""

            pass

        log_model = MagicMock()

    # pytorch stub
    class TorchModule:
        """PyTorch module stub."""

        class Module:
            """Dummy PyTorch Module base class."""

            pass

        log_model = MagicMock()

    # Attach stubs to manager
    mm.sklearn = SklearnModule()
    mm.pyfunc = PyfuncModule()
    mm.pytorch = TorchModule()

    # Attach stubs to mlflow (ModelManager uses THESE)
    mlflow.sklearn = mm.sklearn
    mlflow.pyfunc = mm.pyfunc
    mlflow.pytorch = mm.pytorch

    mlflow.models = MagicMock()
    mm.mlflow = mlflow

    mm.client = MagicMock()
    mm._healthy = True

    monkeypatch.setattr(models_mod.common, "uuid_to_hex", lambda x: x)
    monkeypatch.setattr(models_mod.common, "normalize_uuid", lambda x: x)

    return mm


# ---------- Health / Init ----------


def test_check_tracking_server_health_success(monkeypatch):
    """Test ModelManager health check success path."""
    calls = {}

    def fake_health(uri, timeout):
        """Fake that always succeeds."""
        calls["called"] = (uri, timeout)

    monkeypatch.setattr(models_mod.network, "make_health_check_request", fake_health)
    assert models_mod.ModelManager._check_tracking_server_health() is True
    assert "called" in calls


def test_check_tracking_server_health_failure(monkeypatch):
    """Test ModelManager health check failure path."""

    def fake_health(uri, timeout):
        """Fake that always raises."""
        raise RuntimeError("boom")

    monkeypatch.setattr(models_mod.network, "make_health_check_request", fake_health)

    # Patch error handler to not re-raise anything
    def fake_handle(error, context, raise_as=None, re_raise=False):
        """Fake error handler"""
        # just swallow
        return

    monkeypatch.setattr(models_mod.CogflowErrorHandler, "handle_exception", fake_handle)
    assert models_mod.ModelManager._check_tracking_server_health() is False


def test_model_manager_init_strict_raises(monkeypatch):
    """Test ModelManager init with strict=True and unhealthy server raises."""
    # Force health check to be False
    monkeypatch.setattr(
        models_mod.ModelManager,
        "_check_tracking_server_health",
        lambda self: False,
    )

    # Make log_and_raise actually raise the desired exception
    def fake_log_and_raise(msg, raise_as):
        """Fake that always raises."""
        raise raise_as(msg)

    monkeypatch.setattr(
        models_mod.CogflowErrorHandler, "log_and_raise", fake_log_and_raise
    )

    with pytest.raises(models_mod.CogflowConnectionError):
        models_mod.ModelManager(strict=True)


def test_ensure_health_rechecks(monkeypatch):
    """Test _ensure_health re-checks and updates state."""
    mm = models_mod.ModelManager(strict=False)
    # Simulate unhealthy initially
    mm._healthy = False

    called = {}

    def fake_check(self):
        """Fake health check that returns True."""
        called["ok"] = True
        return True

    monkeypatch.setattr(
        models_mod.ModelManager, "_check_tracking_server_health", fake_check
    )

    assert mm._ensure_health() is True
    assert called["ok"] is True
    assert mm._healthy is True


def test_warn_if_unhealthy_logs(monkeypatch):
    """Test _warn_if_unhealthy does not raise but logs."""
    mm = models_mod.ModelManager(strict=False)
    mm._healthy = False  # force check

    def fake_check(self):
        """Fake health check that returns False."""
        return False

    monkeypatch.setattr(
        models_mod.ModelManager, "_check_tracking_server_health", fake_check
    )

    # Just ensure no exception
    mm._warn_if_unhealthy("some action")


# ---------- load_model ----------


def test_load_model_success(manager):
    """Test load_model success path."""
    manager.mlflow.sklearn.load_model.return_value = "MODEL"

    result = manager.load_model("runs:/abc/model")

    manager.mlflow.sklearn.load_model.assert_called_once_with("runs:/abc/model", None)

    assert result == "MODEL"


def test_load_model_file_not_found(manager, monkeypatch):
    """Exercise error path in load_model (FileNotFoundError)."""
    manager.mlflow.sklearn.load_model.side_effect = FileNotFoundError("nope")

    def fake_handle(exc, context, raise_as, re_raise):
        """Fake error handler that always raises."""
        raise raise_as(str(exc))

    monkeypatch.setattr(models_mod.CogflowErrorHandler, "handle_exception", fake_handle)

    with pytest.raises(exc_mod.CogflowArtifactError):
        manager.load_model("runs:/missing/model")


# ---------- register_model / create_* / delete_registered_model ----------


def test_register_model_success(manager):
    """Test register_model success path."""
    mock_version = MagicMock()
    mock_version.version = 3
    manager.mlflow.register_model.return_value = mock_version

    result = manager.register_model(
        model_uri="runs:/abc/model", model_name="MyModel", await_registration_for=10
    )

    manager.mlflow.register_model.assert_called_once()
    assert result.version == 3


def test_create_registered_model_success(manager):
    """Test create_registered_model success path."""
    reg = MagicMock()
    manager.client.create_registered_model.return_value = reg

    result = manager.create_registered_model(
        model="AwesomeModel",
        tags={"team": "mlops"},
        description="desc",
    )
    manager.client.create_registered_model.assert_called_once()
    assert result is reg


def test_create_model_version_success(manager):
    """Test create_model_version success path."""
    ver = MagicMock()
    ver.version = 2
    manager.client.create_model_version.return_value = ver

    result = manager.create_model_version(
        model="M", source="s3://bucket/model", run_id="abc123"
    )
    manager.client.create_model_version.assert_called_once()
    assert result.version == 2


def test_delete_registered_model_success(manager):
    """Test delete_registered_model success path."""
    manager.client.delete_registered_model.return_value = None

    ok = manager.delete_registered_model("MyModel")

    manager.client.delete_registered_model.assert_called_once_with("MyModel")
    assert ok is True


def test_delete_registered_model_failure(manager, monkeypatch):
    """Exercise error path in delete_registered_model."""
    manager.client.delete_registered_model.side_effect = RuntimeError("fail")

    def fake_handle(e, context, raise_as, re_raise):
        """Fake error handler that just swallows the error to"""
        # ensure we don't raise
        return

    monkeypatch.setattr(models_mod.CogflowErrorHandler, "handle_exception", fake_handle)

    ok = manager.delete_registered_model("X")
    assert ok is False


# ---------- evaluate ----------


def test_evaluate_success(manager):
    """Test evaluate success path."""
    eval_result = MagicMock()
    eval_result.metrics = {"acc": 0.9}
    eval_result.artifacts = {"confusion_matrix": "cm.png"}
    manager.mlflow.evaluate.return_value = eval_result

    result = manager.evaluate(
        model_uri="runs:/abc/model",
        data=[[1, 2], [3, 4]],
        targets=[0, 1],
        model_type="classifier",
        evaluators=["default"],
    )

    manager.mlflow.evaluate.assert_called_once()
    assert result is eval_result


# ---------- log_model (sklearn, pyfunc, pytorch, error) ----------


class BaseEstimator:
    """Dummy base class for Sklearn models."""

    pass


class DummySkModel(BaseEstimator):
    """Dummy Sklearn model."""

    pass


class Module:
    """Dummy PyTorch Module."""

    pass


class DummyTorchModel(Module):
    """Dummy PyTorch model."""

    pass


def test_log_model_sklearn_branch(manager):
    """Test log_model with Sklearn model."""

    class DummySkModel(manager.sklearn.BaseEstimator):
        """Dummy Sklearn model."""

        pass

    model = DummySkModel()
    manager.sklearn.log_model.return_value = "SK_RESULT"

    result = manager.log_model(model, artifact_path="model")

    manager.sklearn.log_model.assert_called_once()
    assert result == "SK_RESULT"


def test_log_model_pyfunc_branch(manager):
    """Test log_model with PyFunc model."""

    class DummyPyModel(manager.pyfunc.PythonModel):
        """Dummy PyFunc model."""

        pass

    model = DummyPyModel()
    manager.pyfunc.log_model.return_value = "PYFUNC_RESULT"

    result = manager.log_model(model, artifact_path="pyfunc_model")

    manager.pyfunc.log_model.assert_called_once()
    assert result == "PYFUNC_RESULT"


@pytest.mark.skipif("torch" not in globals(), reason="torch not installed")
def test_log_model_pytorch_branch(manager):
    """Test log_model with PyTorch model."""
    from torch import nn

    class DummyTorchModel(nn.Module):
        """Dummy PyTorch model."""

        pass

    model = DummyTorchModel()
    manager.pytorch.log_model.return_value = "PT_RESULT"
    result = manager.log_model(model, artifact_path="pt_model")
    manager.pytorch.log_model.assert_called_once()
    assert result == "PT_RESULT"


def test_log_model_unsupported_type_raises(manager, monkeypatch):
    """
    Exercise the error path in log_model (unsupported model type).
    """

    model = object()

    def fake_handle(e, context, raise_as, re_raise):
        """Fake error handler that always raises."""
        raise raise_as(str(e))

    monkeypatch.setattr(models_mod.CogflowErrorHandler, "handle_exception", fake_handle)

    with pytest.raises(models_mod.CogflowModelError):
        manager.log_model(model, artifact_path="unknown")


# ---------- search_model_versions / get_model_uri ----------


def test_search_model_versions_success(manager):
    """Test search_model_versions."""
    mv1 = MagicMock()
    mv2 = MagicMock()
    manager.client.search_model_versions.return_value = [mv1, mv2]

    result = manager.search_model_versions("name='MyModel'")
    manager.client.search_model_versions.assert_called_once()
    assert len(result) == 2


def test_get_model_uri(manager):
    """Test get_model_uri."""
    mv = MagicMock()
    mv.source = "s3://bucket/model"
    manager.client.get_model_version.return_value = mv

    uri = manager.get_model_uri("MyModel", "3")
    manager.client.get_model_version.assert_called_once_with(
        name="MyModel", version="3"
    )
    assert uri == "s3://bucket/model"


# ---------- get_full_model_uri_from_run_or_registry ----------


@dataclass
class MockArtifact:
    """Mock class for MLflow Artifact with path and is_dir."""

    path: str
    is_dir: bool


def test_get_full_model_uri_run_id_single_dir(manager):
    """Test with run_id and single directory artifact."""
    # Run with single directory artifact
    run_info = MagicMock()
    run_info.artifact_uri = "s3://mlflow/0/abc/artifacts"
    run = MagicMock()
    run.info = run_info
    manager.mlflow.get_run.return_value = run

    # list_artifacts returns one dir
    manager.client.list_artifacts.return_value = [
        MockArtifact(path="model_dir", is_dir=True)
    ]

    # search_model_versions -> name/version backfill
    mv = MagicMock()
    mv.name = "BackfillModel"
    mv.version = "1"
    manager.search_model_versions = MagicMock(return_value=[mv])

    info = manager.get_full_model_uri_from_run_or_registry(model_id="abc")

    assert info["model_uri"] == "s3://mlflow/0/abc/artifacts/model_dir"
    assert info["model_name"] == "BackfillModel"
    assert info["model_version"] == "1"
    assert info["model_id"] == "abc"


def test_get_full_model_uri_with_artifact_path(manager):
    """Test specifying custom artifact_path."""
    # Run
    run_info = MagicMock()
    run_info.artifact_uri = "uri"
    run = MagicMock()
    run.info = run_info
    manager.mlflow.get_run.return_value = run

    info = manager.get_full_model_uri_from_run_or_registry(
        model_id="abc", artifact_path="custom/model"
    )

    assert info["model_uri"] == "uri/custom/model"


def test_get_full_model_uri_multiple_dirs_raises(manager, monkeypatch):
    """Test error path when multiple directory artifacts found."""
    run_info = MagicMock()
    run_info.artifact_uri = "uri"
    run = MagicMock()
    run.info = run_info
    manager.mlflow.get_run.return_value = run

    manager.client.list_artifacts.return_value = [
        MockArtifact(path="a", is_dir=True),
        MockArtifact(path="b", is_dir=True),
    ]

    def fake_handle(e, context, raise_as, re_raise):
        """Fake error handler that always raises."""
        raise raise_as(str(e))

    monkeypatch.setattr(models_mod.CogflowErrorHandler, "handle_exception", fake_handle)

    with pytest.raises(models_mod.CogflowModelError):
        manager.get_full_model_uri_from_run_or_registry(model_id="abc")


def test_get_full_model_uri_model_file_no_dir(manager):
    """Test error path when no directory artifacts found."""
    run_info = MagicMock()
    run_info.artifact_uri = "uri"
    run = MagicMock()
    run.info = run_info
    manager.mlflow.get_run.return_value = run

    manager.client.list_artifacts.return_value = [
        MockArtifact(path="model.pkl", is_dir=False)
    ]

    info = manager.get_full_model_uri_from_run_or_registry(model_id="abc")
    assert info["model_uri"] == "uri/model.pkl"


def test_get_full_model_uri_no_artifacts_raises(manager, monkeypatch):
    """Test error path when no artifacts found."""
    run_info = MagicMock()
    run_info.artifact_uri = "uri"
    run = MagicMock()
    run.info = run_info
    manager.mlflow.get_run.return_value = run

    manager.client.list_artifacts.return_value = []

    def fake_handle(e, context, raise_as, re_raise):
        """Fake error handler that always raises."""
        raise raise_as(str(e))

    monkeypatch.setattr(models_mod.CogflowErrorHandler, "handle_exception", fake_handle)

    with pytest.raises(models_mod.CogflowModelError):
        manager.get_full_model_uri_from_run_or_registry(model_id="abc")


# ---------- detect_model_format / detect_model_type ----------


class DummyModelInfo:
    """Dummy class for MLflow ModelInfo with flavors."""

    def __init__(self, flavors):
        """Initialize with flavors."""
        self.flavors = flavors


@pytest.mark.parametrize(
    "flavors, expected",
    [
        ({"sklearn": {}}, "sklearn"),
        ({"python_function": {}}, "mlflow"),
        ({}, "unknown"),
    ],
)
def test_detect_model_format(manager, flavors, expected):
    """Test detect_model_format."""
    manager.mlflow.models.get_model_info.return_value = DummyModelInfo(flavors)
    assert manager.detect_model_format("uri") == expected


@pytest.mark.parametrize(
    "flavors, expected",
    [
        ({"sklearn": {}}, "sklearn"),
        ({"python_function": {}}, "pyfunc"),
        ({}, "unknown"),
    ],
)
def test_detect_model_type(manager, flavors, expected):
    """Test detect_model_type."""
    manager.mlflow.models.get_model_info.return_value = DummyModelInfo(flavors)
    assert manager.detect_model_type("uri") == expected


# ---------- start_run / end_run / set_experiment / set_tag ----------


def test_start_run(manager):
    """Test start_run."""
    run = MagicMock()
    run.info.run_id = "RID"
    manager.mlflow.start_run.return_value = run

    r = manager.start_run(run_name="test_run")
    manager.mlflow.start_run.assert_called_once()
    assert r.info.run_id == "RID"


def test_end_run(manager):
    """Test end_run."""
    run = MagicMock()
    manager.mlflow.end_run.return_value = run

    r = manager.end_run()
    manager.mlflow.end_run.assert_called_once()
    assert r is run


def test_set_experiment(manager):
    """Test set_experiment."""
    manager.set_experiment(experiment_name="exp1")
    manager.mlflow.set_experiment.assert_called_once_with(
        experiment_name="exp1", experiment_id=None
    )


def test_set_tag(manager):
    """Test set_tag."""
    manager.set_tag("team", "mlops")
    manager.mlflow.set_tag.assert_called_once_with(key="team", value="mlops")


# ---------- get_experiment_id_from_run ----------


def test_get_experiment_id_from_run_success(manager, monkeypatch):
    """Test success path for get_experiment_id_from_run."""
    run = MagicMock()
    run.info.experiment_id = "123"
    manager.mlflow.get_run.return_value = run

    # Use trivial uuid_to_hex
    monkeypatch.setattr(models_mod.common, "uuid_to_hex", lambda x: x)

    exp_id = manager.get_experiment_id_from_run("abc")
    manager.mlflow.get_run.assert_called_once_with("abc")
    assert exp_id == "123"


def test_get_experiment_id_from_run_invalid_uuid(manager, monkeypatch):
    """Test error path for get_experiment_id_from_run with invalid UUID."""

    def bad_uuid(x):
        """Function that always raises."""
        raise ValueError("Invalid UUID value")

    monkeypatch.setattr(models_mod.common, "uuid_to_hex", bad_uuid)

    def fake_handle(e, context, raise_as, re_raise):
        """Fake error handler that always raises."""
        raise raise_as(str(e))

    monkeypatch.setattr(models_mod.CogflowErrorHandler, "handle_exception", fake_handle)

    with pytest.raises(exc_mod.CogflowRunError):
        manager.get_experiment_id_from_run("abc")


# ---------- search_runs ----------


def test_search_runs_success(manager):
    """Test success path for search_runs."""
    r1, r2 = MagicMock(), MagicMock()
    manager.client.search_runs.return_value = [r1, r2]

    results = manager.search_runs(
        experiment_ids=["1"], filter_string="metrics.acc > 0.9"
    )
    manager.client.search_runs.assert_called_once()
    assert len(results) == 2


def test_search_runs_failure(manager, monkeypatch):
    """Test error path for search_runs."""
    manager.client.search_runs.side_effect = RuntimeError("boom")

    def fake_handle(e, context, raise_as, re_raise):
        """Fake error handler that always raises."""
        raise raise_as(str(e))

    monkeypatch.setattr(models_mod.CogflowErrorHandler, "handle_exception", fake_handle)

    with pytest.raises(exc_mod.CogflowRunError):
        manager.search_runs(experiment_ids=["1"], filter_string="metrics.acc > 0.9")


# ---------- log_param / log_params / log_metric / log_metrics ----------


def test_log_param(manager):
    """Test log_param."""
    manager.log_param("lr", 0.01)
    manager.mlflow.log_param.assert_called_once_with("lr", 0.01)


def test_log_params(manager):
    """Test log_params."""
    params = {"a": 1, "b": 2}
    manager.log_params(params)
    manager.mlflow.log_params.assert_called_once_with(params)


def test_log_metric(manager):
    """Test log_metric."""
    manager.log_metric("acc", 0.9, step=1)
    manager.mlflow.log_metric.assert_called_once_with("acc", 0.9, step=1)


def test_log_metrics(manager):
    """Test log_metrics."""
    metrics = {"loss": 0.1, "val_loss": 0.2}
    manager.log_metrics(metrics, step=2)
    manager.mlflow.log_metrics.assert_called_once_with(metrics, step=2)


# ---------- log_artifact / log_artifacts (with and without run_id) ----------


def test_log_artifact_without_run_id(manager):
    """Test log_artifact without run_id specified."""
    manager.log_artifact("file.txt", artifact_path="reports")
    manager.mlflow.log_artifact.assert_called_once_with(
        local_path="file.txt", artifact_path="reports"
    )


def test_log_artifact_with_run_id(manager):
    """Test log_artifact with run_id specified."""
    manager.log_artifact("file.txt", artifact_path="reports", run_id="abc")
    manager.client.log_artifact.assert_called_once()
    args, kwargs = manager.client.log_artifact.call_args
    assert kwargs["local_path"] == "file.txt"
    assert kwargs["artifact_path"] == "reports"
    # run_id is passed through uuid_to_hex (patched via fixture to be identity)
    assert kwargs["run_id"] == "abc"


def test_log_artifacts_without_run_id(manager):
    """Test log_artifacts without run_id specified."""
    manager.log_artifacts("dir", artifact_path="models")
    manager.mlflow.log_artifacts.assert_called_once_with(
        local_dir="dir", artifact_path="models"
    )


def test_log_artifacts_with_run_id(manager):
    """Test log_artifacts with run_id specified."""
    manager.log_artifacts("dir", artifact_path="models", run_id="abc")
    manager.client.log_artifacts.assert_called_once()
    args, kwargs = manager.client.log_artifacts.call_args
    assert kwargs["local_dir"] == "dir"
    assert kwargs["artifact_path"] == "models"
    assert kwargs["run_id"] == "abc"


# ---------- search_registered_models ----------


def test_search_registered_models_mixed(manager):
    """Test success path for search_registered_models with mixed return types."""

    class HasDict:
        """Class with to_dictionary method."""

        def to_dictionary(self):
            """Return dict representation."""
            return {"name": "A"}

    class NoDict:
        """Class without to_dictionary method."""

        def __init__(self):
            """Initialize."""
            self.x = 1

    manager.client.search_registered_models.return_value = [HasDict(), NoDict()]

    results = manager.search_registered_models(filter_string="name ILIKE '%A%'")
    assert results[0] == {"name": "A"}
    assert "x" in results[1]


def test_search_registered_models_failure(manager, monkeypatch):
    """Test error path for search_registered_models."""
    manager.client.search_registered_models.side_effect = RuntimeError("fail")

    def fake_handle(err, context=None, raise_as=None, re_raise=None):
        """Fake error handler that always raises."""
        raise raise_as(str(err))

    monkeypatch.setattr(models_mod.CogflowErrorHandler, "handle_exception", fake_handle)

    with pytest.raises(exc_mod.CogflowModelError):
        manager.search_registered_models("name='X'")


# ---------- autolog ----------


def test_autolog_success(manager):
    """Test success path for autolog."""
    manager.autolog()
    manager.mlflow.autolog.assert_called_once()


def test_autolog_failure(monkeypatch):
    """Test error path for autolog."""
    mm = models_mod.ModelManager(strict=False)
    mm._healthy = False  # force recheck but that's fine

    def bad_autolog():
        """Function that always raises."""
        raise RuntimeError("nope")

    # Provide mlflow
    mm.mlflow = MagicMock()
    mm.mlflow.autolog.side_effect = bad_autolog

    with pytest.raises(RuntimeError):
        mm.autolog()


# ---------- set_tracking_uri ----------


def test_set_tracking_uri_success(manager):
    """Test success path for set_tracking_uri."""
    manager.set_tracking_uri("http://mlflow.example.com")
    manager.mlflow.set_tracking_uri.assert_called_once_with("http://mlflow.example.com")


def test_set_tracking_uri_failure(manager, monkeypatch):
    """Test error path for set_tracking_uri."""
    manager.mlflow.set_tracking_uri.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError):
        manager.set_tracking_uri("bad://url")


# ---------- get_artifact_uri ----------


def test_get_artifact_uri_success(manager):
    """Test success path for get_artifact_uri."""
    manager.mlflow.get_artifact_uri.return_value = "uri/x"
    uri = manager.get_artifact_uri("x")
    manager.mlflow.get_artifact_uri.assert_called_once_with(artifact_path="x")
    assert uri == "uri/x"


def test_get_artifact_uri_failure(manager):
    """Test error path for get_artifact_uri."""
    manager.mlflow.get_artifact_uri.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        manager.get_artifact_uri("x")


# ---------- create_experiment ----------


def test_create_experiment_success(manager):
    """Test success path for create_experiment."""
    manager.client.create_experiment.return_value = "42"
    exp_id = manager.create_experiment(
        name="my_exp", artifact_location="s3://bucket", tags={"team": "mlops"}
    )
    manager.client.create_experiment.assert_called_once()
    assert exp_id == "42"


def test_create_experiment_failure(manager):
    """
    Test error path for create_experiment.
    """
    manager.client.create_experiment.side_effect = RuntimeError("fail")
    with pytest.raises(RuntimeError):
        manager.create_experiment(name="bad")


# ---------- get_run ----------


def test_get_run_success(manager):
    """Test get_run returns the MLflow Run object."""
    fake_run = MagicMock()
    fake_run.info.run_id = "abc123"
    manager.mlflow.get_run.return_value = fake_run

    result = manager.get_run("abc123")

    manager.mlflow.get_run.assert_called_once_with("abc123")
    assert result is fake_run


def test_get_run_normalizes_uuid(manager, monkeypatch):
    """Test that get_run passes run_id through uuid_to_hex."""
    calls = {}

    def fake_uuid_to_hex(value):
        calls["got"] = value
        return "normalized-id"

    monkeypatch.setattr(models_mod.common, "uuid_to_hex", fake_uuid_to_hex)
    manager.mlflow.get_run.return_value = MagicMock()

    manager.get_run("abc-123-def")

    assert calls["got"] == "abc-123-def"
    manager.mlflow.get_run.assert_called_once_with("normalized-id")


def test_get_run_failure_wraps_exception(manager):
    """Test get_run wraps errors as CogflowRunError."""
    manager.mlflow.get_run.side_effect = RuntimeError("boom")
    with pytest.raises(models_mod.CogflowRunError):
        manager.get_run("abc123")


# ---------- get_run_artifact_uri ----------


def test_get_run_artifact_uri_success(manager):
    """Test get_run_artifact_uri returns the artifact_uri from the run info."""
    fake_run = MagicMock()
    fake_run.info.artifact_uri = "s3://mlflow/0/abc123/artifacts"
    manager.mlflow.get_run.return_value = fake_run

    uri = manager.get_run_artifact_uri("abc123")

    assert uri == "s3://mlflow/0/abc123/artifacts"
    manager.mlflow.get_run.assert_called_once_with("abc123")


def test_get_run_artifact_uri_failure(manager):
    """Test get_run_artifact_uri propagates errors as CogflowRunError."""
    manager.mlflow.get_run.side_effect = RuntimeError("boom")
    with pytest.raises(models_mod.CogflowRunError):
        manager.get_run_artifact_uri("abc123")


# ---------- list_artifacts_grouped ----------


def test_list_artifacts_grouped_root_files_only(manager):
    """Files at root level should be grouped under empty-string key."""
    file1 = MagicMock()
    file1.path = "README.md"
    file1.is_dir = False
    file2 = MagicMock()
    file2.path = "config.json"
    file2.is_dir = False

    manager.client.list_artifacts.return_value = [file1, file2]

    result = manager.list_artifacts_grouped("run-1")

    assert "" in result
    assert sorted(result[""]) == ["README.md", "config.json"]


def test_list_artifacts_grouped_with_directories(manager):
    """Files inside directories should be grouped by directory name."""
    root_file = MagicMock()
    root_file.path = "README.md"
    root_file.is_dir = False

    model_dir = MagicMock()
    model_dir.path = "model"
    model_dir.is_dir = True

    sub_file1 = MagicMock()
    sub_file1.path = "model/model.pkl"
    sub_file1.is_dir = False
    sub_file2 = MagicMock()
    sub_file2.path = "model/config.yaml"
    sub_file2.is_dir = False

    def list_artifacts(run_id, path=None):
        if path == "model":
            return [sub_file1, sub_file2]
        return [root_file, model_dir]

    manager.client.list_artifacts.side_effect = list_artifacts

    result = manager.list_artifacts_grouped("run-1")

    assert result[""] == ["README.md"]
    assert sorted(result["model"]) == ["config.yaml", "model.pkl"]


def test_list_artifacts_grouped_empty_run(manager):
    """Empty run should return empty dict."""
    manager.client.list_artifacts.return_value = []

    result = manager.list_artifacts_grouped("run-1")

    assert result == {}


def test_list_artifacts_grouped_failure(manager):
    """Errors from MLflow client should be wrapped as CogflowRunError."""
    manager.client.list_artifacts.side_effect = RuntimeError("boom")
    with pytest.raises(models_mod.CogflowRunError):
        manager.list_artifacts_grouped("run-1")
