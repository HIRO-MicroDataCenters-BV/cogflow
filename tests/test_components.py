import pytest
from uuid import UUID

from cogflow.core.pipelines.components import (
    _parse_s3_uri,
    parse_component_yaml,
    register_component,
    load_component_from_id,
    cogcomponent,
    download_yaml_from_minio,
)

from cogflow.utils.exceptions import (
    CogflowComponentValidationError,
    CogflowComponentRegistryError,
    CogflowComponentStorageError,
)

# ============================================================
# _parse_s3_uri
# ============================================================


def test_parse_s3_uri_valid_full():
    category, bucket, obj = _parse_s3_uri("s3://cat/bucket/file.yaml")
    assert category == "cat"
    assert bucket == "bucket"
    assert obj == "file.yaml"


def test_parse_s3_uri_valid_simple():
    category, bucket, obj = _parse_s3_uri("s3://bucket/file.yaml")
    assert category is None
    assert bucket == "bucket"
    assert obj == "file.yaml"


def test_parse_s3_uri_invalid():
    with pytest.raises(CogflowComponentValidationError):
        _parse_s3_uri("http://invalid/path")


# ============================================================
# parse_component_yaml
# ============================================================


def test_parse_component_yaml_from_data():
    yaml_data = """
name: test-comp
inputs:
  - x
outputs:
  - y
"""
    parsed = parse_component_yaml(yaml_data=yaml_data)
    assert parsed["name"] == "test-comp"
    assert parsed["inputs"] == ["x"]
    assert parsed["outputs"] == ["y"]


def test_parse_component_yaml_missing_args():
    with pytest.raises(CogflowComponentValidationError):
        parse_component_yaml()


# ============================================================
# register_component
# ============================================================


def test_register_component_create(mocker):
    mocker.patch(
        "cogflow.core.pipelines.components.parse_component_yaml",
        return_value={"name": "comp", "inputs": [], "outputs": []},
    )
    mocker.patch(
        "cogflow.core.pipelines.components._upload_yaml_to_minio",
        return_value="s3://bucket/comp.yaml",
    )
    mocker.patch(
        "cogflow.core.pipelines.components.make_get_request_raw",
        return_value={"data": []},
    )
    mocker.patch(
        "cogflow.core.pipelines.components.make_post_request",
        return_value={"data": {"id": "123"}},
    )
    mocker.patch("builtins.open", mocker.mock_open(read_data=b"yaml"))

    result = register_component(yaml_data="name: comp")
    assert result["id"] == "123"


def test_register_component_existing_no_overwrite(mocker):
    mocker.patch(
        "cogflow.core.pipelines.components.parse_component_yaml",
        return_value={"name": "comp", "inputs": [], "outputs": []},
    )
    mocker.patch(
        "cogflow.core.pipelines.components._upload_yaml_to_minio",
        return_value="s3://bucket/comp.yaml",
    )
    mocker.patch(
        "cogflow.core.pipelines.components.make_get_request_raw",
        return_value={"data": [{"id": "existing"}]},
    )
    mocker.patch("builtins.open", mocker.mock_open(read_data=b"yaml"))

    with pytest.raises(CogflowComponentValidationError):
        register_component(yaml_data="name: comp", overwrite=False)


# ============================================================
# load_component_from_id
# ============================================================


def test_load_component_from_id_success(mocker):
    fake_minio = mocker.Mock()
    fake_obj = mocker.Mock()
    fake_obj.read.return_value = b"name: comp"
    fake_minio.get_object.return_value = fake_obj

    mocker.patch(
        "cogflow.core.pipelines.components.make_get_request",
        return_value={"data": {"component_file": "s3://bucket/comp.yaml"}},
    )
    mocker.patch(
        "cogflow.core.pipelines.components.minio_client",
        return_value=fake_minio,
    )

    fake_orc = mocker.Mock()
    fake_orc.load_component_from_text.return_value = "component"
    fake_orc._inject_env_into_container_op.return_value = "component"

    mocker.patch(
        "cogflow.core.pipelines.components._orc",
        return_value=fake_orc,
    )

    result = load_component_from_id(UUID("12345678123456781234567812345678"))
    assert result == "component"


def test_load_component_from_id_missing_metadata(mocker):
    mocker.patch(
        "cogflow.core.pipelines.components.make_get_request",
        return_value={"data": None},
    )

    with pytest.raises(CogflowComponentRegistryError):
        load_component_from_id(UUID("12345678123456781234567812345678"))


# ============================================================
# cogcomponent decorator
# ============================================================


def test_cogcomponent_no_register(mocker):
    fake_orc = mocker.Mock()
    fake_orc.create_component_from_func.return_value = "component-op"

    mocker.patch(
        "cogflow.core.pipelines.components._orc",
        return_value=fake_orc,
    )

    @cogcomponent(register=False)
    def my_func():
        pass

    assert my_func == "component-op"


def test_cogcomponent_with_register(mocker):
    fake_comp = mocker.Mock()
    fake_comp.component_spec.to_dict.return_value = {"name": "x"}
    fake_comp.component_spec.name = "x"

    fake_orc = mocker.Mock()
    fake_orc.create_component_from_func.return_value = fake_comp

    mocker.patch(
        "cogflow.core.pipelines.components._orc",
        return_value=fake_orc,
    )
    mocker.patch(
        "cogflow.core.pipelines.components.register_component",
        return_value={"id": "123"},
    )

    @cogcomponent(register=True)
    def my_func():
        pass

    assert my_func == fake_comp


# ============================================================
# download_yaml_from_minio
# ============================================================


def test_download_yaml_from_minio_success(mocker):
    fake_minio = mocker.Mock()
    mocker.patch(
        "cogflow.core.pipelines.components.minio_client",
        return_value=fake_minio,
    )

    download_yaml_from_minio("bucket", "file.yaml", "/tmp/file.yaml")
    fake_minio.fget_object.assert_called_once()


def test_download_yaml_from_minio_failure(mocker):
    fake_minio = mocker.Mock()
    fake_minio.fget_object.side_effect = Exception("fail")

    mocker.patch(
        "cogflow.core.pipelines.components.minio_client",
        return_value=fake_minio,
    )

    with pytest.raises(CogflowComponentStorageError):
        download_yaml_from_minio("bucket", "file.yaml", "/tmp/file.yaml")
