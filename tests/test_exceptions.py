import pytest

from cogflow.utils.exceptions import (
    CogflowError,
    CogflowConnectionError,
    CogflowValidationError,
    CogflowErrorHandler,
)

# ------------------------------------------------------------------
# CogflowError base behavior
# ------------------------------------------------------------------


def test_cogflow_error_str_without_cause():
    err = CogflowError("something went wrong")
    assert str(err) == "[CogFlowError] something went wrong"


def test_cogflow_error_str_with_cause():
    cause = ValueError("bad value")
    err = CogflowError("failed", cause=cause)

    text = str(err)
    assert "failed" in text
    assert "ValueError" in text
    assert "bad value" in text


# ------------------------------------------------------------------
# Error inheritance
# ------------------------------------------------------------------


def test_cogflow_connection_error_is_cogflow_error():
    err = CogflowConnectionError("network down")
    assert isinstance(err, CogflowError)


def test_cogflow_validation_error_is_cogflow_error():
    err = CogflowValidationError("invalid input")
    assert isinstance(err, CogflowError)


# ------------------------------------------------------------------
# to_dict serialization
# ------------------------------------------------------------------


def test_to_dict_with_cogflow_error():
    err = CogflowValidationError("bad param", cause=ValueError("x"))

    data = CogflowErrorHandler.to_dict(err)

    assert data["error_type"] == "CogflowValidationError"
    assert data["message"] == "bad param"
    assert data["cause"] == "x"
    assert "traceback" in data


def test_to_dict_with_plain_exception():
    try:
        raise RuntimeError("boom")
    except Exception as e:
        data = CogflowErrorHandler.to_dict(e)

    assert data["error_type"] == "RuntimeError"
    assert data["message"] == "boom"
    assert "traceback" in data


# ------------------------------------------------------------------
# handle_exception behavior
# ------------------------------------------------------------------


def test_handle_exception_without_raise_as_returns_dict():
    try:
        raise ValueError("oops")
    except Exception as e:
        result = CogflowErrorHandler.handle_exception(e)

    assert result["error_type"] == "ValueError"
    assert result["message"] == "oops"


def test_handle_exception_with_raise_as_wraps_error():
    try:
        raise ValueError("bad input")
    except Exception as e:
        result = CogflowErrorHandler.handle_exception(
            e,
            context="testing",
            raise_as=CogflowValidationError,
        )

    assert result["error_type"] == "CogflowValidationError"
    assert "bad input" in result["message"]


def test_handle_exception_with_reraise_true_raises_wrapped():
    with pytest.raises(CogflowValidationError) as exc:
        try:
            raise ValueError("invalid")
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context="validation",
                raise_as=CogflowValidationError,
                re_raise=True,
            )

    assert isinstance(exc.value.cause, ValueError)


def test_handle_exception_reraise_without_raise_as():
    with pytest.raises(ValueError):
        try:
            raise ValueError("boom")
        except Exception as e:
            CogflowErrorHandler.handle_exception(e, re_raise=True)


# ------------------------------------------------------------------
# log_and_raise
# ------------------------------------------------------------------


def test_log_and_raise_raises_correct_error():
    with pytest.raises(CogflowConnectionError) as exc:
        CogflowErrorHandler.log_and_raise(
            "connection failed",
            raise_as=CogflowConnectionError,
            cause=RuntimeError("timeout"),
        )

    err = exc.value
    assert isinstance(err, CogflowConnectionError)
    assert "connection failed" in err.message
    assert isinstance(err.cause, RuntimeError)
