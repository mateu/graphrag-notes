"""Strict versioned results after Hermes' native registry rendering."""
import json


def decode_envelope(value):
    # Hermes renders successful content as {result: JSON}, and native isError
    # content as {error: JSON}. Only a complete service envelope is evidence.
    # A transport/trust-gate error string must never count as a policy denial.
    for _ in range(8):
        if isinstance(value, str):
            value = json.loads(value)
        elif isinstance(value, dict) and value.get("schema_version") == 1:
            if "data" not in value or "error" not in value:
                break
            return value
        elif isinstance(value, dict):
            if "structuredContent" in value:
                value = value["structuredContent"]
            elif "result" in value:
                value = value["result"]
            elif isinstance(value.get("error"), str):
                value = value["error"]
            else:
                break
        else:
            break
    raise ValueError("Native Hermes result lost its complete version-one envelope")


def checked_result(value, expected_error=None):
    envelope = decode_envelope(value)
    if expected_error:
        error = envelope["error"]
        if envelope["data"] is not None or not isinstance(error, dict) or error.get("code") != expected_error:
            raise ValueError("Native Hermes result did not return the expected categorized service error")
        return {"error_code": error["code"], "retryable": error.get("retryable")}
    if envelope["error"] is not None:
        raise ValueError("Native Hermes tool returned an operation error")
    return envelope["data"]
