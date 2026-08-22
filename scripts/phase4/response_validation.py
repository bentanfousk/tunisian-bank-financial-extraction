import json
from typing import Any

from jsonschema import ValidationError
from jsonschema.validators import validator_for


EXPECTED_FIELDS = [
    "total_assets",
    "total_equity",
    "net_banking_income",
    "operating_income",
    "net_income",
    "customer_deposits",
    "net_customer_loans",
]


def parse_and_validate(
    raw_text: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """
    Parse a raw model response as JSON and validate it
    against the provided schema.

    No repair or cleanup is performed.

    For Track-A extraction documents containing a
    "fields" list, also enforce the frozen contract that
    the seven required fields appear exactly once and in
    the fixed order.
    """

    try:
        parsed = json.loads(raw_text)

    except json.JSONDecodeError as exc:
        return {
            "valid": False,
            "parsed": None,
            "error_type": "json_parse_error",
            "error_message": str(exc),
        }

    validator_class = validator_for(schema)
    validator_class.check_schema(schema)

    validator = validator_class(schema)

    try:
        validator.validate(parsed)

    except ValidationError as exc:
        return {
            "valid": False,
            "parsed": parsed,
            "error_type": "schema_validation_error",
            "error_message": exc.message,
        }

    # Additional Track-A document contract.
    #
    # The JSON Schema guarantees that field names come
    # from the allowed enum, but it does not guarantee
    # that all seven required names appear exactly once
    # and in the frozen order.
    #
    # Only apply this extra validation when the parsed
    # document actually contains a "fields" collection,
    # so parse_and_validate() remains reusable for the
    # smaller schemas used in unit tests.
    if isinstance(parsed, dict) and "fields" in parsed:
        fields = parsed["fields"]

        if not isinstance(fields, list):
            return {
                "valid": False,
                "parsed": parsed,
                "error_type": "contract_validation_error",
                "error_message": (
                    "'fields' must be a list."
                ),
            }

        actual_fields = []

        for field in fields:
            if not isinstance(field, dict):
                return {
                    "valid": False,
                    "parsed": parsed,
                    "error_type": (
                        "contract_validation_error"
                    ),
                    "error_message": (
                        "Every field entry must be "
                        "a JSON object."
                    ),
                }

            actual_fields.append(
                field.get("field")
            )

        if actual_fields != EXPECTED_FIELDS:
            return {
                "valid": False,
                "parsed": parsed,
                "error_type": (
                    "contract_validation_error"
                ),
                "error_message": (
                    "Prediction must contain exactly "
                    "the 7 required fields once each "
                    "and in the fixed order. "
                    f"Expected {EXPECTED_FIELDS}, "
                    f"got {actual_fields}."
                ),
            }

    return {
        "valid": True,
        "parsed": parsed,
        "error_type": None,
        "error_message": None,
    }