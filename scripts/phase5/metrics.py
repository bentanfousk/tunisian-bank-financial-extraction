RELATIVE_TOLERANCE = 0.001


def values_match(gold_value, predicted_value, tolerance=RELATIVE_TOLERANCE):
    if gold_value == 0:
        return predicted_value == 0

    relative_error = abs(predicted_value - gold_value) / abs(gold_value)
    return relative_error <= tolerance


def compare_found_attributes(gold_field, predicted_field):
    return {
        "value_match": values_match(
            gold_field["value"],
            predicted_field["value"],
        ),
        "unit_match": (
            predicted_field["unit_multiplier"]
            == gold_field["unit_multiplier"]
        ),
        "scope_match": (
            predicted_field["scope"]
            == gold_field["scope"]
        ),
        "source_year_match": (
            predicted_field["source_year"]
            == gold_field["source_year"]
        ),
    }

def null_contract_valid(field):
    return (
        field["value"] is None
        and field["unit_multiplier"] is None
        and field["page"] is None
    )


def compare_field_components(gold_field, predicted_field):

    result = {
        "status_match": predicted_field["status"] == gold_field["status"],
        "scope_match": predicted_field["scope"] == gold_field["scope"],
        "source_year_match": (
            predicted_field["source_year"] == gold_field["source_year"]
        ),
        "value_match": None,
        "unit_match": None,
        "null_contract_valid": None,
    }

    if gold_field["status"] == "found" and predicted_field["status"] == "found":
        found_checks = compare_found_attributes(gold_field, predicted_field)

        result["value_match"] = found_checks["value_match"]
        result["unit_match"] = found_checks["unit_match"]

    if predicted_field["status"] in ("not_found", "ambiguous"):
        result["null_contract_valid"] = null_contract_valid(predicted_field)

    return result


def compare_field(gold_field, predicted_field):
    checks = compare_field_components(gold_field, predicted_field)

    failure_reasons = []

    if not checks["status_match"]:
        failure_reasons.append("status_mismatch")

    if not checks["scope_match"]:
        failure_reasons.append("scope_mismatch")

    if not checks["source_year_match"]:
        failure_reasons.append("source_year_mismatch")

    if checks["value_match"] is False:
        failure_reasons.append("value_outside_tolerance")

    if checks["unit_match"] is False:
        failure_reasons.append("unit_mismatch")

    if checks["null_contract_valid"] is False:
        failure_reasons.append("null_contract_violation")

    return {
        "field": gold_field["field"],
        "correct": len(failure_reasons) == 0,
        "failure_reasons": failure_reasons,
        **checks,
    }
EXPECTED_FIELDS = (
    "total_assets",
    "total_equity",
    "net_banking_income",
    "operating_income",
    "net_income",
    "customer_deposits",
    "net_customer_loans",
)

def score_invalid_report(gold_document):
    verdicts = [
        {
            "field": field_name,
            "correct": False,
            "failure_reasons": ["invalid_prediction"],
            "status_match": None,
            "scope_match": None,
            "source_year_match": None,
            "value_match": None,
            "unit_match": None,
            "null_contract_valid": None,
            "citation_value_present": None,
        }
        for field_name in EXPECTED_FIELDS
    ]

    return {
        "doc_id": gold_document["doc_id"],
        "fields": verdicts,
        "summary": {
            "fully_correct_fields": 0,
            "total_fields": len(EXPECTED_FIELDS),
            "fully_correct_rate": 0.0,
            "valid_json": False,
            "missing_field_accuracy": None,
            "citation_accuracy": None,
        },
    }


def score_report(gold_document, predicted_document):
    for key in ("doc_id", "target_year", "target_scope", "currency"):
        if gold_document[key] != predicted_document[key]:
            raise ValueError(f"Document mismatch for '{key}'")

    gold_fields = {
        field["field"]: field
        for field in gold_document["fields"]
    }

    predicted_fields = {
        field["field"]: field
        for field in predicted_document["fields"]
    }

    if set(gold_fields) != set(EXPECTED_FIELDS):
        raise ValueError("Annotation does not contain exactly the 7 expected fields")

    if set(predicted_fields) != set(EXPECTED_FIELDS):
        raise ValueError("Prediction does not contain exactly the 7 expected fields")

    verdicts = [
        compare_field(
            gold_fields[field_name],
            predicted_fields[field_name],
        )
        for field_name in EXPECTED_FIELDS
    ]

    fully_correct_fields = sum(
        verdict["correct"]
        for verdict in verdicts
    )

    total_fields = len(EXPECTED_FIELDS)

    return {
        "doc_id": gold_document["doc_id"],
        "fields": verdicts,
        "summary": {
            "fully_correct_fields": fully_correct_fields,
            "total_fields": total_fields,
            "fully_correct_rate": fully_correct_fields / total_fields,
        },
    }

def missing_field_correct(gold_field, predicted_field):
    if gold_field["status"] != "not_found":
        return None

    return predicted_field["status"] == "not_found"

def normalize_number_text(text):
    return (
        str(text)
        .replace(" ", "")
        .replace("\u00a0", "")
        .replace("\u202f", "")
    )


def value_appears_in_text(value, page_text):
    if value is None:
        return False

    normalized_value = normalize_number_text(value)
    normalized_text = normalize_number_text(page_text)

    return normalized_value in normalized_text

def citation_value_present(predicted_field, page_text):
    if predicted_field["status"] != "found":
        return None

    if predicted_field["page"] is None or page_text is None:
        return False

    return value_appears_in_text(
        predicted_field["value"],
        page_text,
    )