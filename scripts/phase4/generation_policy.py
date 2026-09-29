from __future__ import annotations

import hashlib
from typing import Any

from scripts.phase4.ollama_adapter import generate
from scripts.phase4.response_validation import (
    parse_and_validate,
)


REPAIR_SYSTEM_PROMPT = """
You repair structured JSON output only.

The previous response came from a financial extraction
task and failed schema validation.

Rules:
- Do not re-extract information.
- Do not calculate or invent financial values.
- Preserve the information already present whenever possible.
- Repair only what is necessary to satisfy the output contract.
- Return exactly these 7 fields, exactly once each,
  in this exact order:
  1. total_assets
  2. total_equity
  3. net_banking_income
  4. operating_income
  5. net_income
  6. customer_deposits
  7. net_customer_loans
- If status is "found":
  value must be numeric,
  unit_multiplier must be 1, 1000, or 1000000,
  and page must be an integer.
- If status is "not_found" or "ambiguous":
  value, unit_multiplier, and page must all be null.
- Never use 0 instead of null.
- Return JSON only.
- Never duplicate a field.
- Never replace a missing required field with a duplicate.
- If a required field is completely absent from the
  previous response, do not invent a value:
  return it as "not_found" with value=null,
  unit_multiplier=null, and page=null.
""".strip()


REPAIR_PROMPT_SHA256 = hashlib.sha256(
    REPAIR_SYSTEM_PROMPT.encode("utf-8")
).hexdigest()


def build_repair_messages(
    raw_text: str,
    validation: dict[str, Any],
) -> list[dict[str, str]]:
    error_type = validation.get(
        "error_type"
    )

    error_message = validation.get(
        "error_message"
    )

    user_message = (
        "Validation error type:\n"
        f"{error_type}\n\n"
        "Validation error message:\n"
        f"{error_message}\n\n"
        "Previous model response:\n"
        f"{raw_text}\n\n"
        "Return only the corrected JSON."
    )

    return [
        {
            "role": "system",
            "content": REPAIR_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]


def combine_generation_results(
    first_result: dict[str, Any],
    retry_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Final response = retry response.

    Token/duration metrics represent the total cost of
    first pass + repair pass.
    """

    return {
        "text": retry_result["text"],
        "model_id": retry_result["model_id"],
        "input_tokens": (
            first_result["input_tokens"]
            + retry_result["input_tokens"]
            if (
                first_result["input_tokens"] is not None
                and retry_result["input_tokens"] is not None
            )
            else None
        ),
        "output_tokens": (
            first_result["output_tokens"]
            + retry_result["output_tokens"]
        ),
        "total_duration_ns": (
            first_result["total_duration_ns"]
            + retry_result["total_duration_ns"]
        ),
        "load_duration_ns": (
            first_result["load_duration_ns"]
            + retry_result["load_duration_ns"]
        ),
        "prompt_eval_duration_ns": (
            first_result["prompt_eval_duration_ns"]
            + retry_result["prompt_eval_duration_ns"]
        ),
        "eval_duration_ns": (
            first_result["eval_duration_ns"]
            + retry_result["eval_duration_ns"]
        ),
        "done_reason": retry_result[
            "done_reason"
        ],
    }


def generate_with_one_repair(
    model_id: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    settings: dict[str, Any],
    on_retry: Any = None,
) -> dict[str, Any]:
    """
    Run the normal extraction once.

    If it fails schema validation, make exactly one
    format-only repair call using:
      - the previous response,
      - the validator error,
      - the same model/schema/settings.

    The financial pages are NOT sent again.
    """

    first_result = generate(
        model_id=model_id,
        messages=messages,
        schema=schema,
        settings=settings,
    )

    first_validation = parse_and_validate(
        raw_text=first_result["text"],
        schema=schema,
    )

    if first_validation["valid"]:
        return {
            "generation_result": first_result,
            "validation": first_validation,
            "retry_used": False,
            "first_result": first_result,
            "first_validation": first_validation,
            "retry_result": None,
            "retry_validation": None,
        }

    if on_retry is not None:
        on_retry(first_validation)

    repair_messages = build_repair_messages(
        raw_text=first_result["text"],
        validation=first_validation,
    )

    retry_result = generate(
        model_id=model_id,
        messages=repair_messages,
        schema=schema,
        settings=settings,
    )

    retry_validation = parse_and_validate(
        raw_text=retry_result["text"],
        schema=schema,
    )

    combined_result = combine_generation_results(
        first_result=first_result,
        retry_result=retry_result,
    )

    return {
        "generation_result": combined_result,
        "validation": retry_validation,
        "retry_used": True,
        "first_result": first_result,
        "first_validation": first_validation,
        "retry_result": retry_result,
        "retry_validation": retry_validation,
    }
