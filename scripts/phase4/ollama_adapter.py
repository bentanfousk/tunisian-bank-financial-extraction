from typing import Any

import ollama


def generate(
    model_id: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate one structured extraction response.

    The public return shape is intentionally provider-neutral.
    Ollama-specific response fields are normalized here.
    """

    response = ollama.chat(
        model=model_id,
        messages=messages,
        format=schema,
        options=settings,
        stream=False,
        think=False,
    )

    return {
        "text": response.message.content,
        "model_id": model_id,
        "input_tokens": response.prompt_eval_count,
        "output_tokens": response.eval_count,
        "total_duration_ns": response.total_duration,
        "load_duration_ns": response.load_duration,
        "prompt_eval_duration_ns": response.prompt_eval_duration,
        "eval_duration_ns": response.eval_duration,
        "done_reason": response.done_reason,
    }