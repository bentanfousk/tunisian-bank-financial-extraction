from pathlib import Path

from scripts.benchmark_versions import (
    get_benchmark_version,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    return path.read_text(
        encoding="utf-8"
    )


def build_messages(
    doc_id: str,
    bank: str,
    target_year: int,
    context: str,
    prompt_version: str = "v1",
) -> list[dict[str, str]]:

    version = get_benchmark_version(
        prompt_version
    )

    system_prompt = load_text(
        version.system_prompt_path
    )

    user_template = load_text(
        version.user_template_path
    )

    required_placeholders = [
        "{{DOC_ID}}",
        "{{BANK}}",
        "{{TARGET_YEAR}}",
        "{{CONTEXT}}",
    ]

    missing = [
        placeholder
        for placeholder in required_placeholders
        if placeholder not in user_template
    ]

    if missing:
        raise ValueError(
            "Missing placeholder(s) in frozen "
            f"user template: {missing}"
        )

    user_prompt = user_template.replace(
        "{{DOC_ID}}",
        doc_id,
    )
    user_prompt = user_prompt.replace(
        "{{BANK}}",
        bank,
    )
    user_prompt = user_prompt.replace(
        "{{TARGET_YEAR}}",
        str(target_year),
    )
    user_prompt = user_prompt.replace(
        "{{CONTEXT}}",
        context,
    )

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


if __name__ == "__main__":
    for version_name in [
        "v1",
        "v1_1",
    ]:
        messages = build_messages(
            doc_id="UIB_2024",
            bank="UIB",
            target_year=2024,
            context=(
                "[REPORT_PAGE 140]\n"
                "FAKE STATEMENT TEXT"
            ),
            prompt_version=version_name,
        )

        print(
            version_name,
            "system chars:",
            len(messages[0]["content"]),
        )