from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SCHEMA_PATH = (
    PROJECT_ROOT
    / "schemas"
    / "field_schema.json"
)

DEFAULT_ANNOTATIONS_DIR = (
    PROJECT_ROOT
    / "data"
    / "annotations"
)

DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "manifests"
    / "page_manifest.json"
)

EXPECTED_FIELDS = {
    "total_assets",
    "total_equity",
    "net_banking_income",
    "operating_income",
    "net_income",
    "customer_deposits",
    "net_customer_loans",
}


def load_json(path: Path) -> Any:
    """Load one UTF-8 JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as json_file:
            return json.load(json_file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"{path}: invalid JSON at line "
            f"{error.lineno}, column "
            f"{error.colno}: {error.msg}"
        ) from error


def resolve_path(path: Path) -> Path:
    """Resolve a CLI path relative to project root."""

    if path.is_absolute():
        return path

    return (PROJECT_ROOT / path).resolve()


def collect_json_files(
    paths: list[Path],
) -> list[Path]:
    """
    Collect JSON files supplied through the CLI.

    With no arguments, validates all annotation JSON
    files in data/annotations/.
    """

    if not paths:
        files = sorted(
            DEFAULT_ANNOTATIONS_DIR.glob(
                "*.json"
            )
        )

        if not files:
            raise FileNotFoundError(
                "No annotation JSON files found in: "
                f"{DEFAULT_ANNOTATIONS_DIR}"
            )

        return files

    collected: list[Path] = []

    for supplied_path in paths:
        path = resolve_path(supplied_path)

        if path.is_file():
            if path.suffix.casefold() != ".json":
                raise ValueError(
                    f"Not a JSON file: {path}"
                )

            collected.append(path)

        elif path.is_dir():
            collected.extend(
                sorted(path.rglob("*.json"))
            )

        else:
            raise FileNotFoundError(
                f"Path not found: {path}"
            )

    unique_files = sorted(set(collected))

    if not unique_files:
        raise FileNotFoundError(
            "No JSON files found."
        )

    return unique_files


def format_json_location(
    parts: list[Any],
) -> str:
    """Format jsonschema error path."""

    location = "$"

    for part in parts:
        if isinstance(part, int):
            location += f"[{part}]"
        else:
            location += f".{part}"

    return location


def structural_errors(
    data: Any,
    validator: Draft202012Validator,
) -> list[str]:
    """Return JSON Schema validation errors."""

    errors = sorted(
        validator.iter_errors(data),
        key=lambda error: tuple(
            str(part)
            for part in error.absolute_path
        ),
    )

    formatted_errors: list[str] = []

    for error in errors:
        location = format_json_location(
            list(error.absolute_path)
        )

        formatted_errors.append(
            f"{location}: {error.message}"
        )

    return formatted_errors
def check_exact_field_set(
    data: dict[str, Any],
) -> list[str]:
    """Ensure the seven benchmark fields occur once."""

    issues: list[str] = []

    fields = data.get("fields")

    if not isinstance(fields, list):
        return issues

    names = [
        field.get("field")
        for field in fields
        if isinstance(field, dict)
    ]

    counts = Counter(names)

    duplicates = sorted(
        name
        for name, count in counts.items()
        if name is not None and count > 1
    )

    actual_names = {
        name
        for name in names
        if isinstance(name, str)
    }

    missing = sorted(
        EXPECTED_FIELDS - actual_names
    )

    unexpected = sorted(
        actual_names - EXPECTED_FIELDS
    )

    if duplicates:
        issues.append(
            "Duplicate fields: "
            + ", ".join(duplicates)
        )

    if missing:
        issues.append(
            "Missing fields: "
            + ", ".join(missing)
        )

    if unexpected:
        issues.append(
            "Unexpected fields: "
            + ", ".join(unexpected)
        )

    return issues


def annotation_consistency_errors(
    data: dict[str, Any],
    json_path: Path,
) -> list[str]:
    """Checks specific to human annotations."""

    issues: list[str] = []

    doc_id = data.get("doc_id")
    target_year = data.get("target_year")
    target_scope = data.get("target_scope")
    fields = data.get("fields")

    if (
        isinstance(doc_id, str)
        and json_path.stem != doc_id
    ):
        issues.append(
            f"Filename stem {json_path.stem!r} "
            f"does not match doc_id {doc_id!r}."
        )

    if not isinstance(fields, list):
        return issues

    for index, field in enumerate(fields):
        if not isinstance(field, dict):
            continue

        field_name = field.get("field")
        status = field.get("status")

        if field.get("scope") != target_scope:
            issues.append(
                f"fields[{index}] {field_name!r}: "
                f"annotation scope must equal "
                f"target_scope {target_scope!r}."
            )

        if field.get("source_year") != target_year:
            issues.append(
                f"fields[{index}] {field_name!r}: "
                f"source_year must equal target_year "
                f"{target_year!r}."
            )

        if (
            status == "found"
            and field.get("value") is None
        ):
            issues.append(
                f"fields[{index}] {field_name!r}: "
                "found field cannot have null value."
            )

    return issues


def get_manifest_document(
    manifest: dict[str, Any],
    doc_id: str,
) -> dict[str, Any] | None:
    """Get one document from manifest v2."""

    if manifest.get("manifest_version") != 2:
        raise ValueError(
            "page_manifest.json must use "
            "manifest_version 2."
        )

    documents = manifest.get("documents")

    if not isinstance(documents, dict):
        raise ValueError(
            "Manifest is missing the "
            "'documents' object."
        )

    entry = documents.get(doc_id)

    if entry is None:
        return None

    if not isinstance(entry, dict):
        raise TypeError(
            f"Manifest entry for {doc_id!r} "
            "must be an object."
        )

    return entry


def get_scope_page_sets(
    document: dict[str, Any],
    scope: str,
) -> tuple[set[int], set[int]]:
    """
    Return report-page sets for a scope.

    Returns:
        all_statement_report_pages,
        core_statement_report_pages
    """

    scopes = document.get("scopes")

    if not isinstance(scopes, dict):
        return set(), set()

    scope_entry = scopes.get(scope)

    if not isinstance(scope_entry, dict):
        return set(), set()

    statement_pages = scope_entry.get(
        "statement_pages"
    )

    if not isinstance(statement_pages, dict):
        return set(), set()

    all_pages: set[int] = set()

    for page_ref in statement_pages.values():
        if not isinstance(page_ref, dict):
            continue

        report_page = page_ref.get(
            "report_page"
        )

        if isinstance(report_page, int):
            all_pages.add(report_page)

    core_types = scope_entry.get(
        "core_statement_types",
        [],
    )

    core_pages: set[int] = set()

    if isinstance(core_types, list):
        for statement_type in core_types:
            page_ref = statement_pages.get(
                statement_type
            )

            if not isinstance(page_ref, dict):
                continue

            report_page = page_ref.get(
                "report_page"
            )

            if isinstance(report_page, int):
                core_pages.add(report_page)

    return all_pages, core_pages


def manifest_consistency_errors(
    data: dict[str, Any],
    manifest: dict[str, Any],
) -> list[str]:
    """
    Validate annotation report-page citations against
    the manually verified page manifest.

    Annotation "page" always means report_page,
    never the physical PDF/JSONL page.
    """

    issues: list[str] = []

    doc_id = data.get("doc_id")

    if not isinstance(doc_id, str):
        return issues

    document = get_manifest_document(
        manifest,
        doc_id,
    )

    if document is None:
        return [
            f"No page-manifest entry found for "
            f"doc_id {doc_id!r}."
        ]

    verification = document.get(
        "verification"
    )

    if not isinstance(verification, dict):
        issues.append(
            f"{doc_id}: missing verification "
            "metadata."
        )

    elif (
        verification.get("individual")
        != "manually_verified"
    ):
        issues.append(
            f"{doc_id}: individual pages are "
            "not manually verified."
        )

    (
        individual_pages,
        individual_core_pages,
    ) = get_scope_page_sets(
        document,
        "individual",
    )

    (
        consolidated_pages,
        _,
    ) = get_scope_page_sets(
        document,
        "consolidated",
    )

    fields = data.get("fields")

    if not isinstance(fields, list):
        return issues

    for index, field in enumerate(fields):
        if not isinstance(field, dict):
            continue

        field_name = field.get("field")
        status = field.get("status")
        page = field.get("page")

        if (
            status == "found"
            and isinstance(page, int)
            and page not in individual_core_pages
        ):
            issues.append(
                f"fields[{index}] "
                f"{field_name!r}: authoritative "
                f"report page {page} is not one of "
                f"the verified individual core "
                f"report pages "
                f"{sorted(individual_core_pages)}."
            )

        distractors = field.get(
            "distractors",
            [],
        )

        if not isinstance(distractors, list):
            continue

        for distractor_index, distractor in enumerate(
            distractors
        ):
            if not isinstance(
                distractor,
                dict,
            ):
                continue

            distractor_scope = distractor.get(
                "scope"
            )

            distractor_reason = distractor.get(
                "reason"
            )

            distractor_page = distractor.get(
                "page"
            )

            if (
                distractor_reason == "prior_year"
                and isinstance(
                    distractor_page,
                    int,
                )
                and distractor_page
                not in individual_pages
            ):
                issues.append(
                    f"fields[{index}].distractors"
                    f"[{distractor_index}]: "
                    f"prior-year report page "
                    f"{distractor_page} is not in "
                    f"the verified individual "
                    f"statement pages "
                    f"{sorted(individual_pages)}."
                )

            if (
                distractor_scope == "consolidated"
            ):
                if not consolidated_pages:
                    issues.append(
                        f"fields[{index}].distractors"
                        f"[{distractor_index}]: "
                        "consolidated distractor "
                        "exists but the manifest has "
                        "no verified consolidated "
                        "statement block."
                    )

                elif (
                    isinstance(
                        distractor_page,
                        int,
                    )
                    and distractor_page
                    not in consolidated_pages
                ):
                    issues.append(
                        f"fields[{index}].distractors"
                        f"[{distractor_index}]: "
                        f"consolidated report page "
                        f"{distractor_page} is not "
                        f"in the verified "
                        f"consolidated statement "
                        f"pages "
                        f"{sorted(consolidated_pages)}."
                    )

    return issues


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Track A annotations or model "
            "predictions against field_schema.json."
        )
    )

    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=(
            "JSON files or directories. With no "
            "paths, validates data/annotations/."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=(
            "annotation",
            "prediction",
        ),
        default="annotation",
        help=(
            "Annotation mode also applies manifest "
            "and ground-truth consistency checks."
        ),
    )

    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="Path to field_schema.json.",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to page_manifest.json.",
    )

    parser.add_argument(
        "--no-manifest-check",
        action="store_true",
        help=(
            "Skip page-manifest consistency checks."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    schema_path = resolve_path(
        args.schema
    )

    schema = load_json(
        schema_path
    )

    Draft202012Validator.check_schema(
        schema
    )

    validator = Draft202012Validator(
        schema
    )

    json_files = collect_json_files(
        args.paths
    )

    manifest: dict[str, Any] | None = None

    if (
        args.mode == "annotation"
        and not args.no_manifest_check
    ):
        manifest_path = resolve_path(
            args.manifest
        )

        loaded_manifest = load_json(
            manifest_path
        )

        if not isinstance(
            loaded_manifest,
            dict,
        ):
            raise TypeError(
                "Manifest must be a JSON object."
            )

        manifest = loaded_manifest

    passed = 0
    failed = 0

    for json_path in json_files:
        issues: list[str] = []

        try:
            data = load_json(
                json_path
            )

            issues.extend(
                structural_errors(
                    data,
                    validator,
                )
            )

            if isinstance(data, dict):
                issues.extend(
                    check_exact_field_set(
                        data
                    )
                )

                if args.mode == "annotation":
                    issues.extend(
                        annotation_consistency_errors(
                            data,
                            json_path,
                        )
                    )

                    if manifest is not None:
                        issues.extend(
                            manifest_consistency_errors(
                                data,
                                manifest,
                            )
                        )

        except Exception as error:
            issues.append(
                str(error)
            )

        try:
            relative_path = (
                json_path.relative_to(
                    PROJECT_ROOT
                )
            )
        except ValueError:
            relative_path = json_path

        if issues:
            failed += 1

            print(
                f"[FAIL] {relative_path}"
            )

            for issue in issues:
                print(
                    f"  - {issue}"
                )

        else:
            passed += 1

            print(
                f"[PASS] {relative_path}"
            )

    print("\n" + "=" * 60)
    print("Validation summary")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()