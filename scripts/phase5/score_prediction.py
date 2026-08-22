import argparse

import json
from pathlib import Path

from scripts.phase5.metrics import (
    score_report,
    score_invalid_report,
    citation_value_present,
    missing_field_correct,
)

def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--annotation", required=True)
    parser.add_argument("--prediction")
    parser.add_argument("--validation", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report-text", required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    gold = load_json(Path(args.annotation))
    validation = load_json(Path(args.validation))

    if validation["valid"]:
        if args.prediction is None:
            raise ValueError(
                "A valid Phase-4 run must provide --prediction"
            )

        prediction = load_json(Path(args.prediction))
    else:
        prediction = None


    if not validation["valid"]:
        result = score_invalid_report(gold)

        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    manifest = load_json(Path(args.manifest))
    report_pages = load_jsonl(Path(args.report_text))

    result = score_report(gold, prediction)
    result["summary"]["valid_json"] = validation["valid"]



    statement_pages = manifest["documents"][gold["doc_id"]]["scopes"]["individual"]["statement_pages"]

    report_to_pdf_page = {
        page_info["report_page"]: page_info["pdf_page"]
        for page_info in statement_pages.values()
    }

    page_text_by_pdf_page = {
        record["page"]: record["text"]
        for record in report_pages
    }

    predicted_fields = {
        field["field"]: field
        for field in prediction["fields"]
    }




    gold_fields = {
        field["field"]: field
        for field in gold["fields"]
    }

    missing_results = []

    for field_name, gold_field in gold_fields.items():
        check = missing_field_correct(
            gold_field,
            predicted_fields[field_name],
        )

        if check is not None:
            missing_results.append(check)

    result["summary"]["missing_field_accuracy"] = (
        sum(missing_results) / len(missing_results)
        if missing_results
        else None
    )


    citation_results = []

    for verdict in result["fields"]:
        predicted_field = predicted_fields[verdict["field"]]

        page_text = None

        if predicted_field["page"] is not None:
            pdf_page = report_to_pdf_page.get(predicted_field["page"])

            if pdf_page is not None:
                page_text = page_text_by_pdf_page.get(pdf_page)

        citation_check = citation_value_present(
            predicted_field,
            page_text,
        )

        verdict["citation_value_present"] = citation_check

        if citation_check is not None:
            citation_results.append(citation_check)

    result["summary"]["citation_accuracy"] = (
        sum(citation_results) / len(citation_results)
        if citation_results
        else None
    )
    
    

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()