"""Manual full-pipeline smoke test; writes only under demo_runtime."""

import argparse
import json
import shutil
import uuid
from pathlib import Path

from demo.backend.app.paths import ROOT, RUNTIME
from demo.backend.app.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path, help="Text-based annual report PDF")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()
    source = args.pdf.resolve(strict=True)
    job_id = uuid.uuid4().hex
    job_dir = RUNTIME / job_id
    job_dir.mkdir(parents=True)
    pdf_path = job_dir / "report.pdf"
    shutil.copy2(source, pdf_path)
    result = run_pipeline(
        job_id=job_id, job_dir=job_dir, pdf_path=pdf_path,
        filename=source.name, bank=args.bank, year=args.year,
        progress=lambda stage, message, data: print(stage, message, data or {}, flush=True),
    )
    print(json.dumps({"job_id": job_id, "fields": result["fields"],
                      "validation": result["validation"]}, ensure_ascii=False, indent=2))
    print(f"Runtime result: {job_dir / 'result.json'}")


if __name__ == "__main__":
    main()
