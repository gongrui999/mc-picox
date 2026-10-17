"""Report reviewed CCG workbook field availability without changing source values."""

import argparse
from pathlib import Path

import openpyxl

from pipeline_utils import experiment_manifest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", default="datasets/CCG-PICO/ccg_pico_recommendations.xlsx")
    parser.add_argument("--sheet", default="Sheet1")
    parser.add_argument("--output_file", default="data/ccg/source_statistics.json")
    args = parser.parse_args()
    workbook = openpyxl.load_workbook(args.input_file, read_only=True, data_only=True)
    rows = workbook[args.sheet].iter_rows(values_only=True)
    headers = next(rows)
    records = [dict(zip(headers, values)) for values in rows if values[0] is not None]
    workbook.close()
    if not records:
        raise ValueError("Workbook has no records")
    present = lambda value: value is not None and str(value).strip().lower() not in ("", "null", "none")
    groups = {
        "P": ["Disease", "Age", "Gender", "Comorbidity"],
        "I": ["Intervention_Name", "Intervention_Detail"],
        "O": ["Outcome_Indicator", "Outcome_Target"],
    }
    lengths = [len(str(row.get("Final") or "")) for row in records]
    report = {
        "records": len(records),
        "has_reviewed_fields": {label: sum(any(present(row.get(k)) for k in fields) for row in records)
                                for label, fields in groups.items()},
        "mean_characters": sum(lengths) / len(lengths), "max_characters": max(lengths),
        "provenance": experiment_manifest(args, [Path(args.input_file)]),
    }
    write_json(args.output_file, report)
    print({k: v for k, v in report.items() if k != "provenance"})


if __name__ == "__main__":
    main()
