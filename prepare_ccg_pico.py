"""Convert the reviewed CCG workbook into traceable character-span training data.

export creates exact/whitespace-normalized alignment candidates and an issue log.
build consumes those records after every unresolved offset has been supplied.
The reviewed source fields remain authoritative; offsets are never guessed.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

import openpyxl

from pipeline_utils import PICO_BITS, PICO_CLASSES, experiment_manifest, write_json, write_jsonl, read_jsonl

FIELDS = {
    "PARTICIPANTS": ("Disease", "Subtype", "Age", "Gender", "Other_Demographic", "Comorbidity"),
    "INTERVENTIONS": ("Intervention_Name", "Intervention_Detail"),
    "OUTCOMES": ("Outcome_Indicator", "Outcome_Target"),
}


def meaningful(value):
    return value is not None and str(value).strip().lower() not in ("", "null", "none", "n/a")


def normalized_offsets(text, phrase):
    """Ignore whitespace only, with an exact map back to original character offsets."""
    offsets = [i for i, char in enumerate(text) if not char.isspace()]
    compact = "".join(text[i] for i in offsets)
    query = "".join(str(phrase).split())
    if not query:
        return []
    return [(offsets[m.start()], offsets[m.end() - 1] + 1)
            for m in re.finditer(re.escape(query), compact)]


def export_workbook(path, output, sheet="Sheet1", overrides_path=None):
    overrides = json.loads(Path(overrides_path).read_text(encoding="utf-8")) if overrides_path else {}
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = workbook[sheet].iter_rows(values_only=True)
    headers = list(next(rows))
    required = {"Index", "Final"} | {f for fields in FIELDS.values() for f in fields}
    if not required.issubset(headers):
        raise ValueError(f"Workbook is missing fields: {sorted(required - set(headers))}")
    records, seen, issues = [], set(), []
    for excel_row, values in enumerate(rows, 2):
        source = dict(zip(headers, values))
        if not meaningful(source.get("Index")):
            continue
        record_id, text = str(source["Index"]), str(source.get("Final") or "")
        if record_id in seen or not text:
            raise ValueError(f"Duplicate ID or empty Final at Excel row {excel_row}")
        seen.add(record_id)
        spans, unresolved = [], []
        if record_id in overrides:
            # Explicit overrides describe the complete record, including intended omissions.
            override = overrides[record_id]
            if override.get("text") != text:
                raise ValueError(f"Override text changed for {record_id}")
            spans = override["spans"]
        else:
            for label, fields in FIELDS.items():
                for field in fields:
                    value = source.get(field)
                    if not meaningful(value):
                        continue
                    matches = normalized_offsets(text, value)
                    if len(matches) == 1:
                        start, end = matches[0]
                        spans.append({"type": label, "start": start, "end": end,
                                      "text": text[start:end], "source_field": field})
                    else:
                        unresolved.append({"field": field, "type": label, "value": str(value),
                                           "reason": "unmatched" if not matches else "ambiguous",
                                           "candidates": matches})
        unique = {(s["type"], int(s["start"]), int(s["end"])): s for s in spans}
        spans = sorted(unique.values(), key=lambda s: (s["start"], s["end"], s["type"]))
        for span in spans:
            if (span["type"] not in PICO_BITS
                    or not 0 <= span["start"] < span["end"] <= len(text)
                    or span.get("text", text[span["start"]:span["end"]]) != text[span["start"]:span["end"]]):
                raise ValueError(f"Invalid character offsets for {record_id}")
        record = {"record_id": record_id, "text": text, "spans": spans,
                  "unresolved": unresolved, "offset_status": "ready" if not unresolved else "needs_offsets",
                  "source": {"file": Path(path).name, "sheet": sheet, "excel_row": excel_row},
                  "reviewed_fields": {k: v for k, v in source.items() if meaningful(v)}}
        records.append(record)
        if unresolved:
            issues.append({"record_id": record_id, "text": text, "unresolved": unresolved})
    workbook.close()
    if set(overrides) - seen:
        raise ValueError("Overrides contain unknown record IDs")
    write_jsonl(output, records)
    report = {"records": len(records), "ready_records": len(records) - len(issues),
              "records_needing_offsets": len(issues),
              "field_issues": sum(len(r["unresolved"]) for r in issues),
              "issues": issues,
              "provenance": experiment_manifest(
                  {"operation": "align_reviewed_fields", "sheet": sheet}, [path]
              )}
    write_json(str(output) + ".alignment_report.json", report)
    return report


def template_key(record):
    if record.get("template_id"):
        return str(record["template_id"])
    text = record["text"]
    spans = sorted(record["spans"], key=lambda s: (s["start"], s["end"]))
    # Mask the union, so overlapping fields do not produce distinct fake templates.
    masked = [False] * len(text)
    for span in spans:
        masked[span["start"]:span["end"]] = [True] * (span["end"] - span["start"])
    normalized = re.sub(r"<E>(?:\s*<E>)+", "<E>",
                        "".join("<E>" if hit else c for c, hit in zip(text, masked)))
    normalized = re.sub(r"\d+(?:\.\d+)?", "<N>", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _subset(groups, size):
    reachable = {0: ()}
    for key, records in groups:
        for total, chosen in list(reachable.items())[::-1]:
            target = total + len(records)
            if target <= size and target not in reachable:
                reachable[target] = chosen + (key,)
        if size in reachable:
            return set(reachable[size])
    return None


def grouped_partition(records, train_size, validation_size, test_size, seed=0):
    """Exact sizes, intact template groups; choose best label-balanced seeded split."""
    if len(records) != train_size + validation_size + test_size or min(
        train_size, validation_size, test_size
    ) < 1:
        raise ValueError("Positive split sizes must sum to the number of ready records")
    grouped = defaultdict(list)
    for record in records:
        grouped[template_key(record)].append(record)
    items = [(key, sorted(rows, key=lambda r: r["record_id"]))
             for key, rows in sorted(grouped.items())]
    labels = {r["record_id"]: {s["type"] for s in r["spans"]} for r in records}
    population = Counter(label for r in records for label in labels[r["record_id"]])
    best = None
    rng = random.Random(seed)
    for _ in range(64):
        ordered = list(items)
        rng.shuffle(ordered)
        val_keys = _subset(ordered, validation_size)
        if val_keys is None:
            continue
        remaining = [(key, rs) for key, rs in ordered if key not in val_keys]
        test_keys = _subset(remaining, test_size)
        if test_keys is None:
            continue
        partition = {
            "validation": [r for key, rs in items if key in val_keys for r in rs],
            "test": [r for key, rs in items if key in test_keys for r in rs],
            "train": [r for key, rs in items if key not in val_keys | test_keys for r in rs],
        }
        deviation = 0.0
        for rows in partition.values():
            count = Counter(label for r in rows for label in labels[r["record_id"]])
            deviation += sum(abs(count[label] / len(rows) - population[label] / len(records))
                             for label in PICO_CLASSES)
        signature = tuple(tuple(sorted(r["record_id"] for r in partition[k]))
                          for k in ("train", "validation", "test"))
        if best is None or (deviation, signature) < best[0]:
            best = ((deviation, signature), partition)
    if best is None:
        raise ValueError("Exact split sizes cannot be formed from template groups in seeded search; "
                         "provide a verified split manifest or adjust split sizes")
    return best[1]


def training_row(record):
    text, spans = record["text"], record["spans"]
    labels = [0] * len(text)
    for span in spans:
        for index in range(span["start"], span["end"]):
            labels[index] |= PICO_BITS[span["type"]]
    return {"pmid": record["record_id"], "sentence_id": record["record_id"],
            "tokens": list(text), "labels": labels, "gold_spans": spans}


def span_training_rows(record):
    row = training_row(record)
    truth = defaultdict(set)
    for span in record["spans"]:
        truth[(span["start"], span["end"])].add(span["type"])
    candidates = set(truth)
    # Cross-boundary hard negatives, capped deterministically per recommendation.
    negatives = sorted({(s, e) for s, _ in truth for _, e in truth if s < e} - candidates)
    candidates.update(negatives[:max(1, len(truth))])
    return [{"pmid": record["record_id"], "tokens": row["tokens"][start:end],
             **{label: label in truth.get((start, end), set()) for label in PICO_CLASSES}}
            for start, end in sorted(candidates)]


def build_dataset(source, output, train_size=541, validation_size=90, test_size=90,
                  seed=0, split_manifest=None):
    records = read_jsonl(source)
    if not records or len({r["record_id"] for r in records}) != len(records):
        raise ValueError("Records must be nonempty and have unique record_id")
    if any(r.get("unresolved") or r.get("offset_status") != "ready" for r in records):
        raise ValueError("Some reviewed fields still need unambiguous character offsets; "
                         "supply overrides, then export again")
    for record in records:
        text = record.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("Each record needs its original nonempty Final text")
        for span in record["spans"]:
            start, end = span.get("start"), span.get("end")
            if (span.get("type") not in PICO_BITS
                    or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(text)
                    or span.get("text", text[start:end]) != text[start:end]):
                raise ValueError(f"Invalid span offsets in {record['record_id']}")
    if split_manifest:
        supplied = json.loads(Path(split_manifest).read_text(encoding="utf-8"))["splits"]
        by_id = {r["record_id"]: r for r in records}
        ids = [rid for split in ("train", "validation", "test") for rid in supplied[split]]
        if len(ids) != len(set(ids)) or set(ids) != set(by_id):
            raise ValueError("Split manifest must partition every record exactly once")
        partition = {split: [by_id[rid] for rid in supplied[split]]
                     for split in ("train", "validation", "test")}
    else:
        partition = grouped_partition(records, train_size, validation_size, test_size, seed)
    ownership = {}
    for split in ("train", "validation", "test"):
        for record in partition[split]:
            key = template_key(record)
            if key in ownership and ownership[key] != split:
                raise ValueError("A normalized template crosses split boundaries")
            ownership[key] = split
    for split, rows in partition.items():
        write_jsonl(Path(output) / f"{split}.json", map(training_row, rows))
        write_jsonl(Path(output) / f"{split}_span_clf.json",
                    (span for row in rows for span in span_training_rows(row)))
    manifest = experiment_manifest(
        {"seed": seed, "split_method": "64 seeded group-preserving candidates; minimum P/I/O prevalence deviation",
         "source_is_reviewed_fields": True, "offsets": "original Final character offsets [start,end)"},
        [source] + ([split_manifest] if split_manifest else []),
        splits={split: sorted(r["record_id"] for r in rows) for split, rows in partition.items()},
        template_groups=len(ownership), split_manifest_supplied=bool(split_manifest),
        historical_paper_split_verified=False,
    )
    write_json(Path(output) / "split_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--workbook", default="datasets/CCG-PICO/ccg_pico_recommendations.xlsx")
    export.add_argument("--sheet", default="Sheet1")
    export.add_argument("--overrides", default=None)
    export.add_argument("--output", default="data/ccg/aligned_records.jsonl")
    build = sub.add_parser("build")
    build.add_argument("--records", default="data/ccg/aligned_records.jsonl")
    build.add_argument("--output", default="data/ccg/json")
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--train_size", type=int, default=541)
    build.add_argument("--validation_size", type=int, default=90)
    build.add_argument("--test_size", type=int, default=90)
    build.add_argument("--split_manifest", default=None)
    args = parser.parse_args()
    if args.command == "export":
        result = export_workbook(args.workbook, args.output, args.sheet, args.overrides)
        print(f"Records={result['records']}; ready={result['ready_records']}; "
              f"need offsets={result['records_needing_offsets']}. Source fields remain unchanged.")
    else:
        result = build_dataset(args.records, args.output, args.train_size, args.validation_size,
                               args.test_size, args.seed, args.split_manifest)
        print({split: len(ids) for split, ids in result["splits"].items()})


if __name__ == "__main__":
    main()
