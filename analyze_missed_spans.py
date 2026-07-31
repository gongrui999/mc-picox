"""
Analyze missed (FN) gold spans: extract all gold spans that were completely
missed by the prediction pipeline, and categorize them by PICO type, length,
and whether the boundary detector produced any boundary at all for the sentence.
"""

import json
import os
import sys
from enum import Enum
from collections import defaultdict


class PicoType(Enum):
    PARTICIPANTS = 4
    INTERVENTIONS = 2
    OUTCOMES = 1


class Span:
    def __init__(self, start, length):
        self.start = start
        self.length = int(length)
        self.end = self.start + self.length

    def __eq__(self, other):
        return (self.start, self.length) == (other.start, other.length)

    def __repr__(self):
        return f"Span(s={self.start}, l={self.length})"


def extract_spans_from_labels(label_sequence, pico_type_value):
    labels = [pico_type_value & l if l > 0 else 0 for l in label_sequence]
    spans = []
    i = 0
    while i < len(labels):
        if labels[i] > 0:
            start = i
            while i < len(labels) and labels[i] > 0:
                i += 1
            spans.append(Span(start=start, length=i - start))
        else:
            i += 1
    return spans


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    pred_path = r"D:\PICOX\PICOX\data\bioc\json\step_2_span_clf\test_pico_spans.json"

    if not os.path.exists(pred_path):
        print(f"File not found: {pred_path}")
        sys.exit(1)

    data = load_jsonl(pred_path)
    print(f"Loaded {len(data)} sentences\n")

    all_missed = []
    total_gold = 0
    total_missed = 0
    stats_by_type = defaultdict(lambda: {"gold": 0, "missed": 0})
    stats_by_length = defaultdict(lambda: {"gold": 0, "missed": 0})
    no_boundary_sentences = 0
    missed_in_no_boundary = 0

    for row in data:
        tokens = row["tokens"]
        original_labels = row["original_labels"]
        pico_elements = row.get("pico_elements", {})
        boundary_pred = row.get("boundary_pred", [])
        pmid = row.get("pmid", "?")

        has_any_boundary = any(b != 0 for b in boundary_pred)
        if not has_any_boundary:
            no_boundary_sentences += 1

        for pico_type in PicoType:
            gold_spans = extract_spans_from_labels(original_labels, pico_type.value)
            if not gold_spans:
                continue

            records = pico_elements.get(pico_type.name, [])
            pred_spans = [Span(start=r["span_start"], length=r["span_length"]) for r in records]

            for gs in gold_spans:
                total_gold += 1
                stats_by_type[pico_type.name]["gold"] += 1

                length_bucket = "1-3" if gs.length <= 3 else "4-10" if gs.length <= 10 else "11+"
                stats_by_length[length_bucket]["gold"] += 1

                matched = any(ps == gs for ps in pred_spans)
                if not matched:
                    partial = any(
                        max(ps.start, gs.start) < min(ps.end, gs.end)
                        for ps in pred_spans
                    )
                    if not partial:
                        total_missed += 1
                        stats_by_type[pico_type.name]["missed"] += 1
                        stats_by_length[length_bucket]["missed"] += 1
                        if not has_any_boundary:
                            missed_in_no_boundary += 1

                        span_text = " ".join(tokens[gs.start:gs.end])
                        context_start = max(0, gs.start - 5)
                        context_end = min(len(tokens), gs.end + 5)
                        context_before = " ".join(tokens[context_start:gs.start])
                        context_after = " ".join(tokens[gs.end:context_end])

                        all_missed.append({
                            "pmid": pmid,
                            "type": pico_type.name,
                            "span_text": span_text,
                            "span_length": gs.length,
                            "span_start": gs.start,
                            "context": f"...{context_before} [[[{span_text}]]] {context_after}...",
                            "sentence_has_boundary": has_any_boundary,
                            "sentence_tokens": len(tokens),
                        })

    print("=" * 70)
    print(f"MISSED GOLD SPAN ANALYSIS (completely missed, zero overlap)")
    print("=" * 70)
    print(f"Total gold spans:   {total_gold}")
    print(f"Completely missed:  {total_missed}  ({total_missed/total_gold*100:.1f}%)")
    print(f"Sentences with 0 boundary predictions: {no_boundary_sentences}/{len(data)} ({no_boundary_sentences/len(data)*100:.1f}%)")
    print(f"Missed spans in 0-boundary sentences:  {missed_in_no_boundary}/{total_missed} ({missed_in_no_boundary/total_missed*100:.1f}% of all missed)")
    print()

    print("-" * 70)
    print("By PICO type:")
    print(f"  {'Type':<18} {'Gold':>6} {'Missed':>8} {'Miss%':>8}")
    for t in ["PARTICIPANTS", "INTERVENTIONS", "OUTCOMES"]:
        g = stats_by_type[t]["gold"]
        m = stats_by_type[t]["missed"]
        print(f"  {t:<18} {g:>6} {m:>8} {m/g*100:>7.1f}%")
    print()

    print("-" * 70)
    print("By span length:")
    print(f"  {'Length':<10} {'Gold':>6} {'Missed':>8} {'Miss%':>8}")
    for bucket in ["1-3", "4-10", "11+"]:
        g = stats_by_length[bucket]["gold"]
        m = stats_by_length[bucket]["missed"]
        pct = m / g * 100 if g > 0 else 0
        print(f"  {bucket:<10} {g:>6} {m:>8} {pct:>7.1f}%")
    print()

    print("=" * 70)
    print(f"SAMPLE MISSED SPANS (showing first 50)")
    print("=" * 70)

    all_missed.sort(key=lambda x: x["span_length"], reverse=True)

    for i, item in enumerate(all_missed[:50]):
        bnd = "YES" if item["sentence_has_boundary"] else "NO <--"
        print(f"\n[{i+1}] PMID={item['pmid']}  Type={item['type']}  Length={item['span_length']}  HasBoundary={bnd}")
        print(f"    {item['context']}")

    output_path = r"D:\PICOX\PICOX\missed_spans_full.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_missed, f, ensure_ascii=False, indent=2)
    print(f"\n\nFull list saved to: {output_path} ({len(all_missed)} items)")


if __name__ == "__main__":
    main()
