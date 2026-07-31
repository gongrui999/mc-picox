"""
Deep analysis of missed short spans (1-3 tokens):
- Frequency distribution of missed span text
- Whether the same text appears as gold span elsewhere (train set)
- POS / lexical pattern analysis
- Categorization into patterns (abbreviation, single word, number+unit, etc.)
"""

import json
import re
import os
from collections import defaultdict, Counter
from enum import Enum


class PicoType(Enum):
    PARTICIPANTS = 4
    INTERVENTIONS = 2
    OUTCOMES = 1


def extract_spans_from_labels(label_sequence, pico_type_value):
    labels = [pico_type_value & l if l > 0 else 0 for l in label_sequence]
    spans = []
    i = 0
    while i < len(labels):
        if labels[i] > 0:
            start = i
            while i < len(labels) and labels[i] > 0:
                i += 1
            spans.append((start, i - start))
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


def categorize_span(text):
    tokens = text.split()
    if len(tokens) == 1:
        t = tokens[0]
        if re.match(r'^[A-Z]{2,}[0-9s]*$', t):
            return "ABBREVIATION"
        if re.match(r'^[A-Z][a-z]*[A-Z]', t):
            return "CAMELCASE_TERM"
        if re.match(r'^[\d.,]+$', t):
            return "NUMBER"
        if t in ('(', ')', ',', '.', ';', ':', '-', '/'):
            return "PUNCTUATION"
        if t.lower() in ('the', 'a', 'an', 'of', 'in', 'for', 'and', 'or',
                          'to', 'with', 'by', 'on', 'at', 'from', 'as', 'was',
                          'were', 'is', 'are', 'be', 'been', 'being', 'had',
                          'has', 'have', 'that', 'this', 'these', 'those',
                          'not', 'no', 'but', 'if', 'than', 'vs', 'vs.'):
            return "FUNCTION_WORD"
        return "SINGLE_CONTENT_WORD"

    all_upper_abbrev = all(re.match(r'^[A-Z]{2,}[0-9s]*$', t) or t in ('(', ')', '-', '/', ',') for t in tokens)
    if all_upper_abbrev:
        return "MULTI_ABBREVIATION"

    has_number = any(re.match(r'^[\d.,]+$', t) for t in tokens)
    has_unit = any(t.lower() in ('mg', 'ml', 'kg', 'mm', 'cm', 'g', 'l', '%',
                                  'mg/kg', 'mg/dl', 'mmol', 'mmol/l', 'iu',
                                  'mcg', 'mmhg', 'μg', 'ng', 'days', 'weeks',
                                  'months', 'years', 'hours', 'minutes') for t in tokens)
    if has_number and has_unit:
        return "NUMBER_WITH_UNIT"
    if has_number:
        return "NUMBER_EXPRESSION"

    has_paren = any(t in ('(', ')') for t in tokens)
    if has_paren:
        return "PARENTHETICAL"

    return "SHORT_PHRASE"


def main():
    missed_path = r"D:\PICOX\PICOX\missed_spans_full.json"
    train_path = r"D:\PICOX\PICOX\data\bioc\json\train.json"
    pred_path = r"D:\PICOX\PICOX\data\bioc\json\step_2_span_clf\test_pico_spans.json"

    with open(missed_path, "r", encoding="utf-8") as f:
        all_missed = json.load(f)

    short_missed = [m for m in all_missed if m["span_length"] <= 3]
    print(f"Total missed spans: {len(all_missed)}")
    print(f"Short missed spans (1-3 tokens): {len(short_missed)}")
    print()

    # --- Build train gold span vocabulary ---
    print("Building training set gold span vocabulary...")
    train_data = load_jsonl(train_path)
    train_span_texts = defaultdict(set)
    for row in train_data:
        tokens = row["tokens"]
        labels = row["labels"]
        for pt in PicoType:
            for start, length in extract_spans_from_labels(labels, pt.value):
                if length <= 3:
                    text = " ".join(tokens[start:start + length]).lower()
                    train_span_texts[text].add(pt.name)

    print(f"Unique short gold span texts in train: {len(train_span_texts)}")
    print()

    # --- Analyze by length ---
    print("=" * 70)
    print("DISTRIBUTION BY TOKEN LENGTH")
    print("=" * 70)
    len_counter = Counter(m["span_length"] for m in short_missed)
    for l in sorted(len_counter):
        print(f"  {l} token(s): {len_counter[l]} missed spans")
    print()

    # --- Analyze by PICO type ---
    print("=" * 70)
    print("DISTRIBUTION BY PICO TYPE")
    print("=" * 70)
    type_counter = Counter(m["type"] for m in short_missed)
    for t, c in type_counter.most_common():
        print(f"  {t}: {c}")
    print()

    # --- Categorize by lexical pattern ---
    print("=" * 70)
    print("DISTRIBUTION BY LEXICAL PATTERN")
    print("=" * 70)
    category_counter = Counter()
    category_examples = defaultdict(list)
    for m in short_missed:
        cat = categorize_span(m["span_text"])
        category_counter[cat] += 1
        if len(category_examples[cat]) < 8:
            category_examples[cat].append(m["span_text"])

    for cat, count in category_counter.most_common():
        pct = count / len(short_missed) * 100
        examples = " | ".join(category_examples[cat])
        print(f"\n  {cat}: {count} ({pct:.1f}%)")
        print(f"    examples: {examples}")
    print()

    # --- Check if missed spans appear in training set ---
    print("=" * 70)
    print("TRAINING SET COVERAGE")
    print("=" * 70)
    seen_in_train = 0
    not_seen = 0
    not_seen_examples = []
    for m in short_missed:
        text_lower = m["span_text"].lower()
        if text_lower in train_span_texts:
            seen_in_train += 1
        else:
            not_seen += 1
            if len(not_seen_examples) < 30:
                not_seen_examples.append(f"{m['span_text']} [{m['type']}]")

    print(f"  Seen in train gold spans:     {seen_in_train}/{len(short_missed)} ({seen_in_train/len(short_missed)*100:.1f}%)")
    print(f"  NOT seen in train gold spans: {not_seen}/{len(short_missed)} ({not_seen/len(short_missed)*100:.1f}%)")
    print()
    print("  Examples of NOT seen in train:")
    for ex in not_seen_examples:
        print(f"    - {ex}")
    print()

    # --- Most frequently missed span texts ---
    print("=" * 70)
    print("MOST FREQUENTLY MISSED SPAN TEXTS (top 40)")
    print("=" * 70)
    text_type_counter = Counter()
    for m in short_missed:
        key = f"{m['span_text'].lower()} [{m['type']}]"
        text_type_counter[key] += 1

    print(f"  {'Span Text':<55} {'Count':>5}")
    print(f"  {'-'*55} {'-----':>5}")
    for text, count in text_type_counter.most_common(40):
        in_train = "Y" if text.split(" [")[0] in train_span_texts else "N"
        print(f"  {text:<55} {count:>5}  train={in_train}")
    print()

    # --- Boundary detector analysis: did it detect ANY boundary near the missed span? ---
    print("=" * 70)
    print("BOUNDARY DETECTOR BEHAVIOR ON MISSED SHORT SPANS")
    print("=" * 70)
    pred_data = load_jsonl(pred_path)
    pred_by_pmid = {}
    for row in pred_data:
        key = (row["pmid"], len(row["tokens"]))
        pred_by_pmid[key] = row.get("boundary_pred", [])

    near_boundary = 0
    no_near_boundary = 0
    for m in short_missed:
        key = (m["pmid"], m["sentence_tokens"])
        bp = pred_by_pmid.get(key, [])
        if not bp:
            no_near_boundary += 1
            continue
        start = m["span_start"]
        end = start + m["span_length"]
        window = 3
        region = bp[max(0, start - window): min(len(bp), end + window)]
        if any(b != 0 for b in region):
            near_boundary += 1
        else:
            no_near_boundary += 1

    print(f"  Has boundary within ±3 tokens: {near_boundary} ({near_boundary/len(short_missed)*100:.1f}%)")
    print(f"  NO boundary within ±3 tokens:  {no_near_boundary} ({no_near_boundary/len(short_missed)*100:.1f}%)")
    print()
    print("  --> 'NO boundary nearby' means the boundary detector completely")
    print("      ignored this region. These are the hardest cases.")

    # --- Save detailed results ---
    output = {
        "total_short_missed": len(short_missed),
        "by_length": dict(len_counter),
        "by_type": dict(type_counter),
        "by_category": dict(category_counter),
        "train_coverage": {
            "seen": seen_in_train,
            "not_seen": not_seen,
        },
        "boundary_nearby": {
            "yes": near_boundary,
            "no": no_near_boundary,
        },
        "top_missed_texts": text_type_counter.most_common(100),
    }
    out_path = r"D:\PICOX\PICOX\short_missed_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {out_path}")


if __name__ == "__main__":
    main()
