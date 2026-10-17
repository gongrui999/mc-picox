"""Run a fixed three-shot Qwen baseline through an OpenAI-compatible endpoint.

No API request is made on import. Credentials are read from an environment variable.
"""

import argparse
import json
import os
from pathlib import Path
import re
from urllib.request import Request, urlopen

from pipeline_utils import (
    PICO_CLASSES, experiment_manifest, gold_spans, read_jsonl, sentence_id, write_json, write_jsonl,
)

TYPE_MAP = {"P": "PARTICIPANTS", "I": "INTERVENTIONS", "O": "OUTCOMES",
            **{label: label for label in PICO_CLASSES}}


def render_text(tokens, joiner):
    text, offsets = "", []
    for i, token in enumerate(tokens):
        if i:
            text += joiner
        offsets.append((len(text), len(text) + len(token)))
        text += token
    return text, offsets


def exact_response_spans(content, tokens, joiner=" "):
    clean = re.sub(r"^\s*" + r"\x60{3}(?:json)?\s*|\s*\x60{3}\s*$", "", content)
    value = json.loads(clean)
    if not isinstance(value, list):
        raise ValueError("Expected a JSON list of text/type entities")
    text, offsets = render_text(tokens, joiner)
    starts, ends = {a: i for i, (a, _) in enumerate(offsets)}, {b: i + 1 for i, (_, b) in enumerate(offsets)}
    found, unmatched = set(), []
    for entity in value:
        if not isinstance(entity, dict):
            raise ValueError("Each entity must be an object")
        phrase, label = entity.get("text"), TYPE_MAP.get(str(entity.get("type", "")).upper())
        if not isinstance(phrase, str) or not phrase or label is None:
            unmatched.append(entity)
            continue
        matches = [m for m in re.finditer(re.escape(phrase), text)
                   if m.start() in starts and m.end() in ends]
        if not matches:
            unmatched.append(entity)
        for match in matches:
            start, end = starts[match.start()], ends[match.end()]
            found.add((label, start, end - start))
    return found, unmatched


def demonstrations(train, example_ids, evaluation):
    if len(example_ids) != 3 or len(set(example_ids)) != 3:
        raise ValueError("Provide exactly three distinct training sentence IDs")
    indexed = {sentence_id(row, i): row for i, row in enumerate(train)}
    if len(indexed) != len(train):
        raise ValueError("Training sentence IDs must be unique")
    if any(key not in indexed for key in example_ids):
        raise ValueError("All demonstrations must come from training")
    if {row["pmid"] for row in train} & {row["pmid"] for row in evaluation}:
        raise ValueError("Training and evaluation document IDs overlap")
    return [indexed[key] for key in example_ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--evaluation_file", required=True)
    parser.add_argument("--example_ids", nargs=3, required=True)
    parser.add_argument("--base_url", required=True, help="Provider URL ending in /v1")
    parser.add_argument("--model", default="Qwen2.5-72B-Instruct")
    parser.add_argument("--api_key_env", default="QWEN_API_KEY")
    parser.add_argument("--joiner", choices=["space", "none"], default="space")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    train, evaluation = read_jsonl(args.train_file), read_jsonl(args.evaluation_file)
    examples = demonstrations(train, args.example_ids, evaluation)
    key = os.environ.get(args.api_key_env)
    if not key:
        raise ValueError(f"Set the {args.api_key_env} environment variable")
    if Path(args.output_file).exists():
        raise ValueError("Choose a new output file to preserve existing baseline results")
    joiner = " " if args.joiner == "space" else ""
    messages = [{"role": "system", "content":
                 "Extract exact text spans for Participants (P), Interventions (I), Outcomes (O). "
                 "Allow overlapping spans. Return only a JSON list of objects with text and type. "
                 "Copy substrings exactly. Do not add explanations."}]
    for row in examples:
        text, _ = render_text(row["tokens"], joiner)
        answer = [{"text": joiner.join(row["tokens"][s:s + length]), "type": label}
                  for label, s, length in sorted(gold_spans(row))]
        messages.extend([{"role": "user", "content": text},
                         {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}])
    manifest = experiment_manifest(
        {k: v for k, v in vars(args).items() if k != "api_key_env"},
        [args.train_file, args.evaluation_file],
        prompt=messages, decoding={"temperature": 0, "top_p": 1},
        duplicate_text_policy="all exact occurrences with token-aligned endpoints",
        status="running",
    )
    write_json(str(args.output_file) + ".manifest.json", manifest)
    outputs = []
    raw_path = Path(str(args.output_file) + ".raw.jsonl")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8") as raw_stream:
        for i, original in enumerate(evaluation):
            text, _ = render_text(original["tokens"], joiner)
            payload = {"model": args.model, "messages": messages + [{"role": "user", "content": text}],
                       "temperature": 0, "top_p": 1, "max_tokens": args.max_tokens}
            request = Request(
                args.base_url.rstrip("/") + "/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            )
            with urlopen(request, timeout=args.timeout) as response:
                raw = json.load(response)
            raw_stream.write(json.dumps({"sentence_id": sentence_id(original, i), "response": raw},
                                        ensure_ascii=False) + "\n")
            raw_stream.flush()
            content = raw["choices"][0]["message"]["content"]
            found, unmatched = exact_response_spans(content, original["tokens"], joiner)
            row = dict(original)
            row["sentence_id"] = sentence_id(original, i)
            row["original_labels"] = row.pop("labels", row.get("original_labels"))
            row["pico_elements"] = {}
            row["unmatched_outputs"] = unmatched
            for label, start, length in sorted(found):
                row["pico_elements"].setdefault(label, []).append(
                    {"span_start": start, "span_length": length, "confidence": 1.0}
                )
            outputs.append(row)
    write_jsonl(args.output_file, outputs)
    manifest.update(status="complete", records=len(outputs))
    write_json(str(args.output_file) + ".manifest.json", manifest)
    print(f"Saved {len(outputs)} predictions; evaluate using step_3_evaluate.py")


if __name__ == "__main__":
    main()
