# CCG-PICO reviewed recommendation annotations

The workbook contains 721 Chinese guideline records with reviewed structured P/I/O
fields and their Final recommendation text. The original source file is preserved.
Data preparation and full-model validation are in progress.

<!--
A field value and a strict entity offset are different representations: the workbook
does not contain character start/end columns.
-->

## Field mapping and exact offsets

The converter uses:
- P: Disease, Subtype, Age, Gender, Other_Demographic, Comorbidity.
- I: Intervention_Name, Intervention_Detail.
- O: Outcome_Indicator, Outcome_Target.

Intervention category/action and guideline metadata are retained as source fields,
not automatically treated as entity text. The current Control field is empty.

~~~bash
python prepare_ccg_pico.py export --workbook datasets/CCG-PICO/ccg_pico_recommendations.xlsx --output data/ccg/aligned_records.jsonl
~~~

Every aligned span retains its source field. Coordinates refer to original Final
characters, use [start,end), and preserve the exact original substring. Whitespace
normalization may help locate a value but never rewrites the source text.

<!--
Unmatched and repeated values are retained as unresolved issues with candidates.

On the current source, 10 records uniquely align all mapped fields; 711 records
have unresolved offsets. This is a conversion-status count, not a judgment that
their structured annotations were not reviewed. The report includes each record
ID, original value, reason and candidate positions.
-->

## Supply complete-record overrides

For a record requiring reviewed offset decisions, provide its complete intended
span list. The original Final text must match exactly; an explicit empty span list
can represent a deliberate no-entity decision. The example below is synthetic:

~~~json
{
  "EXAMPLE_ID": {
    "text": "甲病指南：甲病患者接受乙药。",
    "spans": [
      {"type": "PARTICIPANTS", "start": 5, "end": 9, "text": "甲病患者"},
      {"type": "INTERVENTIONS", "start": 11, "end": 13, "text": "乙药"}
    ]
  }
}
~~~

Use real record IDs and exact Final text. An override replaces the complete
record's automatically aligned spans; it does not just append to them.

~~~bash
python prepare_ccg_pico.py export --overrides PATH_TO_OFFSETS.json --output data/ccg/aligned_records.jsonl
~~~

Unknown IDs, changed text, invalid categories and mismatched substrings are rejected.
The source workbook remains unchanged.

## Fixed, template-preserving splits

~~~bash
python prepare_ccg_pico.py build --records data/ccg/aligned_records.jsonl --output data/ccg/json --train_size 541 --validation_size 90 --test_size 90 --seed 0
~~~

<!--
Build refuses unresolved offsets.
-->

Build creates boundary and multilabel span JSONL,
retains explicit gold_spans, and saves all split record IDs and source hashes.

The default template key masks aligned spans and numbers in normalized text;
an explicit template_id may be included in an aligned record when known.
The splitter retains template groups, tries 64 seeded group assignments with
exact target sizes, and chooses the assignment with smallest P/I/O prevalence
deviation. It raises an actionable error when those sizes cannot be formed
during that search. This method approximates label stratification while honoring
group and size constraints.

For an original verified partition, use --split_manifest PATH. The builder
checks complete, disjoint record membership and no cross-split template group.
A new 541/90/90 partition must not be described as the original paper partition.

Run the common training/calibration/evaluation pipeline on data/ccg/json.
Paddle ERNIE-Health requires the [conversion adapter](../../docs/ERNIE_HEALTH.md).

## Attribution

This dataset is contributed by the MC-PICOX project authors. For redistribution
or use beyond research, contact the repository owner.
