# CCG-PICO Recommendation Dataset

`ccg_pico_recommendations.xlsx` contains structured Chinese clinical guideline
recommendations for constructing and evaluating a Chinese PICO extraction
corpus.

## Content

The spreadsheet includes:

- recommendation text;
- disease, age, gender, and comorbidity information for Participants;
- intervention names and intervention details;
- outcome indicators and target values;
- normalized final recommendation text.

Structured fields are not automatically equivalent to gold text spans.
Paraphrases, combined fields, and repeated expressions require string alignment
and manual boundary review before the data are used for strict span-level PICO
evaluation.

## Suggested Use

1. Map structured P/I/O fields to exact substrings in the recommendation text.
2. Review unmatched and ambiguous boundaries manually.
3. Convert the reviewed annotations to the JSONL format expected by MC-PICOX.
4. Split the resulting corpus into training, validation, and test sets without
   using test records for threshold selection.

## Attribution

This dataset is contributed by the MC-PICOX project authors. For redistribution
or use beyond research, contact the repository owner.
