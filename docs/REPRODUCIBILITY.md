# Experiment protocol

## Freeze inputs before comparing variants

1. Preserve raw-source hashes and a verified split manifest. Keep all records from
   the same document/template in a single split.
2. Use one actual encoder/tokenizer ID and revision per language. Record hidden
   size and position limit. The public base default is not a substitute for a
   claimed large-model experiment.
3. Train the span classifier once per seed and dataset version; reuse its
   checkpoint across the boundary-model ablation conditions.
4. Fix class/NMS thresholds and long-input policy across comparisons. If selecting
   settings, use validation only and record that selection.
5. Keep the final test set out of model/threshold/demonstration selection.

## Cumulative conditions

| Condition | Boundary training | Decoder |
|---|---|---|
| Baseline | --model_type baseline, CE | Fixed documented rule/threshold |
| MC | --model_type multichannel --loss_type ce | Same fixed rule/threshold |
| MC+FL | --model_type multichannel --loss_type focal | Same fixed rule/threshold |
| MC+FL+AT | Reuse the MC+FL training checkpoint | Validation-selected separate thresholds |

For an isolated calibration comparison, use --decode_rule sum for all fixed
conditions and record their fixed thresholds. The original PICOX OR rule remains
available as --decode_rule legacy_or for a separately identified historical
comparison. Changing both the OR/sum rule and threshold tuning must be stated;
it is not a pure threshold-only intervention.

The paper's historical settings and original training artifacts have not all been
recovered. The protocol above defines the repaired implementation; it does not
retroactively validate the paper's tables.

## Metrics and selection

Boundary checkpoint selection defaults to pooled start/end recall over labeled
subwords, with argmax five-class predictions. BOTH contributes one start and one
end. Ignored special/padding positions contribute neither labels nor counts.
Window-overlap labeled subwords are counted as the encoded evaluation examples;
this is not the final word-level span metric.

Threshold selection uses strict end-to-end micro F1 on validation: entity type,
start and length must all match. Searches cover the inclusive range 0.05..0.30
in steps of 0.01 for each endpoint. Class mass is summed before subword/window
max aggregation. All probability and boundary settings are saved.

The evaluator emits integer TP/FP/FN, per-type metrics, micro metrics and a macro
average over the three P/I/O categories, including zero scores for absent classes.
Exact duplicate predictions do not inflate TP or FP. Classification score
comparisons are inclusive; boundary comparisons are strict, as in the method.

## Artifacts for each reported table row

- Code commit and working-tree diff hash; use clean commits for final experiments.
- Dataset source hash, reviewed offsets and split manifest.
- Actual encoder/tokenizer path or ID, revision and saved configuration.
- Random seed, optimizer settings, epochs, batch size, loss and focal parameters.
- Selection metric, best checkpoint, validation metrics and training state.
- Probability file, frozen threshold configuration and complete search grid.
- Checkpoint hashes, tokenization/window settings, classification and NMS settings.
- Final predictions, raw TP/FP/FN and evaluation provenance.
- For a generative baseline: fixed demonstrations, prompt, provider/model identity,
  decoding settings, raw responses and unmatched/parse diagnostics.

The scripts record manifests next to the generated artifacts. A training manifest
never reads the test set. Record training seeds individually; if reporting mean
and standard deviation, retain each run and state the aggregation convention.
Fixed seeds do not by themselves establish bit-identical training on different
GPU hardware or runtime versions.

## Offline verification

~~~bash
python -m unittest discover -s tests -v
~~~

The tests create tiny models/tokenizers locally. They cover the actual Trainer
interface and checkpoint restoration, full public command flow, long windows,
focal math/masking, confidence NMS, schema shape, deterministic splits, all 676
threshold pairs, test-calibration rejection, reviewed-field alignment, template
groups, explicit span gold, Qwen parsing and demonstration leakage prevention.

The numerical scores from this tiny fixture are software checks, not biomedical
extraction results.
