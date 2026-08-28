# Partitioned specialist-teacher inference

This three-seed execution reruns the exact scientific configuration of the
`specialist-distillation` variant from `specialist-distillation-v1` after an
implementation-only optimization. Each rollout observation row is routed to
exactly one of the verified initial, ecology, or combat teachers. Teacher
networks evaluate compact disjoint sub-batches instead of each evaluating the
entire mixed-stage batch.

All scientific results are identical to the previous implementation for seeds
42, 43, and 44: the same best update and action count, retained on-food and
adjacent-food survival, skirmish kills, and skirmish damage. All 3/3 runs remain
jointly qualified.

Mean throughput increased from 274.8 to 303.4 sampled actions/s (+10.4%) and
from 46,115.3 to 50,905.0 simulation quanta/s (+10.4%). Relative to the original
control mean, the remaining gap is approximately 2.1% for actions and 2.4% for
simulation quanta, down from 11.3% and 11.6%. Per-seed action-rate changes
versus the earlier full-batch implementation were +19.8%, +17.3%, and -3.1%; the
third result is a reminder that short wall-clock comparisons remain noisy and
were not interleaved on the same binary.

The execution contract hash distinguishes the optimized trainer binary while
the experiment configuration hashes match the earlier specialist runs for each
seed. `aggregate.json` contains the independently verified qualification and
throughput summary.
