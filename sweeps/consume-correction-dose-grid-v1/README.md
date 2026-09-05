# Consume-focused correction dose grid V1

This paired grid tests whether the relatively small `Consume -> Attack` margin
measured after the first correction A/B is actually learnable. Every arm starts
from the V4 hard-context control clone and trains for two epochs at `5e-5`, seed
42, with exactly 14,806 presentations and 64 optimizer updates per epoch. The
control spends all presentations on the original 12-dataset rehearsal mixture.
Treatments add only the pure-Consume on-food correction corpus at weights 3.75,
7.5, and 15: respectively 20%, 33.3%, and 50% of each fixed epoch.

Eight fresh evaluation seeds span `1000000101` through `1000000808` using the
documented nonuniform suffixes. Exact-input margin probes use seed
`1010000101`. No adjacent-food movement corrections enter training.

## Result

The Consume boundary is learnable, and the smallest tested correction share is
sufficient. All three treatment arms have identical measured greedy rollout
behavior, so weight 3.75 dominates the larger doses.

| metric | rehearsal | 20% correction | 33.3% correction | 50% correction |
|---|---:|---:|---:|---:|
| presentations | 29,612 | 29,612 | 29,612 | 29,612 |
| optimizer updates | 128 | 128 | 128 | 128 |
| on-food survival | 0% | 100% | 100% | 100% |
| on-food intake / initial cell | 57.000 | 249.000 | 249.000 | 249.000 |
| adjacent-food survival | 0% | 37.3% | 37.3% | 37.3% |
| adjacent intake / initial cell | 23.930 | 127.297 | 127.298 | 127.298 |

The selected 20% arm consumes essentially the entire starting plant and keeps
every on-food cell alive through the evaluation horizon. Its behavior is not
yet a complete foraging policy: once the plant is depleted it continues to
choose `Consume`. On the new policy-induced frontier, the teacher labels 1,024
Consume choices followed by 1,024 Move choices, while the policy chooses
Consume throughout. The post-depletion `Move -> Consume` error has a mean
policy advantage of 11.131 logits.

Larger correction shares do not improve rollout behavior and make this new
error more confident: 11.876 logits at 33.3% and 12.428 at 50%. The 20% arm is
therefore the only reasonable parent for a second DAgger round. That round
should label the states reached after depletion and restore movement without
reintroducing the already-crossed `Consume -> Attack` boundary. Promotion is
still blocked: adjacent-food survival is only 37.3%, and combat retention has
not yet been rerun.

## Bound artifacts

| artifact | rehearsal | 20% correction | 33.3% correction | 50% correction |
|---|---|---|---|---|
| behavior-clone metadata | `41b8f3ad4e4555859236d95337bd38edc28633b7383fb0e9a4825a06cf570824` | `8e3f713d2008937ed913a431fb9f2539085d3de102c0cfee668f0a0573cfc032` | `62dc798cabe4f30eafe07ca19a782ae5de2f26cb3bbd4b8f265e57e72f5e8c7f` | `c3d20f13a7adaf3bbd9abb777b0c85cb9d715abbcbdfedc6eae92fae08c9d8b2` |
| model | `944186356d589f356b6231fafe31a6172f650db0bb317aa647643fa4d763f442` | `e098ec7dc6e2d4cc4ed796add7e52ef01caccab339ef5ed7c30ca6e07e2e2bbd` | `62a125bdb77b871f90191768b60471ca6f2f96d036cd29f8b0c5cfb8f69989ae` | `da4532aca1a75ddf72d94a6244812386a99ceeb2b4f7366b21dcd1a6d55281ec` |
| feeding evaluation | `e8d304c4e57908af838688f8650fabc831a1bfaecafb9df4fd7bb33f9b40e6d9` | `21b1572aff0762b9bd75ad924011f495e38d8a3be6d8da14d008cc05136c35a2` | `866cd2efbac015ee32b47a3249f8032e2809ef6448e2d7c839d13a891ea4f949` | `db6242b8d868e29f59895d771c755a88e17c93e325c2159df5f89b7da0511ccd` |
| margin manifest | `6462cf55cb83a3052ee1c8e0f10fe3a783cabb06b28e3765556d9f8717f1177a` | `89941a796d2423a36dab5aa6d6b33da74f5eef16cb9fd4a9053428476986b853` | `c6e89ace3114ecac3f5ff6c2e07d57ca05fff8fe93899a9a329a62af0ed62c27` | `dcf05546f22c42606c71f0d5151668089a00d94df40db5c81b7292792b134b6f` |

Local artifacts are retained under
`training-output/consume-correction-dose-grid-v1`. The reusable runner is
`scripts/run_consume_correction_dose_grid.sh`.
