# ImageNet-100 Evidence Status

ImageNet-100 experiments are **reported, with verification pending**. This status separates reported observations, illustrative examples, modeled metrics, and projections. It does not assert that experiments were never run.

## Evidence available

`results/RESULTS_SUMMARY.md` reports best validation accuracies of 92.12% and 91.92%, and total runtimes of 514.7 and 519.8 minutes. Two result dashboards are committed under `results/visualizations/`. No raw ImageNet-100 run logs, machine-readable metrics, or checkpoints were found in the audited main tree at `3e66290cbf8951eb1fcaefabee9ff6b7a39df75b`. The reports therefore remain unverified; the dashboards do not replace raw artifacts.

The 77.2% accuracy, 89.9% savings, 10.05% activation, 184.2 minutes, and 9.9× speedup in the original setup examples are illustrative, not measured evidence.

## Metric definitions and limitations

- **Accuracy:** distinguish best accuracy across both stages, final AST accuracy, and warmup accuracy. A best score achieved during warmup does not establish AST accuracy preservation.
- **Baseline:** the README lists 92.18%, without a linked baseline run. 92.12% minus 92.18% is −0.06 percentage points; it is not an improvement. Repeated matched runs are needed to assess uncertainty.
- **Modeled energy savings:** scripts assume cost 1.0 per selected sample and 0.01 per skipped sample. These arbitrary units are not measured joules or kWh. Overall savings weight the AST estimate by the fraction of epochs outside warmup.
- **Estimated speedup:** scripts use `1 / (warmup_fraction + ast_fraction * target_activation_rate)`. With 10 warmup epochs out of 50 this gives 1.92× at 40% target activation and 2.78× at 20%. There is no matched timed baseline in that calculation.
- **Activation:** the production implementation forwards every sample, then masks losses; selection fraction is not total compute savings. Its `avg_activation` export uses the last epoch, rather than an average across the complete run. Smoothed controller rates and actual selected-sample counts must be distinguished.
- **Scope:** these reports concern pretrained ResNet50 fine-tuning on ImageNet-100, not ImageNet-1K training or validation of sparse-attention projections. The production script defaults to 50 epochs, including 10 warmup epochs.

## Evidence required before stronger claims

- [ ] Run ID, commit, seed, complete configuration, software versions, and hardware.
- [ ] Dataset source, 100-class manifest, train/validation split counts, and overlap checks.
- [ ] Per-epoch validation accuracy, selected/total sample counts, and timings, with warmup and AST phases identified.
- [ ] Checkpoint hash and reproducible evaluation procedure.
- [ ] Matched full-data baseline using the same initialization, hardware, training budget, and optimizations.
- [ ] Repeated runs and uncertainty for accuracy-preservation claims.
- [ ] Recorded power/energy measurements for measured energy-savings claims.

## Safe publication wording

ImageNet-100 experiments are reported in this repository, including best validation accuracies of 92.12% and 91.92%. Raw artifacts needed for independent reproduction are not committed. Energy savings use an assumed sample-cost model, and speedups are activation-based estimates rather than measured comparisons with a matched baseline.
