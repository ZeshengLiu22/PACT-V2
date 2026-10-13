# Full-Field Storm Surge Emulator

A PyTorch training framework over the existing model-ready monthly sample
lists, organized into data, model, and training modules. `smoke` is a trainable
linear pipeline check, not a scientific emulator.
The reserved `base`, `igarss27`, `full_v1`, and `full_v2` architectures are not
implemented.

## Repository structure

```text
scripts/
    __init__.py
    train.py                      # CLI and experiment orchestration
configs/ncep.yaml                 # Existing experiment configuration
emulator/
    __init__.py
    data/
        __init__.py
        dataset.py                # FullFieldDataset: mapped months and global indexing
        mesh.py                   # Shared mesh loading and schema validation
        splits.py                 # Deterministic simulation-year partitions
        normalization.py          # TRAIN-only monthly FP64 moments and statistics
        loaders.py                # Training/evaluation loaders and samplers
    models/
        __init__.py
        smoke.py                  # Existing linear SmokeModel
    training/
        __init__.py
        config.py                 # YAML loading and supported-option validation
        engine.py                 # Training/evaluation execution and accumulation
        losses.py                 # Spatial weights and physical error totals
        metrics.py                # Complete-population overall RMSE/MAE
        checkpoints.py            # Saving, resume validation, and state restoration
        runtime.py                # Device/DDP setup, timing, and memory monitoring
        smoke.py                  # Loading benchmark and shuffle diagnostics
tests/
    __init__.py
    fixtures.py                   # Shared temporary monthly fixtures
    test_dataset.py               # Splits, indexing, history, mesh/node identity
    test_numerics.py              # Normalization, weights, loss, metrics, gradients
    test_training.py              # Forward/backward and checkpoint/resume
README.md
```

Package imports are explicit, for example
`from emulator.data.dataset import FullFieldDataset`. `scripts/train.py`
coordinates the experiment; numerical code and model definitions live in the
package. The
structure follows the responsibility boundaries of
[StormSurge](https://github.com/ZeshengLiu22/StormSurge), retaining PACT-V2's
full-field tensor contracts and loading strategy. No registry, manager, callback
framework, or future-model placeholder is required.

## Run

Run commands from the repository root so Python resolves both the `scripts` and
`emulator` packages. The canonical entry point is `python -m scripts.train`.

Dependencies: PyTorch >= 2.7 and PyYAML. The verified local environment is
`/glade/work/zel/conda-envs/pact-casper-cu126/bin/python` (PyTorch 2.7.1+cu126,
PyYAML 6.0.3). The default shell Python does not have PyTorch.

```bash
cd /glade/u/home/zel/scratch/PACT-V2
/glade/work/zel/conda-envs/pact-casper-cu126/bin/python -m unittest discover -s tests -v
/glade/work/zel/conda-envs/pact-casper-cu126/bin/python -m scripts.train --config configs/ncep.yaml --smoke-test
```

Smoke mode keeps the YAML defaults, fits statistics on **all TRAIN samples**,
benchmarks up to six consecutive globally shuffled batches, trains up to four
real batches, and evaluates a reproducible subset of at most two batches per
actual held-out split (128 samples with the default batch size of 64). It saves
and reloads the validation-selected checkpoint and verifies its validation
metrics. Smoke metrics are limited-run integration results. Each fresh smoke
run gets its own timestamped directory under the configured
`experiment.output_dir` (`results/` by default).

```bash
# Copy the resolved config and set training.resume_from to the run's last.pt.
cp results/RUN/resolved_config.yaml /tmp/pact-resume.yaml
# After editing that YAML, continue smoke mode for one limited training pass.
python -m scripts.train --config /tmp/pact-resume.yaml --smoke-test
# Full configured training and complete VAL/TEST evaluation.
python -m scripts.train --config configs/ncep.yaml
# Ordinary DDP, with the number of processes chosen for the allocated resources.
torchrun --standalone --nproc_per_node=2 \
    --module scripts.train --config configs/ncep.yaml
```

Resume uses the YAML `training.resume_from` key; there is no resume CLI flag.
It writes back to the checkpoint directory and restores model, Adam, scheduler,
TRAIN normalization, and epoch state. Keep `best_overall.pt` alongside `last.pt`.
Resume checks the data, split, mesh
identity, loss, and training configuration. This is basic epoch-boundary resume;
it does not promise bitwise continuation of random sampling after interruption.
BF16 autocast runs only on supported CUDA devices. CPU execution reports AMP
as disabled. No batch-size adjustment or automatic OOM retry is performed.

## Data and numerical behavior

`data.root/data.source` selects one source; there is no source whitelist. No
input files are rewritten. Simulation-year directory names define split groups,
including calendar-month files from the following year. Splitting follows
StormSurge's [year-split algorithm](https://github.com/ZeshengLiu22/StormSurge/blob/main/emulator/data/graph_store.py):
sorted years, optional Python seeded shuffle, Python `round`, and the original
small-dataset boundaries. `future_only` means strictly greater than the threshold.
A training run requires a real VAL split. Missing TEST is reported explicitly.

`emulator/data/dataset.py` maps each monthly `.pt` list once and builds a global
sample index. The configured Linux DataLoader workers inherit these mappings.
PyTorch's built-in
`calculate_storage_offsets=True` avoids unnecessary random archive-header reads
on the shared filesystem. Before the refactor, a real monthly-file benchmark
reduced loading from 24.038 s to 0.156 s with identical storage
offsets/shapes/dtypes. The OS loads and shares file-backed pages on demand;
this avoids both full preloading and
repeated deserialization of random monthly files. The observed NCEP archive is
about 231 GiB. Initialization checks sample shapes and canonical node identity
against metadata, then discards full metadata. It retains only timestamps and
file/local indices. The shared mesh stays outside every dynamic batch.

TRAIN forcing statistics use the selected most-recent history and FP64
accumulation. Target statistics are one unweighted WW-only global mean/std.
Normalization eagerly loads and reduces one month per worker, using the
configured DataLoader worker count (at least one), then releases that month.
Only small FP64 moments are combined, in file order; the whole TRAIN population
is never retained in RAM. Eager reads avoid the measured slow page-fault I/O
of the initial sequential mapped scan (the pre-refactor one-month probe peaked
at 1.69 GiB). Stored non-WW surge NaNs and all four state labels remain unchanged. Models
receive forcing/tide inputs and the shared mesh; target surge/state are never
model inputs. Float64 tide values and times are preserved.

Only physical-meter Global MSE is trained. Spatial weights must carry exactly
matching ordered `node_id`; zero weights exclude supervision/evaluation,
positive weights are binarized in uniform mode. Overall metrics are unweighted
on eligible WW node-hours and reduce sums/counts over the whole evaluated
population. DDP evaluation partitions indices without padding. Gradient
accumulation divides summed gradients by the complete accumulation window's
weight count, including partial windows and distributed reduction.

## Outputs and timing

Each run writes `resolved_config.yaml`, `split_manifest.json`, `normalization.pt`,
`last.pt`, `best_overall.pt`, `training_metrics.jsonl`, and `summary.json`.
Only VAL AllRMSE selects `best_overall.pt`; final TEST uses that checkpoint.
No predictions or per-node diagnostics are exported.

`summary.json` includes measured initialization, standalone random-access
DataLoader timing, model-forward-plus-loss, backward, optimizer, and validation
execution time. Training batch-wait timing is reported separately because
prefetch can overlap computation. CPU memory includes main-process peak RSS
and process-tree PSS sampled every 0.5 seconds (shared mappings/IPC pages are
apportioned instead of counted once per worker). Sampling can miss brief peaks.
CUDA peak allocated memory is null when CUDA is unavailable. Timings reflect
filesystem caching and concurrent HPC activity, not guaranteed throughput.

## Known limitations

`smoke` uses a linear layer over spatially averaged latest forcing and broadcasts
its six outputs to every mesh node. It is an integration model. Reserved
`base`, `igarss27`, `full_v1`, and `full_v2` names fail explicitly.

Tail and GT-aligned Episode Peak losses await the Episode Audit. Nonzero weights
fail clearly. The nine canonical extreme/episode metrics, exceedance checkpoint
selection, real model architectures, and cross-dataset inference remain pending.
The only reported scientific metrics are `all_rmse` and `all_mae`, following the
[reference metrics](https://github.com/ZeshengLiu22/StormSurge/blob/main/docs/METRICS.md).


## Refactor verification (2026-10-13)

The pre-refactor baseline passed all nine unit tests. The reorganized suite
passes those same tests plus one checkpoint schema/resume-guard regression test
(10 total). All package imports and the unchanged CLI help were checked.

Small deterministic CPU smoke, resume, and two-process Gloo DDP runs matched the
saved pre-refactor implementation: model, Adam, scheduler, and normalization
states were bitwise identical; metric logs, checkpoint fields, and artifact
schemas matched exactly. Time and memory measurements were excluded from value
comparisons. DDP included uneven held-out partitions and partial gradient
accumulation windows. A two-worker persistent-loader smoke comparison also
matched, and the refactored CLI resumed a checkpoint written by the old code.
All new validation outputs were written outside the repository's `results/`.

The full NCEP normalization benchmark and GPU/BF16 execution were not repeated.
The measurements below are historical evidence from the original pipeline.

## Historical NCEP verification (2026-10-13, before this refactor)

All real NCEP checks and measurements below were completed by the preceding
implementation. They were preserved as evidence and were **not rerun or
remeasured for this structural refactor**. Refactor validation uses focused
unit tests and small CPU fixtures; the expensive full TRAIN normalization is
not required to validate module moves. GPU/BF16 execution remains unvalidated.

The existing `dataset/NCEP` symlink points to
`/glade/u/home/zel/scratch/FullField/Processed_Samples/NCEP`. Its 177 monthly
sample/metadata pairs contain 21,280 samples across simulation years 1979–2014.
The exact groups are TRAIN 1979–2000 (13,224 samples), VAL 2001–2007 (4,208), and
TEST 2008–2014 (3,848). The 2014 group has November/December only: Full/Tidal
target coverage ends at 2015-01-01 00:00 UTC, so later complete days are absent.
No input files or preprocessing were changed.

Observed tensors:

| Tensor | Shape | dtype |
| --- | --- | --- |
| forcing | [9,5,29,32] | float32 |
| tide_boundary | [141,8,5] | float64 |
| tide_potential | [8,5] | float64 |
| tide_time | [6] | float64 |
| surge / state | [159444,6] | float32 / int64 |
| node_id | [159444] | int64 |
| pos / x | [159444,2] / [159444,5] | float64 / float32 |
| edge_index / edge_attr | [2,907014] / [907014,3] | int64 / float64 |
| face / boundary_node_index | [3,293923] / [141] | int64 |

Before the refactor, nine focused tests, two-process CPU DDP, real-data smoke,
and real-data YAML resume all passed. Checks cover indexing, simulation-year
isolation, global shuffling, history slices, separate mesh, WW normalization, spatial weighting,
NaN-safe physical loss, accumulated gradients, complete-population metric
reduction, forward/backward, validation, checkpoint reload, and resume.
The first real batch contained samples from 22 years and 49 monthly files;
all four returned batches were checked against their global shuffled indices.

The unchanged default batch/worker settings were 64 / 4, with pinning enabled,
persistent workers, and prefetch factor 2. PyTorch correctly disabled actual
pinning on this CPU-only allocation. Initial-run measurements:

| Measurement | Observed |
| --- | ---: |
| Dataset/mesh/schema initialization | 53.10 s |
| Full TRAIN normalization (12,587,140,442 WW observations) | 817.36 s |
| DataLoader construction | 0.00142 s |
| First standalone random batch | 45.64 s |
| Subsequent five random batches | 7.54, 0.90, 0.00022, 29.08, 7.60 s |
| Standalone loader throughput, including startup | 4.23 samples/s |
| Subsequent-batch throughput | 7.09 samples/s |
| Mean forward plus physical loss | 1.732 s |
| Mean backward | 0.473 s |
| Mean Adam step | 0.000759 s |
| Mean compute per training step (excluding loading) | 2.206 s |
| VAL: 128 samples | 54.28 s; 2.36 samples/s |
| TEST: 128 samples | 115.94 s; 1.10 samples/s |
| Sampled peak process-tree PSS / main peak RSS | 15.45 / 7.75 GiB |
| GPU allocated memory / GPU throughput | Not tested: CUDA unavailable |

Cold I/O is a substantial limitation. The identical six-batch benchmark during
resume benefited from filesystem cache: 270.61 samples/s including startup.
This is not a guarantee for a full shuffled epoch. All timings are actual
measurements, not estimates. No GPU, BF16, or multi-GPU performance is claimed.

Resume advanced Adam from four to eight updates and the scheduler to epoch two,
reused normalization, and selected epoch 1 using VAL only. Final limited-run
metrics were VAL AllRMSE 0.166941 m / AllMAE 0.125555 m and TEST AllRMSE 0.190682 m /
AllMAE 0.138172 m. These each cover 128 samples, not the full held-out splits.

Run artifacts: [initial measurements](results/NCEP_Smoke_smoke_20261013T142506Z/summary_initial.json),
[resume/final summary](results/NCEP_Smoke_smoke_20261013T142506Z/summary.json),
[exact split manifest](results/NCEP_Smoke_smoke_20261013T142506Z/split_manifest.json),
and [verification evidence](results/verification/). The run also contains
`resolved_config.yaml`, `normalization.pt`, `last.pt`, `best_overall.pt`, and
`training_metrics.jsonl`. Verification retains the stopped I/O experiments as
incomplete attempts, not successful smoke runs.
