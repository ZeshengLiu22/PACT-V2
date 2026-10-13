# Full-Field Storm Surge Emulator

A PyTorch training framework over the existing model-ready monthly sample
lists, organized into data, model, and training modules. `smoke` is a trainable
linear pipeline check, not a scientific emulator.
`base` implements MLP, GNN, and temporal-first ST-GNN baselines. `igarss27`
implements temporal encoding, a spatial U-Net, bilinear mesh projection,
static mesh encoding, scale fusion, and MLP/GNN decoding. `full_v1` (A0) and
`full_v2` (A1) remain independent interfaces that raise `NotImplementedError`.

## Repository structure

```text
scripts/
    __init__.py
    train.py                      # CLI and experiment orchestration
    benchmark_models.py           # One-update real-data model diagnostics
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
        base.py                   # Base MLP / GNN / ST-GNN
        igarss27.py                # IGARSS temporal / spatial / fusion / decoder variants
        common.py                 # Physical bilinear projector and shared sparse GNN
        full_a0.py                # Deferred Full A0 interface
        full_a1.py                # Deferred Full A1 interface
    training/
        __init__.py
        config.py                 # YAML loading and supported-option validation
        engine.py                 # Training/evaluation execution and accumulation
        losses.py                 # Spatial weights and physical error totals
        metrics.py                # Complete-population overall RMSE/MAE
        checkpoints.py            # Saving, resume validation, and state restoration
        runtime.py                # Device/DDP setup, timing, and memory monitoring
        logging.py                # Timestamped console/train.log messages and summaries
        smoke.py                  # Loading benchmark and shuffle diagnostics
tests/
    __init__.py
    fixtures.py                   # Shared temporary monthly fixtures
    test_dataset.py               # Splits, indexing, history, mesh/node identity
    test_numerics.py              # Normalization, weights, loss, metrics, gradients
    test_training.py              # Forward/backward and checkpoint/resume
    test_models.py                # All histories/variants and geometry regressions
    test_model_integration.py     # Model variants through trainer and saved-config reload
    test_logging.py               # Formatting, persistence, resume, and rank ownership
README.md
```

Package imports are explicit, for example
`from emulator.data.dataset import FullFieldDataset`. `scripts/train.py`
coordinates the experiment; numerical code and model definitions live in the
package. The
structure follows the responsibility boundaries of
[StormSurge](https://github.com/ZeshengLiu22/StormSurge), retaining PACT-V2's
full-field tensor contracts and loading strategy. The existing SmokeModel stays
in its original module; model selection is a small conditional in the trainer.

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

## Base and IGARSS models

Every model returns `model(batch, mesh)["surge"]` as normalized `[B,N,6]`.
The existing trainer still owns normalization, physical-meter loss, WW validity,
spatial weighting, metrics, and checkpoint selection. Base/IGARSS ignore tidal
predictors and use weather channels 0:3. No bathymetry or target/state inputs
enter these models. No Dataset, preprocessing, source-data, or loss changes are
needed. History remains `data.history_hours`, one of `0,6,...,48`:

| History hours | 48 | 42 | 36 | 30 | 24 | 18 | 12 | 6 | 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Input steps T | 9 | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 |
| Base MLP/GNN input width | 32 | 29 | 26 | 23 | 20 | 17 | 14 | 11 | 8 |

Base MLP/GNN derive their first-layer width at initialization from
`3*T+5`; a mismatched input history raises a clear error. ST-GNN and IGARSS use
history-independent temporal weights and accept every nonempty supported
sequence, including the single issue-time frame at H=0. All heads stay width 6.

Base architectures (default hidden width 64):

- `mlp`: flatten raw weather to 3*T channels, bilinear projection, concatenate
  five mesh features, Linear(64)/GELU/Linear(64)/GELU/Linear(6).
- `gnn`: the same raw projection and static features, two mean-neighbor GNN
  layers of width 64, Linear(6).
- `stgnn`: project each raw weather time independently, shared single-layer
  GRU(3,32) per node, concatenate five mesh features, two width-64 GNN layers,
  Linear(6). Time is never flattened into input features for this variant.

IGARSS uses a temporal-only encoder independently at each atmospheric cell:
single-layer GRU/LSTM(3,32), a two-layer padded Conv1d temporal CNN with mean
readout, or a three-layer causal Conv1d TCN with dilations 1/2/4 and final-time
readout. The TCN receptive field covers all nine allowed frames; left padding
also supports T=1. None of these blocks mixes neighboring atmospheric cells.

The separate spatial U-Net has encoder widths [32,64,128], two 3x3/ReLU
convolutions per block, bilinear down/up resizing, and concatenated skip
connections. F0/F1 are decoded full/half-resolution features; F2 is the coarse
bottleneck. For the real 29x32 grid these are 29x32, 14x16, and 7x8. Resizing
uses `align_corners=True` and exact skip sizes, preserving endpoint coordinates
and odd dimensions. The U-Net architecture is unchanged across scale subsets.
Only selected features pass through 1x1 convolutions to 32 channels before
bilinear projection. Any nonempty subset of [0,1,2] is supported.

The **static mesh encoder** is a separate two-layer, width-32 GNN over the five
mesh features, evaluated once per forward. Fusion concatenates projected
features and broadcasts static features. Adaptive fusion uses a shared
Linear(64,32)/Tanh/Linear(32,1) score, softmax over scales, and **weighted
concatenation**. Its default 2,113 extra parameters leave the fused width
unchanged: 32*number_of_scales+32. Single-scale adaptive weight is exactly one.
The **dynamic mesh decoder** is either two width-64 Linear/ReLU layers or a
separate two-layer width-64 GNN, followed by Linear(6). Static and dynamic GNNs
have independent weights.

Every graph layer uses self features plus the incoming-neighbor mean. One
normalized sparse [N,N] adjacency handles features reshaped to [N,B*C]; no
batched edge copies or dense adjacency are constructed. Sparse multiplication
and interpolation use FP32 inside autocast. DDP does not broadcast immutable
geometry buffers; coarse-only IGARSS projections enable unused-parameter
detection for the unused finer U-Net decoder weights.

Replace only the `model` section of the existing complete YAML, for example:

```yaml
model:
  name: base
  variant: stgnn                 # mlp | gnn | stgnn
  hidden_dim: 64
  gnn_layers: 2
  temporal_hidden_dim: 32
```

```yaml
model:
  name: igarss27
  temporal_encoder: gru         # gru | lstm | temporal_cnn | tcn
  temporal_hidden_dim: 32
  spatial_encoder: unet
  unet_channels: [32, 64, 128]
  scales: [0, 1, 2]
  projector: bilinear
  projected_dim: 32
  mesh_encoder: gnn
  mesh_hidden_dim: 32
  mesh_encoder_layers: 2
  fusion: adaptive              # concat | adaptive
  decoder: gnn                  # mlp | gnn
  decoder_hidden_dim: 64
  decoder_gnn_layers: 2
```

The default complete YAML remains the diagnostic smoke experiment with batch
size 64 and accumulation 1. Resume reconstructs Base/IGARSS from the saved model
configuration and `data.history_hours` before loading weights; changing history
or architecture on resume is rejected by the existing identity checks. Full A0
and A1 have only independent, explicitly unimplemented interfaces.

### Physical projection geometry

The actual NCEP grid has longitude increasing from -99.375 to -41.25 degrees
at 1.875-degree spacing, and latitude decreasing from 54.284599 to 0.952368
degrees with slightly nonuniform spacing (about 1.9047 degrees). It is
rectilinear. Canonical mesh bounds are longitude [-97.852112,-59.858116] and
latitude [7.912744,45.973202]. All 159,444 nodes are covered; outside count is 0.
The first and last sample in each of all 177 monthly files were checked:
all 354 nine-frame coordinate histories match the same grid.

The projector takes original physical longitude/latitude from the Dataset
before normalization. It uses interval lookup on each observed axis to compute
fractional source indices, including descending and nonuniform axes, and caches
normalized sampling positions once. All scales use the same endpoint-aligned
positions with `grid_sample(..., align_corners=True)`. Node order is preserved.
Nonrectilinear grids or outside nodes raise errors; outside errors include the
count and first ten node indices/locations. Coordinates are never silently
reversed, clipped, or interpreted as normalized weather features. This is
bilinear interpolation, not a neural operator. No unresolved NCEP geometry
issue was found.

### Model validation and measured CPU runtime (2026-10-13)

Implemented and tested on PyTorch 2.7.1+cu126, CPU:

- Complete unit suite: **38 tests passed**, including the existing 19 pipeline
  regressions and the new model/integration tests.
- All nine histories for all three Base variants (27 forward/backward cases).
- All nine histories for all four IGARSS temporal encoders, both fusions, both
  decoders, and all seven nonempty scale subsets (1,008 forward/backward cases).
  Every used parameter has finite gradients. Coarse-only subsets intentionally
  leave finer U-Net decoder parameters unused while preserving the architecture.
- Temporal cell independence, TCN causality and oldest-frame influence, odd-size
  skip connections, analytic bilinear interpolation and orientation, independent
  static/dynamic GNN weights, shared graph topology, and adaptive weighted fusion.
- Saved-config model reconstruction at both history endpoints; 20 existing-trainer
  update/evaluation/checkpoint/resume cases across ten representative variants.
- Fifty real NCEP trainer updates: ten representative cases at H=48,42,36,24,0,
  batch size 1, all 159,444 nodes and 907,014 directed edges. Every result was
  `[1,159444,6]` with finite output, physical loss, and all parameter gradients.
- Actual CLI smoke and resume for Base GNN H0 and IGARSS TCN/adaptive/GNN H42
  passed on a small synthetic five-year dataset, including global shuffle and
  selected-checkpoint verification. Evidence is retained in
  `/glade/u/home/zel/tmp/pact_models_cli_83ovsfcp/report.json`.
- Two-process CPU DDP passed at H0/H48 for Base GNN/ST-GNN and IGARSS adaptive
  GNN decoding, including coarse-only [2] and [1,2] scales. CPU BF16 diagnostics
  passed for temporal variants; this does not validate CUDA BF16 kernels.

Real-data diagnostics reused the previously saved **full TRAIN H48** normalization
and source-year 1979 samples. Shorter histories reused these statistics only to
test shape/runtime; their losses are not scientific evaluation results. Normal
training still computes full TRAIN statistics for its configured history.

The following are measured first-update times, with four CPU threads, batch 1,
H48, full mesh, and default widths. Forward excludes the physical loss; backward
includes loss gradients; Adam timing excludes gradient validation. These are
single diagnostic updates including lazy initialization, not throughput estimates.

| Case | Configuration | Parameters | Forward s | Backward s | Adam s |
| --- | --- | ---: | ---: | ---: | ---: |
| B1 | Base MLP | 6,662 | 0.0413 | 0.2621 | 0.0011 |
| B2 | Base GNN | 12,806 | 0.1239 | 0.2483 | 0.0005 |
| B3 | Base ST-GNN | 16,998 | 0.4186 | 1.7079 | 0.0007 |
| I1 | GRU, all scales, concat, MLP | 505,926 | 0.1655 | 0.3217 | 0.0024 |
| I2 | GRU, all scales, adaptive, MLP | 508,039 | 0.2169 | 0.4005 | 0.0026 |
| I3 | GRU, all scales, adaptive, GNN | 520,327 | 0.3242 | 0.5849 | 0.0026 |
| I4 | GRU, scale [0], concat, MLP | 495,622 | 0.1362 | 0.2960 | 0.0023 |
| LSTM | LSTM, all scales, concat, MLP | 507,110 | 0.1956 | 0.3198 | 0.0024 |
| Temporal CNN | temporal CNN, all scales, concat, MLP | 505,798 | 0.1606 | 0.3225 | 0.0027 |
| TCN | causal TCN, all scales, concat, MLP | 508,902 | 0.1655 | 0.3282 | 0.0027 |

CPU process cumulative high-water RSS reached 0.760 GiB after B1, 0.852 GiB
after B2, and 2.484 GiB after B3 and all subsequent cases. These are process
high-water marks, not isolated per-model memory measurements. Separate component
profiling found the B3 temporal GRU took 0.292/0.424 s of forward time (69%).
For I3, the dynamic GNN decoder took 0.146/0.327 s (45%) and the static mesh GNN
took 0.073 s (22%). No performance-driven architectural changes were made.

Full per-history measurements and profiling evidence are retained at
`/glade/u/home/zel/tmp/pact_v2_models_evidence/cpu_histories.json` and
`/glade/u/home/zel/tmp/pact_v2_models_evidence/cpu_stage_profiles.json`.
Geometry evidence is in the same directory's `geometry_consistency.json`.

**Not tested:** CUDA forward/backward, GPU BF16/sparse operations, GPU memory,
and batch size 64. This allocation has no available CUDA device. The configured
batch size remains 64; no GPU-memory feasibility claim is made. On a GPU
allocation, run from the repository root:

```bash
/glade/work/zel/conda-envs/pact-casper-cu126/bin/python -m scripts.benchmark_models \
  --device cuda --batch-sizes 1 64 --histories 48 42 36 24 0 \
  --output /glade/u/home/zel/tmp/pact_v2_models_evidence/gpu.json
```

The benchmark synchronizes CUDA timings, reports GPU model and peak allocated/
reserved memory, records OOM/failure without reducing batch size, and exits
nonzero if any case fails. Smaller batches cover all ten cases and selected
histories; larger batches cover B1-B3/I1-I4 at H48. Use `--device cpu
--batch-sizes 1` for CPU diagnostics, or `--profile-stages --cases B3 I3
--histories 48` for component timings. Full A0/A1, scientific model selection,
and full training experiments are deferred.

## Outputs and timing

Each run writes `resolved_config.yaml`, `split_manifest.json`, `normalization.pt`,
`last.pt`, `best_overall.pt`, `training_metrics.jsonl`, and `summary.json`, plus
`train.log` for human-readable messages. The JSONL and summary schemas remain
unchanged. Only VAL AllRMSE selects `best_overall.pt`; final TEST uses that checkpoint.
No predictions or per-node diagnostics are exported.

`summary.json` includes measured initialization, standalone random-access
DataLoader timing, model-forward-plus-loss, backward, optimizer, and validation
execution time. Training batch-wait timing is reported separately because
prefetch can overlap computation. CPU memory includes main-process peak RSS
and process-tree PSS sampled every 0.5 seconds (shared mappings/IPC pages are
apportioned instead of counted once per worker). Sampling can miss brief peaks.
CUDA peak allocated memory is null when CUDA is unavailable. Timings reflect
filesystem caching and concurrent HPC activity, not guaranteed throughput.

## Training logs

Python writes timestamped `[YYYY-MM-DD|HH:MM:SS]` messages to the console and
`<output_dir>/train.log`; no shell launcher is needed. Early device/indexing
messages appear immediately in the console. File logging begins after resume
identity and output-directory checks, with a concise initialization summary.
The log includes normalization progress, actual model parameter counts,
checkpoint events, final metrics, and existing timing/memory measurements.

Every epoch reports weighted TRAIN GlobalMSE in m², the actual learning rate,
and newly measured VAL AllRMSE/AllMAE in m when validation is scheduled.
Otherwise it explicitly reports that VAL was not scheduled. Displayed epochs
are 1-based; checkpoint and JSONL epoch numbering is unchanged.

Normal batch progress appears after the first batch, every 25 batches, the
final batch, or after at least 60 seconds since the previous report, checked
when a batch completes. Smoke mode retains every batch diagnostic.
Normalization still reports every ten monthly files and the last file.

New runs start a fresh `train.log`. YAML resume appends to the original run's
log and prints a `RESUMING TRAINING` separator, restored/starting epochs,
previous best checkpoint state, and normalization reuse. Under DDP only rank
zero writes normal messages and the shared file. Batch progress is explicitly
rank-zero-local; epoch summaries retain the existing global reductions.
Other ranks report errors with their rank in the console. Exceptions retain
tracebacks and still fail the process; available rank-zero tracebacks are also
written to `train.log`.

## Known limitations

`smoke` uses a linear layer over spatially averaged latest forcing and broadcasts
its six outputs to every mesh node. It is an integration model. Reserved
`full_v1` and `full_v2` names fail explicitly; Base and IGARSS are implemented.

Tail and GT-aligned Episode Peak losses await the Episode Audit. Nonzero weights
fail clearly. The nine canonical extreme/episode metrics, exceedance checkpoint
selection, Full A0/A1 architectures, and cross-dataset inference remain pending.
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
