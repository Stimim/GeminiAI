# Minesweeper Flags

A two-player Minesweeper game and AI playground. Players alternate opening cells: claiming a mine scores a point and grants another move; a safe cell reveals its adjacent-mine count (zeros flood-fill) and passes the turn. Each player also has one 5×5 bomb: it claims unopened mines, reveals safe cells, and always ends the turn. First to a majority of the mines wins. The default board is 16×16 with 51 mines, first to 26.

## Architecture

- **Exact constraint solver:** derives mine probabilities from the visible clues and enumerates consistent assignments.
- **ResNet evaluator:** fully convolutional mine, open-policy, bomb-policy, and value heads, using 20 public-information input planes. Open policy is `beta * mine_logit + residual(features)`; the bomb head starts from clipped 5×5 expected mine counts. Open and bomb actions share a masked softmax over `2 × board cells`.
- **Information-set MCTS:** searches from public information, sampling consistent determinizations with the constraint solver.

### Search controls

`MCTSAgent` reuses only descendants matching the observed public board, scores,
player and bomb inventory; unseen opponent extra turns can traverse several
levels. A missing outcome starts a fresh tree. No sampled mine layout is retained.
Offline arena and paired comparisons use the same reuse rule.

`Search(..., search="puct"|"gumbel", sims=100, gumbel_k=16)` supports ordinary
PUCT and Gumbel-top-k root sampling with budgeted sequential halving. Each
halving round gives surviving candidates equal visits before eliminating half.
The candidate count is reduced when necessary to fit every halving round in the
budget. As before, a fresh root evaluation counts as one simulation; a reused
expanded root spends the whole budget on edges. Gumbel's final action maximizes
the surviving candidates' Gumbel + log-prior + completed-Q score, without another
random draw. `improved_policy()` returns the legal masked softmax of log-prior
plus `(50 + max_visits) * 0.1 * normalized_completed_Q`; completed values are
min-max normalized to [0,1] (a constant range maps to zero). Unvisited actions
use the prior-weighted visited-value / network-value mixture from
[Danihelka et al. (ICLR 2022)](https://iclr.cc/virtual/2022/poster/6418).
`gumbel_c_visit=50` and `gumbel_c_scale=0.1` match the reference Mctx transform.
Q is accumulated in each node's mover perspective, so the root's completed Q
already has the root player's sign, including mine extra turns and bomb passes.
Play/arena and compare disable Gumbel noise by default (`gumbel_noise=False`);
self-play enables it. `--gumbel-noise` / `--no-gumbel-noise` explicitly select
sampling or deterministic top-k. With noise off, one candidate and one simulation
exactly reproduce the greedy policy action.
Non-root selection remains PUCT: it already averages sampled chance outcomes
and preserves the existing player-relative backup/extra-turn handling, isolating
the root-allocation change rather than changing two selection algorithms at once.
Open and bomb actions share the same masked action distribution.

`next_leaves(K)` / `provide_batch(policies, values)` reserve independent pending
leaves with virtual visits and loss. Reservations live in separate counters,
so completion reverts virtual loss exactly without modifying real sums.
Duplicate unexpanded public outcomes end a batch; root expansion and halving
boundaries are evaluation barriers. `K=1` preserves sequential PUCT behavior.
`MCTSAgent(..., leaf_batch_size=4, tree_reuse=True)` controls play batching.
Play/server/offline-arena MCTS agents and training/arena CLI use K=4 by default;
pass `leaf_batch_size=1` or `--leaf-batch-size 1` for the exact sequential path.
Self-play and compare accept `--leaf-batch-size`, `--search` (compare: `--mode`)
and `--gumbel-k`. Self-play stores the completed-Q target for Gumbel, visit targets
for PUCT. Registry categories and published web identities are unchanged.

For example, an unrecorded fresh-layout comparison against the pre-reuse baseline:

```sh
uv run python -m msflags.compare --checkpoint-a checkpoints/run-medium/selfplay-1794006042719861991.pt --checkpoint-b checkpoints/run-medium/selfplay-1794006042719861991.pt --mode gumbel --sims 32 --mode-b puct --sims-b 100 --no-tree-reuse-b --leaf-batch-size 1 --games 200 --seed 940000 --threads 1
```

Compare independently configured sides with `--mode-b`, `--sims-b`,
`--leaf-batch-size-b` and `--no-tree-reuse-b` (a fresh-tree baseline).
The result includes paired-layout bootstrap intervals and each side's total
search/inference seconds divided by its move count. Concurrent-game timings
are amortized throughput costs, not single-game interactive latency.

Self-play accepts `--workers 4 --cpu-share 5` or explicit `--threads 1
--worker-threads 1`. `--cpu-share` reserves one trainer thread and splits the
remaining threads evenly among spawned workers; it rejects budgets too small
to give every process one thread. Across N runs on a 20-thread machine, allocate
at most `floor(20/N)` threads per run (including its trainer), subtracting other
active workloads first. Four simultaneous runs can therefore use four workers
plus a trainer each only if no additional CPU workload needs a share; six workers
need at least seven threads per run. More workers than concurrent game batches
cannot help. This is a thread budget, not CPU affinity or an OS scheduling quota.

Spawned workers keep CPU search state only. Their leaf requests are coalesced in
the trainer process, which owns the sole network/GPU context; inference is split
into at most 64 leaves per call to bound activation memory. Weights therefore
cannot go stale between training iterations, and adding workers no longer
duplicates CUDA contexts or model state. Worker requests and inference failures
are resolved before pool shutdown; Linux workers are killed if their parent dies.
Before large benchmark pools, check `free -m`; retain at least 3 GiB available
memory, and close each pool before starting the next configuration.

Training retention measures the configured disk root globally but deletes only
obsolete checkpoints within that run's own `--output` directory. Benchmark output
belongs under `/tmp` or `checkpoints/dev-*/`; it must not prune live run directories.

## Setup

```sh
uv sync
```

## Training

```sh
uv run python -m msflags.train bootstrap --output checkpoints/bootstrap-medium --data checkpoints/bootstrap-bomb-2000.npz --network-size medium --games 2000 --steps 8000 --workers 12
uv run python -m msflags.train bootstrap --output checkpoints/bootstrap-small --data checkpoints/bootstrap-bomb-2000.npz --network-size small --steps 8000
uv run python -m msflags.train arena --agent-a r64x4-bs8k.mcts100-c1.5 --agent-b solver --games 200
```

Check subcommand options with `uv run python -m msflags.train bootstrap --help`, `uv run python -m msflags.train selfplay --help`, or `uv run python -m msflags.train arena --help`.

Train from scratch or continue a supported 20-plane checkpoint:

```sh
uv run python -m msflags.train bootstrap --checkpoint checkpoints/bootstrap-medium/latest.pt --output checkpoints/continued --data checkpoints/bootstrap-bomb-2000.npz --steps 8000 --device cuda
```

Bomb solver baseline: open any certain mine first; otherwise detonate the highest expected-mine window when its expectation is at least three (and exceeds the best single opening). A bomb gives up the usual extra turn, so the threshold trades immediate multi-mine gain against saving the one-use resource. Bootstrap explores all legal actions and stores a joint open/bomb policy; uncertain solver fallback positions are excluded from exact labels.

A quick alternating-seat sweep (40 games each, seeds 81000–81039) gave thresholds two and four each 20/40 wins against threshold three (6.3 seconds total). This small experiment does not establish an optimum; three remains the conservative midpoint baseline.

### Fresh bomb bootstrap

The registered medium (64 channels × 4 blocks) and small (32 × 3) networks start from random weights and share 2,000 explored bomb-solver games. `--network-size small|medium` selects architecture; numeric `--size` remains the square board width. No transferred variant is registered. Its checkpoint remains on disk; retired-agent arena results are preserved in `checkpoints/arena/archive-transferred.jsonl`, with the pre-reset ledger backed up alongside it. Solver/random-only results remain active.

The shared corpus contains **112,302 exact-labelled positions** (epsilon 0.08, seed 20261107), generated in 11.9 seconds. Both sizes trained for 8,000 steps at lr=0.001, batch size 128; medium took 126.5 seconds end to end (including corpus generation), small 98.0 seconds using the same dataset. On 200 alternating-seat greedy games against solver (seed 310000), medium won **136/200 (68.0%, 95% CI 61.2–74.1%)** and small **105/200 (52.5%, 45.6–59.3%)**.

Medium MCTS-100 won **131/200 (65.5%, 95% CI 58.7–71.7%)** against solver with the same arena seed, in 412.3 seconds under concurrent load; greedy evaluations each took 9.7 seconds. These evaluations are unrecorded head-to-head tests. The fresh 20-games-per-pair round robin added 200 ledger games in 195.0 seconds. With 21 retained solver/random games, the 221-game leaderboard is medium greedy **1145.4 ±113.2**, small greedy **1054.8 ±109.4**, medium MCTS **1019.3 ±109.3**, solver **1000**, random **6.9 ±607.0** Elo. Small pair counts do not establish a reliable ordering between close agents.

The reset archived **100** retired-agent games and retained **21** solver/random games. Backup: `checkpoints/arena/games-before-reset-1793983677327548715.jsonl`. Bootstrap and promoted self-play checkpoints are pinned in `msflags/generations.json`.

## Web UI

Run the server with `uv run python -m msflags.server` and open http://127.0.0.1:8000. Neural catalog entries use pinned timestamped checkpoints; random and solver need no weights. Use `--host`, `--port`, and `--device cpu|cuda` to configure it. The interface supports human-vs-AI, AI-vs-AI, and hot-seat games, with hints and move-by-move animation.

### AI catalog and arena

The catalog has six categories: Random, Greedy exact solver, and ResNet 64×4 / 32×3 with either Greedy policy or MCTS-100 (c 1.5). Each neural category has immutable checkpoint generations; both search modes of an architecture share its pinned generation list. Generation 0 is bootstrap; generation k is completed self-play iteration k. Category IDs (`r64x4.greedy`, `r64x4.mcts100-c1.5`, `r32x3.greedy`, `r32x3.mcts100-c1.5`) select the latest available generation. Baselines have one generation each. Category detail shows architecture, search settings, training summaries and generation ratings; generation detail retains full checkpoint provenance.

Generation IDs remain `{arch}-{training}.{search}`: for example `r64x4-bs8k-sp12.greedy`. They derive from checkpoint metadata, not filenames or list position. Bootstrap uses `bs{steps in k}`, followed by `-sp{completed iterations}` for self-play. Search is `greedy` or `mcts{sims}-c{c_puct}` (plus `-f{fpu}` for nondefault FPU reduction; default 0.2). Missing provenance and duplicate identities are rejected. Arena records and Elo always belong to concrete generations, never moving categories. Play’s simulation budget applies only to MCTS; arena always uses the category’s fixed settings.

Only 20-plane checkpoints with joint open/bomb policies are supported.


Arena games alternate the starting seat and append to `checkpoints/arena/games.jsonl`. One match runs at a time per server process; live boards expose only public information. Match progress is in memory; completed games survive restarts. Omit the API seed for fresh random games, or supply a nonnegative seed to reproduce a match. The offline round-robin batches up to 32 games for efficient neural inference; the web arena runs one game at a time for a coherent live view. Ratings reread the ledger, so newly completed CLI games appear without restarting the server. Run one server process when using the arena.

#### Round-robin CLI

`--games` is the number of games **per pair**. By default the latest available generation of each category participates; six categories produce fifteen pairs, so `--games 20` records 300 games. Use `--agents r64x4.greedy solver r32x3.mcts100-c1.5` to select categories, or pass immutable generation IDs. `--seed 200000` makes the tournament reproducible. The `ratings` command refits all persisted results without additional games. The training CLI's `arena` command accepts generation IDs; `--checkpoint` / `--checkpoint-b` override the corresponding neural weights for unrecorded evaluations.

```sh
uv run python -m msflags.arena roundrobin --games 20 --device cuda
uv run python -m msflags.arena ratings
```

#### Ratings method

Ratings fit all recorded games jointly with a Bradley–Terry logistic model, fixing the greedy solver at **1000 Elo**. The win probability is `1 / (1 + 10 ** ((rating_b - rating_a) / 400))`. The fit maximizes the likelihood with a Gaussian regularizer centered at 1000, standard deviation **800 Elo**, for every non-anchor agent, keeping perfect win/loss records finite. The leaderboard reports games played and approximate 95% uncertainty half-widths (`1.96 × standard error`) from the inverse fitted Hessian, conditional on the fixed anchor; these are not guarantees of future performance. Random is rated, not used as the anchor.


API: `GET /api/agents` returns `{agents: [category, …]}`. Each category has `id`, `name`, `kind`, `summary`, `available`, `default_generation` (latest available ID), and `generations`. Generation rows include `id`, `category_id`, `generation`, `label`, `available`, `games`, and, when rated, `elo` and `uncertainty` (95% half-width). `GET /api/agents/{id}` accepts category or generation IDs. Game seats and hints also accept either; game creation resolves categories once and stores concrete generation IDs. `POST /api/arena/matches` (`{a,b,games,seed}`) resolves contenders before recording results. `GET /api/arena/matches/{id}` includes the live public board; `GET /api/ratings` retains per-generation ratings. Unknown/unavailable agents and invalid counts are rejected; a second active match returns HTTP 409.

#### Adding a checkpoint or agent

1. Train to a timestamped checkpoint, keeping its run's `metrics.jsonl` alongside it. Only completed immutable checkpoints are eligible; never promote `latest.pt`.
2. Run `uv run python -m msflags.registry promote checkpoints/my-run/selfplay-<timestamp>.pt`. This validates the checkpoint and architecture, rejects duplicate paths/identities, and appends it to that architecture's tracked generation list in `msflags/generations.json`. Both search categories acquire the generation together.
3. Inspect `uv run python -m msflags.registry list`, open `/agents.html`, and rate the generation with `uv run python -m msflags.arena roundrobin --agents <generation-id> solver --games 20 --device cuda`.

Never overwrite, rename or repoint pinned weights: historical results are keyed by their derived generation IDs. A missing pinned file remains visible as unavailable, and category defaults skip it. `registry.pinned_checkpoints()` exposes all referenced paths so checkpoint pruning preserves promoted weights. New architecture/search categories are explicit definitions in `registry.py`; promotion does not invent categories.

## Training runs

`checkpoints/run-medium/` (64×4) and `checkpoints/run-small/` (32×3) resume existing optimizer, replay, and iteration counters. The output directory's `latest.pt` always takes precedence over `--checkpoint`. `--iterations 0` is open-ended; positive values are total target iteration numbers, not additional iterations. Interrupted unpublished work is regenerated; the command's learning rate overrides the restored optimizer learning rate.

### Round 3: active control continuations

The completed trial decision keeps both controls: medium resumes **g113** with
lr **2e-4**, small resumes **g82** with lr **1e-3**; both use value weight **0.25**.
Those immutable finals remain promoted. The losing value/LR forks stay stopped.
The historical trial/round-2 sections below are records of earlier launches,
not the current process configuration.

Round 3 runs from main under regular **Python 3.14.4**, open-ended
(`--iterations 0`), using PUCT-100, K=4, tree reuse, and a **single parent GPU
evaluator per run**. Each run has **four CPU-only workers plus one trainer
thread** (`--cpu-share 5`): ten budgeted CPU threads across both runs, below the
twenty-thread host limit and below ten per run. This is a conservative,
memory-first split, not permission to duplicate GPU models in the workers.
Each iteration retains 96 games / 16 concurrent per task, c=1.5/FPU=.2,
Dirichlet alpha=.15/fraction=.25, temperature 1 for 30 moves, 300 updates ×128,
50,000-position replay, mine weight 2, and each control's original seed.
Optimizer, replay and counters resume from that directory's `latest.pt`.
Retention is enabled and deletion is scoped to that run's output directory.

Current detached process groups: **medium 1805211**, **small 1805222**; the
authoritative IDs are in each `train.pid`. Exact argv, source revision,
starting iteration and pre-launch metrics are in `checkpoints/round3-launch.json`.
**Do not execute these launch commands again while the groups are active.**

```sh
setsid nohup uv run python -m msflags.train selfplay \
  --output checkpoints/run-medium --iterations 0 --games 96 --concurrent-games 16 \
  --workers 4 --worker-threads 1 --cpu-share 5 --threads 1 \
  --search puct --sims 100 --leaf-batch-size 4 --c-puct 1.5 --fpu-reduction .2 \
  --steps 300 --batch-size 128 --replay-size 50000 --lr .0002 --value-weight .25 --mine-weight 2 \
  --eval-every 5 --eval-games 100 --value-validation-every 5 --value-validation-games 16 \
  --seed 610000 --device cuda --disk-root checkpoints \
  >> checkpoints/run-medium/train.log 2>&1 &
echo $! > checkpoints/run-medium/train.pid

setsid nohup uv run python -m msflags.train selfplay \
  --output checkpoints/run-small --iterations 0 --games 96 --concurrent-games 16 \
  --workers 4 --worker-threads 1 --cpu-share 5 --threads 1 \
  --search puct --sims 100 --leaf-batch-size 4 --c-puct 1.5 --fpu-reduction .2 \
  --steps 300 --batch-size 128 --replay-size 50000 --lr .001 --value-weight .25 --mine-weight 2 \
  --eval-every 5 --eval-games 100 --value-validation-every 5 --value-validation-games 16 \
  --seed 620000 --device cuda --disk-root checkpoints \
  >> checkpoints/run-small/train.log 2>&1 &
echo $! > checkpoints/run-small/train.pid
```

Monitor logs and available RAM:

```sh
tail -F checkpoints/run-medium/train.log checkpoints/run-small/train.log
free -m
```

Graceful stop (wait for final iteration/checkpoint and process-group exit before
restarting either directory):

```sh
kill -TERM -- -"$(cat checkpoints/run-medium/train.pid)"
kill -TERM -- -"$(cat checkpoints/run-small/train.pid)"
```


#### First completed round-3 iterations

Both resume events confirmed the expected control counters (**113 / 82**) and
**50,000 replay positions** each. Both completed and published one new iteration,
including their scoped retention event, and both detached process groups remained
alive for the open-ended continuation.

| Run | First iteration | New positions | Generation / train / save seconds | Positions/s | Previous final-control positions/s |
|---|---:|---:|---:|---:|---:|
| medium | 114 | 8,075 | 150.73 / 7.01 / .179 | **53.57** | 13.13 |
| small | 83 | 10,998 | 228.74 / 6.90 / .160 | **48.08** | 14.38 |

Total first-iteration wall times were **158.37s / 236.18s**. These rates exceed
the earlier ~15–30 positions/s range; against the exact final pre-restart control
records they are **4.08× / 3.34×**. The historical records had different contention,
so these are end-to-end pipeline/resource gains, not an isolated Python speedup.

Available RAM: **11,717 MiB before launch**, **5,749 MiB after both resumes and
worker startup**, **5,076 MiB after medium114**, **4,677 MiB after small83**.
The one-second observer's sampled minimum was **4,216 MiB**, above the required
3 GiB floor; no safety stop was triggered. Each run retains four workers rather
than expanding both to six and spending that RAM margin.

First-iteration records and memory observations:
`checkpoints/round3-first-iterations.json`, also copied into
`checkpoints/dev-python314-round3-1794044881330994265/`.
The development worktree was removed only after unlinking its `checkpoints`
symlink; its fully merged branch was deleted, and the real main-checkout control
checkpoints were preserved.

### Runtime decision: Python 3.14

Main is pinned to regular **Python 3.14** (`.python-version`); the validated
interpreter is **3.14.4**, with Torch **2.15.0+cu132** and NumPy **2.5.4**.
The normal runtime has its GIL enabled. `uv sync` creates the matching environment.

Quiet-machine A/B: medium g80 checkpoint
`checkpoints/run-medium/selfplay-1794006042719861991.pt`, PUCT-100/K=4,
four CPU-only search workers, one shared parent GPU evaluator, one thread per
process, 48 games / eight concurrent games per task. Every timing uses the same
generation seed **2200000**, warm-up seed **2100000**, c=1.5/FPU=.2,
Dirichlet alpha=.15/fraction=.25 and temperature 1 for 30 moves. Three passes per
interpreter were interleaved 3.12→3.14, 3.14→3.12, 3.12→3.14; all pools closed
between timings. Asynchronous GPU batching can change trajectories slightly;
throughput uses each run's actual generated position count.

| Python | Repeat positions/s | Median positions/s | Min–max |
|---|---|---:|---:|
| 3.12.14 | 49.24, 50.27, 50.83 | 50.27 | 49.24–50.83 |
| 3.14.4 | 49.41, 51.51, 51.70 | 51.51 | 49.41–51.70 |

The median difference is **+2.47%** for 3.14, inside observed run-to-run spread,
not an established speedup. A supplementary paired-seed cohort (2200000,
2210000, 2220000) gave medians 50.23 vs 49.21 positions/s (−2.02%), also overlapping
spread. There is no material regression beyond that noise, so the user's 3.14
preference wins. Minimum sampled available RAM in the identical-seed comparison
was **8923 MiB**.

Tests: main trial changes **141 passed**; merged search code **193 passed**;
the separate 3.14 environment **193 passed**; after pinning and `uv sync`,
**193 passed in 32.24s**. Raw timings and summaries:
`checkpoints/dev-python314-round3-1794044881330994265/`.

**Free-threaded availability, not adopted:** `uv python list` offered 3.14t;
the throwaway probe `/tmp/msflags-py314t-round3-probe` installed binary NumPy 2.5.4,
Torch 2.15.0+cu132 and Triton 3.9.0 under **CPython 3.14.7 free-threaded**.
`Py_GIL_DISABLED=1`; the GIL stayed disabled after importing both NumPy and Torch,
and Torch reported CUDA available. This establishes wheel/import availability,
not free-threaded training performance. The first copy-mode install filled the
7.8 GiB `/tmp` tmpfs; `uv pip install --only-binary=:all: --link-mode=symlink
--reinstall numpy torch` against the probe interpreter fixed that without copying
CUDA libraries into tmpfs. Production remains regular 3.14, not 3.14t.


### Phase B: paired value/LR trials

The controlled comparisons are medium `run-medium` versus `run-medium-vw1` and small `run-small` versus `run-small-lr2e4`. Each fork starts from **exactly the control's published weights, optimizer, replay and iteration**; its immutable starting copy is `trial-start.pt`. `checkpoints/phase-b-trials.json` records the starting/target iterations, SHA-256 checksums, launch arguments and process-group PIDs. `train.pid` in each directory is the current process group. No generations are promoted merely for starting a trial.

| Directory under `checkpoints/` | Process group | Start → target | Learning rate | Value weight |
|---|---:|---:|---:|---:|
| `run-medium` (control) | 1757910 | 98 → 113 | 0.0002 | 0.25 |
| `run-medium-vw1` | 1757911 | 98 → 113 | 0.0002 | 1.0 |
| `run-small` (control; resumed from69) | 1782829 | 67 → 82 | 0.001 | 0.25 |
| `run-small-lr2e4` | 1757957 | 67 → 82 | 0.0002 | 0.25 |

All four receive equal resources: **one single-process self-play/trainer, one Torch thread, 16 concurrent games** per run. This replaces the old two spawned workers per control, keeping four aggregate search processes instead of eight workers plus four trainers. Controls remain active, not time-sliced. All retain 96 games, 100 simulations, c_puct 1.5, FPU reduction 0.2, Dirichlet alpha 0.15/fraction 0.25, temperature 1 for 30 moves, 300 updates ×128, 50k replay, mine weight 2 and the pair's existing seed. Targets are **15 additional completed iterations**, expressed as absolute `--iterations` values in the manifest. Runs stop automatically at their own targets; do not silently continue one arm farther than its mate.

All four launched detached after the old controls exited gracefully. At the initial four-arm resource snapshot, GPU allocation was **1,645 / 8,188 MiB**, host RAM had **8,397 MiB available**, and swap use was 54 MiB. These are instantaneous startup observations, not peak-memory or throughput guarantees; no control pausing was needed.

**Small-control interruption:** the original `run-small` process group 1757956 exited after publishing iteration **69**, not its target 82; its final retention event is epoch **1794013912.781088**, with no traceback or logged warning. Both medium arms reached 113 and the small fork reached 82. Historical exit codes were not retained. `journalctl -k` covering epochs 1794013500–1794014500 contained only GPU ioctl messages, and the captured kernel warning/error log had no OOM-killer entries: **the termination cause is undetermined, not a confirmed OOM**. At investigation time the shared host had only 2,978 MiB available and 3,069/4,096 MiB swap used, so subsequent evaluation/resume scheduling must preserve headroom alongside the dev-worktree benchmarks. Never compare control69 against fork82 or promote a small winner before equal-generation comparisons. Any approved continuation uses the manifest's exact original command, restoring iteration69 optimizer/replay and targeting82; it is a continuation of the same trial, not a new seed or settings change.

The approved small-control continuation launched detached at epoch **1794029396.1286852**, process group **1782829**. It uses the **exact original argv** from `phase-b-trials.json`, including lr0.001/value_weight0.25, seed620000, one thread/worker, MCTS100/c1.5/FPU0.2 and target82. Its resume event confirms **completed_iterations=69 / replay_positions=50,000**; optimizer restoration follows the existing checkpoint resume path. Headroom had recovered to **10,488 MiB available** immediately before launch. `checkpoints/run-small/resume-status.json` records the exact command, interruption provenance, new process group and eventual exit code; supervisor `SmallControlResume82` waits for completion. The other three arms remain stopped. Do not begin a new training round until the search-performance branch is merged and restart approval is given.

The continuation **completed iteration82 and exited with status0** at epoch **1794038776.5077822**. Final immutable control checkpoint: `checkpoints/run-small/selfplay-1794038775379820495.pt`. The final audit found **all four trial process groups empty, all targets reached, final retention records present, and no logged warning/error text**. No new training round was launched. The original three completed arms' historical OS exit codes remain unavailable; the resumed small control's exit code is explicitly retained.

First completed iterations after restart (96 games and 50,000 replay positions each; value is the final 100-update block's unweighted training MSE):

| Run | Iteration | New positions | Generation / train / save seconds | Value MSE | Bomb / safe / mine actions |
|---|---:|---:|---:|---:|---:|
| medium control | 99 | 7,717 | 592.45 / 9.56 / 0.151 | 0.249 | 184 / 4,217 / 3,316 |
| medium vw1 | 99 | 7,651 | 580.67 / 9.40 / 0.107 | 0.247 | 180 / 4,155 / 3,316 |
| small control | 68 | 10,347 | 872.02 / 7.14 / 0.175 | 0.500 | 151 / 6,706 / 3,490 |
| small lr2e4 | 68 | 10,319 | 866.91 / 7.44 / 0.093 | 0.516 | 157 / 6,708 / 3,454 |

All four published immutable checkpoints and reported exact counts for all 96 games; no warning/error text was found in their logs. Shared snapshot/seed does not imply bitwise-identical self-play: the observed first-iteration trajectories differ slightly, so treat the arms as controlled starting conditions, not identical replay trajectories. These first losses do not decide the trial. The first scheduled held-out validation is medium iteration 100 / small iteration 70; first-iteration `validation_seconds=0` is expected. Full initial records are in `/tmp/phase-b-first-iterations.json`.

Every five iterations, each run keeps the 100-game greedy-vs-solver sentinel and additionally generates **16 separate fresh self-play games** (`--value-validation-every 5 --value-validation-games 16`). Those complete games never enter replay. Validation uses the current self-play search/noise/temperature distribution, a separate seed namespace, final outcome labels, and frozen post-update predictions; it reports value MSE, sign accuracy, zero/score-leader baselines and total-score bands. Positions within a game are correlated: these are not thousands of independent validation games. Set `--value-validation-every 0` to disable explicitly.

`selfplay_games` now logs exact per-game moves/bomb/safe/mine counts, including terminal actions, and marks `split=replay|validation`. Iteration metrics separate `generation_seconds`, `training_seconds` (including replay encoding), `save_seconds` (replay plus checkpoint publication) and `validation_seconds`; total time also includes the greedy sentinel. Retention still occurs outside the iteration timer. Checkpoints now preserve their complete `training_settings`.

The search value audit found **no sign/perspective defect for the supported odd-mine game**: backup follows player identity (not depth), mine openings retain the turn, bombs/safe openings pass it, chance outcomes share parent action statistics, and terminal winners override leaf estimates. The regression suite covers both players, terminal bombs, mixed chance outcomes, Q/FPU initialization and root reuse; `uv run pytest -q` passed **141 tests**. Fresh-root expansion consumes one simulation, and self-play subtree reuse retains visit counts; these explain differences from fresh-tree arena search, not incorrect value signs. Even-mine tie handling is outside the supported training configuration and was not changed.

**Medium g80 search ablation:** immutable `selfplay-1794006042719861991.pt`, fresh trees/no noise, fixed independent layout seeds, both seats per layout, versus that same checkpoint's greedy policy:

| Search setting | Wins / games | Win rate | Wilson 95% |
|---|---:|---:|---:|
| 100 sims, FPU 0.2, c=1.5 | 108 / 200 | 54.0% | 47.1–60.8% |
| 200 sims, FPU 0.2, c=1.5 | 22 / 40 | 55.0% | 39.8–69.3% |
| 100 sims, FPU 0, c=1.5 | 112 / 200 | 56.0% | 49.1–62.7% |
| 100 sims, FPU 0.2, c=1.0 | 21 / 40 | 52.5% | 37.5–67.1% |
| 100 sims, FPU 0.2, c=2.5 | 23 / 40 | 57.5% | 42.2–71.5% |

FPU 0 won the exploratory 40-game screen, but its matched advantage over baseline is only **+2 percentage points, paired-layout bootstrap 95% −6 to +10**. On the 160-game extension excluding screening it scored 88 wins versus baseline's 89. No setting establishes an improvement; all trial arms therefore retain **100 sims/c=1.5/FPU=0.2**. Doubling simulations roughly halved measured games/hour (~538 versus ~1,031 for baseline) without established strength benefit. The other 40-game settings remain screening results, not adequately powered final comparisons.

On 96 fixed positions (32 per early/mid/late score stratum, sampled from iteration 92 replay), baseline search agreed with prior argmax **76.0%**, versus **75.0%** for 200 sims and **79.2%** for FPU 0; visit support increased from 6.29 to 10.60/11.64. This larger stratified sample shows more policy changes than Phase A's 24-position sample: prior dominance is not universal. Greater exploration did not establish stronger play. Exact seeds, checkpoint hash, position snapshot and full measurements are retained in `/tmp/search_ablation.py`, `/tmp/search-ablation-positions.npz` and `/tmp/search-ablation-results.{json,txt}`.

**Decision after 15 iterations:**

1. Wait until both arms of a pair reach the manifest's target. Use immutable target checkpoints (or the now-stopped `latest.pt` files). Compare at equal generation count, not whichever run finished first.
2. Run both greedy and fresh-tree/no-noise MCTS comparisons, **at least 200 games per mode and pair**. Every layout is independently seeded and played with both seats; action/search RNG cannot change later layouts:

   ```sh
   for mode in greedy mcts; do
     uv run python -m msflags.compare --checkpoint-a checkpoints/run-medium-vw1/latest.pt --checkpoint-b checkpoints/run-medium/latest.pt --mode "$mode" --games 200 --seed 910000 --device cuda --threads 1
     uv run python -m msflags.compare --checkpoint-a checkpoints/run-small-lr2e4/latest.pt --checkpoint-b checkpoints/run-small/latest.pt --mode "$mode" --games 200 --seed 920000 --device cuda --threads 1
   done
   ```

3. Treat MCTS strength as primary; inspect fresh-game value error and greedy strength as guardrails. The comparator reports a whole-layout paired-bootstrap 95% interval. If it overlaps 50%, retain the control or expand to 400–800 games; do not declare a win from a noisy point estimate. Confirm an apparent winner on a fresh seed bank before adopting it.
4. Promote **only the chosen immutable winner**, using `uv run python -m msflags.registry promote <checkpoint>`, and resume that arm's settings. Fork and control at the same iteration share the catalog's derived generation identity, so do not try to promote both. Keep the losing trial/checkpoint and its manifest as comparison provenance; do not repoint an existing generation.

Never start another trainer into an active directory. Stop any current arm gracefully with `kill -TERM -- -"$(cat checkpoints/<run>/train.pid)"`, then wait for its complete iteration/checkpoint and process exit before relaunching. The manifest contains the exact detached commands; the historical commands below are not the four-arm launch configuration.


### Historical round 2 launch and operation

Both runs use **96 games / iteration, MCTS-100, c_puct=1.5, unchanged Dirichlet noise, 300 updates × 128 samples**, and 50,000-position replay. Each has two persistent spawned self-play processes, each batching 16 games with one Torch thread; the trainer uses two Torch threads. Equal CPU budgets avoid assigning all 20 hardware threads to either run. Each worker owns its CUDA evaluator; completed games merge in seed/batch order.

Medium (detached process group **1734838**, also in `train.pid`):

```sh
setsid nohup uv run python -m msflags.train selfplay --output checkpoints/run-medium --checkpoint checkpoints/bootstrap-medium/latest.pt --iterations 0 --games 96 --concurrent-games 16 --workers 2 --worker-threads 1 --sims 100 --steps 300 --batch-size 128 --lr 0.0002 --threads 2 --replay-size 50000 --eval-every 5 --eval-games 100 --seed 610000 --device cuda >> checkpoints/run-medium/train.log 2>&1 &
echo $! > checkpoints/run-medium/train.pid
```

Small (detached process group **1734839**):

```sh
setsid nohup uv run python -m msflags.train selfplay --output checkpoints/run-small --checkpoint checkpoints/bootstrap-small/latest.pt --iterations 0 --games 96 --concurrent-games 16 --workers 2 --worker-threads 1 --sims 100 --steps 300 --batch-size 128 --lr 0.001 --threads 2 --replay-size 50000 --eval-every 5 --eval-games 100 --seed 620000 --device cuda >> checkpoints/run-small/train.log 2>&1 &
echo $! > checkpoints/run-small/train.pid
```

Monitor:

```sh
tail -f checkpoints/run-medium/train.log checkpoints/run-small/train.log
nvidia-smi
```

Stop gracefully at the next complete iteration; workers finish games and publish checkpoint/replay before exit:

```sh
kill -TERM -- -"$(cat checkpoints/run-medium/train.pid)"
kill -TERM -- -"$(cat checkpoints/run-small/train.pid)"
```

Wait for process groups to exit before repeating launch commands. Never launch two trainers into one run directory. `latest.pt` is atomically published after immutable weights, optimizer state, and replay are complete. Two replay slots preserve the previous published checkpoint during writes; only the latest checkpoint is guaranteed a matching replay for continuation. Metrics include self-play generation, losses, iteration throughput/reuse, evaluations every five iterations, and retention decisions.

### Throughput measurements

Fixed seed **730000**, bootstrap weights, **32 games / 100 simulations / 100 updates**, batch size 128, CUDA, outputs under `/tmp`. Both original live process groups were SIGSTOP-paused during each benchmark pair and SIGCONT-resumed afterward; the concurrent arena was paused for optimized measurements. GPU percentages are sampled whole-GPU means; CPU is approximate steady-state search-process utilization (100% = one logical CPU). Iteration time includes startup/pool costs.

| Network | Pipeline | Games/s | Seconds/iteration | GPU utilization | Search CPU |
|---|---|---:|---:|---:|---:|
| medium | baseline, 1×32 | 0.332 | 99.03 | 34% | ~102% |
| medium | optimized, 2×16 | 0.581 | 58.30 | 37% | ~206% |
| small | baseline, 1×32 | 0.350 | 93.78 | 29% | ~103% |
| small | optimized, 2×16 | 0.576 | 58.38 | 38% | ~203% |

Generation improved **1.75× medium / 1.65× small**. The earlier profile attributed 64% to forward+encoding and 15% to solver sampling. Encoding now batches neighbour sums instead of looping over states: a 32-state CPU microbenchmark improved **4.01 ms → 0.282 ms (14.2×)**. Neighbour sums accumulate in place; solver sampling caches conditional distributions and decoded reservoir cells without changing RNG draws. Persistent workers overlap CPU tree traversal and independent batched CUDA evaluation. Existing inference mode and CUDA BF16 remain enabled; search budgets, selection/noise, and sampling distributions are unchanged. Replay encoding uses bounded 1,024-position chunks.

First completed iterations of the final tested-code launches, with both runs active:

| Run | Iteration | Seconds | Games/s | New/replay positions | Sample reuse |
|---|---:|---:|---:|---:|---:|
| medium | 6 | 286.03 | 0.345 | 6,655 / 44,995 | 5.77× |
| small | 5 | 250.98 | 0.404 | 6,190 / 35,041 | 6.20× |

These contention-inclusive results are not the isolated benchmark. Both detached launches remain running and resumed previously published self-play weights, not bootstrap weights. The concurrent GPU snapshot reached 85% utilization at 2,024 / 8,188 MiB (including the arena).

### Disk budget

The budget is **100 GB decimal for the whole project**, including checkpoints, datasets, replay, logs, and the environment. After these iterations: **6.92 GB total project; 1.31 GB checkpoints** (logical file sizes, hard-linked `latest.pt` counted once). Self-play weights with AdamW occupy **3,934,581 bytes medium / 883,833 bytes small**. Each replay position is **3,339 bytes**: one saturated 50,000-position slot is ~167 MB; both slots for both runs plateau at ~668 MB. Until saturation, ~6,200–6,700 new positions add ~21–22 MB per run/iteration; replay is subsequently replaced, not accumulated.

Pruning runs before generation and after each published iteration. It keeps **all bootstrap checkpoints; every tenth self-play iteration; the newest three iterations per run; `latest.pt`; both replay slots; datasets; and every path returned by lazily imported `registry.pinned_checkpoints()`**. Only other immutable `selfplay-*.pt` files are removed. Iteration numbers come from checkpoint metadata, never timestamps. An unavailable registry hook keeps everything and emits a warning. If protected data still puts the project/checkpoints over 100 GB, training warns loudly and stops; milestones and pinned generations are never sacrificed.

```sh
uv run python -m msflags.train prune --dry-run
uv run python -m msflags.train prune
```

Without retention, a completed iteration pair adds ~4.82 MB of weights; every-tenth retention averages ~0.482 MB per pair, plus rolling newest-three files and additional pins. At observed concurrent iteration times, milestone growth is **~0.15 GB/day**, versus ~1.5 GB/day unpruned. Replay growth stops at its bound; logs and extra registered generations add to the estimate. The hard cap eventually stops training rather than deleting permanent generations.

Promote immutable timestamped weights using the registry CLI described above (`uv run python -m msflags.registry --help`), never `latest.pt`. Never repoint a registered generation to different weights.

## Tests

```sh
uv run pytest -q
```

## Deployment considerations (future: public hosting)

- Games and active arena matches live in process-memory dictionaries without expiry or eviction; memory grows, and multiple processes/replicas do not share them. Use a shared store (e.g. SQLite/Postgres or Redis) and expire sessions.
- Arena results use a local JSONL ledger, while web matches run in a server thread. For hosting, move persistence to a database and execution to a background job worker; restrict arena-match creation to authenticated admins because it starts long CPU/GPU jobs.
- MCTS is CPU-bound Python search with per-move network evaluations; one process may serve few concurrent AI moves. Cap simulations server-side, add rate limits and request timeouts, and consider separate batched inference; GPU versus CPU-only hosting is a cost choice (CPU PyTorch images are much smaller than CUDA images).
- The board's mine layout is withheld until game end, but client-supplied seeds make layouts reproducible and hints expose AI help. For ranked/online play, use server-side secret seeds and disable hints.
- Anyone with a game ID can act for either human seat (hot-seat design). Online human-vs-human needs per-seat tokens/accounts and push updates (WebSocket/SSE) instead of polling.
- Checkpoints are loaded with `torch.load(weights_only=False)`; never load untrusted files. Prefer `weights_only=True` or `safetensors` for deployment.
- Put HTTPS behind a reverse proxy; static files can be served by a proxy/CDN. Add a health-check endpoint and structured logs.


## Search performance experiments: medium g80

**TL;DR:** PUCT-100 with tree reuse and K=4 is the conservative search default:
about 2.4× cheaper than fresh-tree PUCT-100, with no detected strength loss.
Six CPU-only workers with one shared GPU evaluator reached 63.53 positions/s;
use fewer workers when other runs need the CPU/RAM budget.

Checkpoint: `checkpoints/run-medium/selfplay-1794006042719861991.pt`.
Hardware: RTX 4070 Laptop 8 GB, i7-13700H (20 threads), 15 GiB RAM.
Experiments began alongside four controlled training runs and another benchmark
worker; concurrent load changed during the experiments. No live run was paused
or edited. Each benchmark parent and search worker used one Torch thread.

### Design and measurement

- Reuse matches public board, scores, turn and bomb inventory only. Missing
  observed outcomes start fresh; opponent extra turns can traverse several levels.
- Virtual loss has separate reservation counters, avoiding floating-point undo
  of real values. K=1 is tested against sequential selection exactly.
- Gumbel uses joint legal open/bomb top-k candidates and equal-visit sequential
  halving. Candidate count is reduced as needed to fit the fixed budget; fresh
  root expansion consumes one simulation, as in the original PUCT implementation.
- Completed Q uses the root mover's perspective, a prior-weighted mixed value
  for unvisited actions, min-max normalization to [0,1], and
  `sigma=(50+max_visits)*0.1*normalized_Q`. Non-root selection remains PUCT.
- Evaluation disables Gumbel noise; self-play can enable it. Both settings were
  measured. A one-candidate, one-simulation, noise-off test reproduces greedy exactly.
- CPU workers now share one parent GPU evaluator, capped at 64 leaves per call.
  Failed requests are released before shutdown; Linux parent death kills workers.
- Retention measures the configured disk root globally but deletes only within
  the run's own output directory; a regression test protects a neighboring run.

Throughput uses 22 independent early/mid/late public positions from greedy games
(seeds 730100–730103), with three interleaved passes. Every position starts fresh.
Parallel=1 approximates interactive search; parallel=16 amortizes inference.
Strength uses 200 games per matchup: 100 independently generated layouts played
in both seats, in five interleaved 40-game blocks. The 95% intervals bootstrap
layout-pair scores (20,000 draws), not 200 independent outcomes.
Seconds/move are each side's total search/inference time divided by moves,
amortized over up to 16 concurrent games—not interactive latency.
Ranges below are observed repeat/block min–max under shared load.
All PUCT searches use c=1.5 and FPU reduction=0.2. Candidates reuse trees;
the baseline is fresh-tree PUCT-100, K=1, without Dirichlet noise.

### PUCT and initial Gumbel throughput

Initial Gumbel used noise-on and an unnormalized unit-scale Q transform.
Its rows are retained as diagnostic evidence, superseded by corrected results below.

| Search | Parallel | Positions/s median [min–max] |
|---|---:|---:|
| Initial Gumbel-16 | 1 | 8.86 [7.84–10.46] |
| Initial Gumbel-16 | 16 | 63.39 [30.03–69.31] |
| Initial Gumbel-32 | 1 | 4.41 [4.37–5.31] |
| Initial Gumbel-32 | 16 | 33.73 [30.48–34.53] |
| PUCT-100 K=1 | 1 | 1.44 [1.41–1.55] |
| PUCT-100 K=1 | 16 | 8.74 [8.49–10.41] |
| PUCT-100 K=4 | 1 | 4.52 [4.48–4.70] |
| PUCT-100 K=4 | 16 | 10.72 [7.08–12.41] |
| PUCT-100 K=8 | 1 | 7.48 [6.22–7.56] |
| PUCT-100 K=8 | 16 | 7.94 [4.22–11.99] |

### PUCT and initial Gumbel paired results

Block seeds: `940000 + repeat*10000 + candidate_index*1000 + opponent_offset`.
Indices 0–4 are Gumbel-16, Gumbel-32, PUCT-100 K=1, K=4, K=8;
opponent offset is 500 for baseline, otherwise zero.

| Candidate | Opponent | Wins/games | Win rate [95% CI] | Candidate s/move [min–max] | Opponent s/move [min–max] |
|---|---|---:|---:|---:|---:|
| Initial Gumbel-16 | baseline | 36/200 | 18.0% [13.0–23.0%] | .0282 [.0166–.0553] | .1547 [.0900–.3003] |
| Initial Gumbel-16 | greedy | 41/200 | 20.5% [15.0–26.5%] | .0291 [.0146–.0520] | .0012 [.0006–.0022] |
| Initial Gumbel-32 | baseline | 40/200 | 20.0% [14.5–25.5%] | .0565 [.0289–.1092] | .1574 [.0825–.3026] |
| Initial Gumbel-32 | greedy | 44/200 | 22.0% [16.5–28.0%] | .0583 [.0324–.1097] | .0012 [.0007–.0023] |
| PUCT-100 K=1 | baseline | 100/200 | 50.0% [43.5–56.5%] | .1689 [.0916–.3075] | .1675 [.0876–.3056] |
| PUCT-100 K=1 | greedy | 104/200 | 52.0% [44.5–59.0%] | .1743 [.0971–.3455] | .0013 [.0007–.0027] |
| PUCT-100 K=4 | baseline | 99/200 | 49.5% [43.5–55.5%] | .0690 [.0415–.1198] | .1655 [.0952–.2864] |
| PUCT-100 K=4 | greedy | 114/200 | 57.0% [50.0–63.5%] | .0695 [.0388–.1295] | .0013 [.0007–.0028] |
| PUCT-100 K=8 | baseline | 96/200 | 48.0% [41.5–54.5%] | .0499 [.0308–.0785] | .1589 [.0870–.2587] |
| PUCT-100 K=8 | greedy | 106/200 | 53.0% [45.5–60.0%] | .0513 [.0329–.0901] | .0012 [.0008–.0023] |

### Corrected Gumbel throughput

`off` means deterministic top-k (g=0); `on` means standard Gumbel noise.
Both use normalized completed Q, c_visit=50 and c_scale=0.1, with K=1.

| Search | Parallel | Positions/s median [min–max] |
|---|---:|---:|
| Gumbel-16 off | 1 | 15.12 [15.04–15.41] |
| Gumbel-16 off | 16 | 107.48 [50.48–110.23] |
| Gumbel-16 on | 1 | 14.96 [13.94–15.70] |
| Gumbel-16 on | 16 | 120.50 [64.60–121.99] |
| Gumbel-32 off | 1 | 7.61 [7.30–7.70] |
| Gumbel-32 off | 16 | 56.20 [44.64–58.12] |
| Gumbel-32 on | 1 | 7.56 [7.34–7.58] |
| Gumbel-32 on | 16 | 54.28 [54.14–55.06] |

### Corrected Gumbel paired confirmation

Fresh block seeds: `1200000 + repeat*10000 + candidate_index*1000 + opponent_offset`.
Indices 0–3 are G16-off, G32-off, G16-on, G32-on; opponent offset is as above.

| Candidate | Opponent | Wins/games | Win rate [95% CI] | Candidate s/move [min–max] | Opponent s/move [min–max] |
|---|---|---:|---:|---:|---:|
| Gumbel-16 off | baseline | 90/200 | 45.0% [38.5–51.5%] | .0168 [.0140–.0188] | .1044 [.0896–.1164] |
| Gumbel-16 off | greedy | 101/200 | 50.5% [44.0–57.0%] | .0168 [.0136–.0198] | .0009 [.0007–.0012] |
| Gumbel-16 on | greedy | 64/200 | 32.0% [25.5–39.0%] | .0160 [.0133–.0187] | .0007 [.0006–.0008] |
| Gumbel-32 off | baseline | 96/200 | 48.0% [41.0–55.0%] | .0305 [.0263–.0341] | .0966 [.0849–.1086] |
| Gumbel-32 off | greedy | 101/200 | 50.5% [43.5–57.5%] | .0326 [.0260–.0375] | .0008 [.0007–.0009] |
| Gumbel-32 on | greedy | 75/200 | 37.5% [31.0–44.0%] | .0308 [.0268–.0375] | .0007 [.0006–.0009] |

The initial 18–22% collapse was not grounds for rejecting Gumbel itself.
The audit found inappropriate evaluation noise and a mismatched Q transform,
not a root-value sign error. Tests explicitly cover both players, mine extra
turns, safe/bomb passes, mixed-value arithmetic, normalization and illegal masks.
After correction, noise-off G16/G32 each score 50.5% against greedy.
Noise-on still loses significantly here: noisy top-k samples from the policy,
rather than restricting candidates to its deterministic highest-prior actions.
These data support disabling exploration noise for evaluation; they do not show
a statistically established strength improvement from Gumbel planning.

### Memory-safe persistent self-play worker scaling

48 complete games per timing, eight concurrent games per task (six tasks),
three interleaved repeats. Each CPU worker and parent has one Torch thread.
Pools warm before timing and close between configurations. Inference is capped
at 64 leaves; minimum available RAM is sampled before each inference.
Generation seeds are `2200000 + repeat*10000`; worker counts use identical seeds.
Dirichlet alpha=.15/fraction=.25, temperature 1 for 30 moves; no optimizer updates.

| Search | Workers (+1 trainer) | Positions/s median [min–max] | Speedup vs 2 workers | Minimum available RAM (MiB) |
|---|---:|---:|---:|---:|
| PUCT-100 K=1 | 2 | 13.24 [12.41–13.56] | 1.00× | 7134 |
| PUCT-100 K=1 | 4 | 20.44 [18.47–21.06] | 1.54× | 6554 |
| PUCT-100 K=1 | 6 | 31.43 [30.78–34.06] | 2.37× | 5889 |
| PUCT-100 K=4 | 2 | 34.68 [34.51–36.28] | 1.00× | 7610 |
| PUCT-100 K=4 | 4 | 48.21 [45.50–48.50] | 1.39× | 6948 |
| PUCT-100 K=4 | 6 | 63.53 [60.00–65.97] | 1.83× | 6266 |

### Superseded worker architecture: diagnostic timings

Earlier workers each owned a GPU model/context. This is not recommended on the
shared 15 GiB machine. Its Gumbel search also predates the transform correction.
The final six-worker repeat was interrupted; completed repeat counts are explicit.

| Search | Workers | Repeats | Positions/s median [min–max] |
|---|---:|---:|---:|
| Initial Gumbel-32 | 2 | 3 | 70.29 [51.47–71.43] |
| Initial Gumbel-32 | 4 | 3 | 72.52 [66.79–84.86] |
| Initial Gumbel-32 | 6 | 2 | 91.45 [90.43–92.46] |
| PUCT-100 K=1 | 2 | 3 | 18.58 [17.35–21.90] |
| PUCT-100 K=1 | 4 | 3 | 24.31 [23.88–26.31] |
| PUCT-100 K=1 | 6 | 2 | 28.61 [28.59–28.63] |

### Recommendations and safety

1. **Search-enabled play/arena:** PUCT-100, tree reuse, K=4. It scored 49.5%
   against the baseline at .0690 vs .1655 seconds/move: about 2.4× cheaper,
   without detected strength loss (not a formal non-inferiority guarantee).
   Interactive-position throughput improved 3.1×. K=8 is a lower-latency option,
   but K=4 had better parallel throughput and the strongest point estimate vs greedy.
   The greedy policy remains the cheapest option; a clear search strength gain
   over greedy has not been established by these intervals.
2. **Gumbel:** noise-off G16 is a useful fast option (.0168 vs baseline .1044
   seconds/move), but its 45% baseline win rate leaves meaningful strength
   uncertainty. It is selectable, not the default. No new registry category is
   recommended from this evidence alone; any later identity must distinguish
   its fixed budget, candidate cap, normalized transform and noise-off evaluation.
3. **Self-play:** retain PUCT visit targets, use K=4. These matches do not establish
   which target learns better; completed-Q targets are available but unpromoted.
   Six CPU-only workers were fastest here: `--workers 6 --cpu-share 7
   --leaf-batch-size 4 --concurrent-games 8 --games 48` reached 63.53 positions/s,
   vs 13.24 for two workers/K=1 (4.8×), with at least 6266 MiB sampled available RAM.
   Keep at least as many game batches as workers; six idle workers cannot speed
   up a single batch. Leave the running controlled trials unchanged.
4. **Across runs:** budget at most `floor(20/N)` CPU threads per run, including
   the trainer, after subtracting other workloads. Four workers plus one trainer
   per run consumes all 20 threads across four runs; six workers each would not
   fit that fair split. CPU feasibility is not a RAM guarantee: on 15 GiB, start
   conservatively and raise worker counts only while retaining at least 3 GiB
   available memory. The measured six-worker result is for one benchmark run
   alongside existing workloads, not four simultaneous six-worker runs.

The original benchmark had `timeout=0` (reported as no deadline), but exited
143/SIGTERM after 15489.48 seconds. This is not exit 137/SIGKILL. The sender could
not be identified: completed tool metadata had expired, and this agent had not
sent a termination signal before that exit. Six orphaned development-only workers
showed about 1.68 GiB RSS each; those workers and their development resource tracker
were stopped. Available RAM rose from 2969 to 10586 MiB. No main-checkout training
process was touched. This does not establish the cause of the separately reported
training-process death. The new worker architecture prevents Linux orphan survival
and avoids per-worker GPU contexts. Final scaling runs checked `free -m` before
pools, sampled at least 5889 MiB available across all configurations, and released
their pools between runs. After completion, `free -m` showed 9708 MiB available.

Raw block/repeat data and machine-readable summaries are preserved in the allowed
shared-checkpoint artifact directory:
`checkpoints/dev-search-perf-1794035211243760185/`.

### Delivery

Changed: `msflags/mcts.py`, `msflags/agents.py`, `msflags/arena.py`,
`msflags/compare.py`, `msflags/train.py`, `tests/test_search_modes.py`,
`tests/test_selfplay_workers.py`, `tests/test_retention.py`, and this README.
Registry categories, generation pins and web pages are unchanged.

Final verification: `uv run pytest -q` — **193 passed in 28.65 seconds**.
Coverage includes exact reservation reversal, K=1/sequential equality, public-only
reuse and hidden-layout invariance, Gumbel budget accounting and greedy sanity,
root value perspectives, normalized completed-Q arithmetic/masks, worker CPU
budgets, shared inference chunk ordering/error cleanup, and output-scoped retention.

Implementation commits on `search-perf`: `dd82f00`, `46f5cc3`, `5490743`.
The final report and measured K=4 defaults are committed together after verification.
