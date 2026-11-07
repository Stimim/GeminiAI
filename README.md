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
plus `(50 + max_visits) * completed_Q`; unvisited actions use the prior-weighted
visited-value / network-value mixture from [Danihelka et al. (ICLR 2022)](https://iclr.cc/virtual/2022/poster/6418).
Non-root selection remains PUCT: it already averages sampled chance outcomes
and preserves the existing player-relative backup/extra-turn handling, isolating
the root-allocation change rather than changing two selection algorithms at once.
Open and bomb actions share the same masked action distribution.

`next_leaves(K)` / `provide_batch(policies, values)` reserve independent pending
leaves with virtual visits and loss. Reservations live in separate counters,
so completion reverts virtual loss exactly without modifying real sums.
Duplicate unexpanded public outcomes end a batch; root expansion and halving
boundaries are evaluation barriers. `K=1` preserves sequential PUCT behavior.
`MCTSAgent(..., leaf_batch_size=4, tree_reuse=True)` controls play batching;
self-play and compare accept `--leaf-batch-size`, `--search` (compare: `--mode`)
and `--gumbel-k`. Self-play stores the completed-Q target for Gumbel, visit targets
for PUCT. Registry categories and published web identities are unchanged.

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

### Phase B: paired value/LR trials

The controlled comparisons are medium `run-medium` versus `run-medium-vw1` and small `run-small` versus `run-small-lr2e4`. Each fork starts from **exactly the control's published weights, optimizer, replay and iteration**; its immutable starting copy is `trial-start.pt`. `checkpoints/phase-b-trials.json` records the starting/target iterations, SHA-256 checksums, launch arguments and process-group PIDs. `train.pid` in each directory is the current process group. No generations are promoted merely for starting a trial.

| Directory under `checkpoints/` | Process group | Start → target | Learning rate | Value weight |
|---|---:|---:|---:|---:|
| `run-medium` (control) | 1757910 | 98 → 113 | 0.0002 | 0.25 |
| `run-medium-vw1` | 1757911 | 98 → 113 | 0.0002 | 1.0 |
| `run-small` (control) | 1757956 | 67 → 82 | 0.001 | 0.25 |
| `run-small-lr2e4` | 1757957 | 67 → 82 | 0.0002 | 0.25 |

All four receive equal resources: **one single-process self-play/trainer, one Torch thread, 16 concurrent games** per run. This replaces the old two spawned workers per control, keeping four aggregate search processes instead of eight workers plus four trainers. Controls remain active, not time-sliced. All retain 96 games, 100 simulations, c_puct 1.5, FPU reduction 0.2, Dirichlet alpha 0.15/fraction 0.25, temperature 1 for 30 moves, 300 updates ×128, 50k replay, mine weight 2 and the pair's existing seed. Targets are **15 additional completed iterations**, expressed as absolute `--iterations` values in the manifest. Runs stop automatically at their own targets; do not silently continue one arm farther than its mate.

All four launched detached after the old controls exited gracefully. At the initial four-arm resource snapshot, GPU allocation was **1,645 / 8,188 MiB**, host RAM had **8,397 MiB available**, and swap use was 54 MiB. These are instantaneous startup observations, not peak-memory or throughput guarantees; no control pausing was needed.

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

