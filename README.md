# Minesweeper Flags

A two-player Minesweeper game and AI playground. Players alternate opening cells: claiming a mine scores a point and grants another move; a safe cell reveals its adjacent-mine count (zeros flood-fill) and passes the turn. Each player also has one 5×5 bomb: it claims unopened mines, reveals safe cells, and always ends the turn. First to a majority of the mines wins. The default board is 16×16 with 51 mines, first to 26.

## Architecture

- **Exact constraint solver:** derives mine probabilities from the visible clues and enumerates consistent assignments.
- **ResNet evaluator:** fully convolutional mine, open-policy, bomb-policy, and value heads, using 20 public-information input planes. Open policy is `beta * mine_logit + residual(features)`; the bomb head starts from clipped 5×5 expected mine counts. Open and bomb actions share a masked softmax over `2 × board cells`.
- **Information-set MCTS:** searches from public information, sampling consistent determinizations with the constraint solver.

## Setup

```sh
uv sync
```

## Training

```sh
uv run python -m msflags.train bootstrap --games 1800 --steps 8000 --workers 12
uv run python -m msflags.train selfplay --iterations 2 --games 32 --sims 100 --steps 300 --lr 0.0001 --eval-games 40
uv run python -m msflags.train arena --agent-a r64x4-tr-bs8k.mcts100-c1.5 --agent-b solver --games 200
```

Check subcommand options with `uv run python -m msflags.train bootstrap --help`, `uv run python -m msflags.train selfplay --help`, or `uv run python -m msflags.train arena --help`.

Train from scratch or continue a supported 20-plane checkpoint:

```sh
uv run python -m msflags.train bootstrap --checkpoint checkpoints/bomb/bootstrap-1793979495800377617.pt --output checkpoints/bomb --games 300 --steps 8000 --workers 12 --device cuda
```

Bomb solver baseline: open any certain mine first; otherwise detonate the highest expected-mine window when its expectation is at least three (and exceeds the best single opening). A bomb gives up the usual extra turn, so the threshold trades immediate multi-mine gain against saving the one-use resource. Bootstrap explores all legal actions and stores a joint open/bomb policy; uncertain solver fallback positions are excluded from exact labels.

A quick alternating-seat sweep (40 games each, seeds 81000–81039) gave thresholds two and four each 20/40 wins against threshold three (6.3 seconds total). This small experiment does not establish an optimum; three remains the conservative midpoint baseline.

### Results

| Agent/model | Result |
|---|---:|
| Pinned network vs solver (200 games) | 60.0% |
| MCTS-100 vs solver (100 games) | 61.0% |

The pinned `r64x4-tr-bs8k.greedy` checkpoint historically started from transferred weights, then used 300 explored solver games (16,542 exact-labelled positions) and 8,000 bootstrap steps at learning rate 0.0001, completing in 112.2 seconds on CUDA. It won 120/200 against the solver: **60.0%** (95% CI 53.1–66.5), in 8.3 seconds. The `tr` identity preserves that historical initialization; weight transfer is not a supported command.

Bomb MCTS-100 won 61/100 against the solver: **61.0%** (95% CI 51.2–70.0), in 206.9 seconds. The 20-games-per-pair bomb tournament took 121.8 seconds: all three nonrandom agents beat random 20–0; solver beat greedy 13–7; MCTS beat solver 13–7; greedy beat MCTS 14–6. The resulting ledger leaderboard (including one existing solver–random game) is greedy **1014.4 ±123.6**, solver **1000**, MCTS **991.4 ±123.5**, random **52.4 ±621.1** Elo. Small pair counts explain why this ordering differs from the larger head-to-head evaluations.

## Web UI

Run the server with `uv run python -m msflags.server` and open http://127.0.0.1:8000. Neural catalog entries use pinned timestamped checkpoints; random and solver need no weights. Use `--host`, `--port`, and `--device cpu|cuda` to configure it. The interface supports human-vs-AI, AI-vs-AI, and hot-seat games, with hints and move-by-move animation.

### AI catalog and arena

Use **Play** (`/`), **AIs** (`/agents.html`), and **Arena** (`/arena.html`). The catalog explains each strategy and reads network shape and training provenance from real checkpoints and their `metrics.jsonl`. Evaluators are loaded lazily and shared by checkpoint. Neural IDs are derived as `{arch}-{training}.{search}`: `r{channels}x{blocks}`, `bs{bootstrap steps in k}` plus `-sp{completed self-play iterations}` when applicable, and `greedy` or `mcts{sims}-c{c_puct}` (with `-f{fpu}` only for nondefault FPU reduction, default 0.2). Missing training provenance and duplicate derived IDs raise errors rather than guessing. The registered neural entries are `r64x4-tr-bs8k.greedy` and `r64x4-tr-bs8k.mcts100-c1.5`. Play defaults to the MCTS variant: its sims box overrides that entry’s configured default only. Arena always uses the listed fixed settings.

Models historically initialized from transferred weights retain `-tr` before the bootstrap identity, for example `r64x4-tr-bs8k.greedy`. Only 20-plane checkpoints with joint open/bomb policies are supported.


Arena games alternate the starting seat and append to `checkpoints/arena/games.jsonl`. One match runs at a time per server process; live boards expose only public information. Match progress is in memory; completed games survive restarts. Omit the API seed for fresh random games, or supply a nonnegative seed to reproduce a match. The offline round-robin batches up to 32 games for efficient neural inference; the web arena runs one game at a time for a coherent live view. Ratings reread the ledger, so newly completed CLI games appear without restarting the server. Run one server process when using the arena.

#### Round-robin CLI

`--games` is the number of games **per pair**. By default, every available registry entry participates; four agents produce six pairs, so `--games 20` records 120 games. Use `--agents r64x4-tr-bs8k.greedy solver r64x4-tr-bs8k.mcts100-c1.5` to select contenders, and `--seed 200000` for a reproducible tournament. The `ratings` command refits all persisted results without playing additional games. The training CLI's `arena` command also accepts catalog IDs; `--checkpoint` / `--checkpoint-b` still override the corresponding neural weights for unrecorded evaluations, with `--sims`, `--c-puct`, and `--fpu-reduction` controlling overridden search.

```sh
uv run python -m msflags.arena roundrobin --games 20 --device cuda
uv run python -m msflags.arena ratings
```

#### Ratings method

Ratings fit all recorded games jointly with a Bradley–Terry logistic model, fixing the greedy solver at **1000 Elo**. The win probability is `1 / (1 + 10 ** ((rating_b - rating_a) / 400))`. The fit maximizes the likelihood with a Gaussian regularizer centered at 1000, standard deviation **800 Elo**, for every non-anchor agent, keeping perfect win/loss records finite. The leaderboard reports games played and approximate 95% uncertainty half-widths (`1.96 × standard error`) from the inverse fitted Hessian, conditional on the fixed anchor; these are not guarantees of future performance. Random is rated, not used as the anchor.


API: `GET /api/agents`, `GET /api/agents/{id}`, `POST /api/arena/matches` (`{a,b,games,seed}`), `GET /api/arena/matches/{id}` (including `current` live board), and `GET /api/ratings`. Unknown/unavailable agents and invalid counts are rejected; a second active match returns HTTP 409. Existing game, move, AI and hint routes accept catalog IDs (plus `human` for game seats).

#### Adding a checkpoint or agent

1. Train to a new timestamped checkpoint, such as `checkpoints/my-run/bootstrap-<timestamp>.pt`; keep the run's `metrics.jsonl` alongside it. Shape comes from `net_cfg`; completed training comes from checkpoint metadata (`bootstrap_steps`, `selfplay_iterations`) or completed metrics, not requested iteration counts.
2. Add a `(checkpoint, search)` line to `VARIANTS` in `msflags/registry.py`. Use `None` for greedy policy, or search settings for MCTS:

   ```python
   ('checkpoints/my-run/bootstrap-<timestamp>.pt', None),
   ('checkpoints/my-run/bootstrap-<timestamp>.pt',
    dict(sims=100, c_puct=1.5, fpu_reduction=0.2)),
   ```

3. IDs, display names, summaries, and explanations are generated automatically. Restart the server, open `/agents.html`, and run `uv run python -m msflags.arena roundrobin --agents <derived-id> solver --games 20 --device cuda`.
4. Never repoint an entry to different weights or use `latest.pt` / newest-file globs: persisted results are keyed by derived ID. If a new checkpoint derives an existing ID, do not register it alongside the existing variant or overwrite its weights; use a genuinely distinct architecture/training/search configuration.

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

