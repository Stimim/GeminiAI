"""Native exact mine probabilities and batched consistent-layout sampling.

The C engine is compiled lazily at first import using gcc and cached beside its
source under a content-hashed name. No Python extension or extra dependency is
needed. Components are enumerated exactly up to ``node_limit``; on overflow,
48 randomized backtracking solutions give approximate statistics. Global mine
conditioning uses log-space dynamic programming, including interior binomials.

Like Analysis, probabilities on opened and claimed cells are zero (the array
is intended for selecting unknown cells). Samples include claimed mines. Exact
components retain all enumerated solutions, so sampling within a mine-count
bucket is exactly uniform rather than limited to a reservoir. Random streams
are reproducible but not identical to Analysis's NumPy stream.
"""
from __future__ import annotations

import ctypes
import copy
import hashlib
import operator
import os
from pathlib import Path
import subprocess
import tempfile
import weakref

import numpy as np

from .game import PublicState


def _load_library():
    source = Path(__file__).with_name("csrc") / "fastsolve.c"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    target = source.with_name(f"_fastsolve_{digest}.so")
    if not target.exists():
        fd, temporary = tempfile.mkstemp(prefix="_fastsolve_", suffix=".so", dir=source.parent)
        os.close(fd)
        try:
            subprocess.run(
                ["gcc", "-O3", "-march=native", "-shared", "-fPIC", str(source), "-o", temporary, "-lm"],
                check=True, capture_output=True, text=True,
            )
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    library = ctypes.CDLL(str(target))
    library.fs_create.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.c_char_p,
    ]
    library.fs_create.restype = ctypes.c_void_p
    library.fs_free.argtypes = [ctypes.c_void_p]
    library.fs_free.restype = None
    library.fs_sample_many.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64, ctypes.c_void_p]
    library.fs_sample_many.restype = ctypes.c_int
    return library


_LIB = _load_library()


def _seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))


class FastAnalysis:
    """Drop-in probability/sampling API with native enumeration and batching.

    Approximate fallback searches have a minimum budget of 10,000 nodes per
    attempt, so small enumeration limits can still yield consistent samples.
    Inconsistent constraints or mine counts raise ValueError.
    """

    def __init__(self, pub: PublicState, *, node_limit: int = 50_000,
                 rng: np.random.Generator | None = None):
        self.pub = pub
        self.rng = rng if rng is not None else np.random.default_rng()
        node_limit = operator.index(node_limit)
        if node_limit < 1 or node_limit > np.iinfo(np.uintp).max:
            raise ValueError("node_limit must be a positive size_t")
        cfg = pub.cfg
        if (not 0 < cfg.h <= np.iinfo(np.int32).max
                or not 0 < cfg.w <= np.iinfo(np.int32).max // cfg.h):
            raise ValueError("invalid board dimensions")
        if not 0 <= cfg.mines <= cfg.n or not 0 <= pub.remaining_mines <= cfg.n:
            raise ValueError("invalid mine count")
        raw = np.asarray(pub.board)
        if raw.shape != (cfg.n,) or not np.all((raw >= -1) & (raw <= 10)) or not np.all(raw == np.floor(raw)):
            raise ValueError("invalid public board")
        board = np.ascontiguousarray(raw, dtype=np.int8)
        self.probs = np.empty(cfg.n, dtype=np.float64)
        exact = ctypes.c_int()
        error = ctypes.create_string_buffer(256)
        handle = _LIB.fs_create(cfg.h, cfg.w, cfg.mines, pub.remaining_mines,
                                board.ctypes.data, node_limit, _seed(self.rng),
                                self.probs.ctypes.data, ctypes.byref(exact), error)
        if not handle:
            raise ValueError(error.value.decode())
        self._handle = handle
        self._finalizer = weakref.finalize(self, _LIB.fs_free, handle)
        self.exact = bool(exact.value)

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Return one consistent flat boolean mine layout."""
        return self.sample_many(1, rng)[0]

    def sample_many(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Return ``n`` consistent layouts; all sampling work is performed in C."""
        n = operator.index(n)
        if n < 0:
            raise ValueError("sample count must be nonnegative")
        result = np.empty((n, self.pub.cfg.n), dtype=np.bool_)
        if n and not _LIB.fs_sample_many(self._handle, n, _seed(rng if rng is not None else self.rng), result.ctypes.data):
            raise MemoryError("could not allocate native sampling workspace")
        return result


_LIB.fs_expert_scores.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.c_char_p,
]
_LIB.fs_expert_scores.restype = ctypes.c_int


def expert_scores_native(pub: PublicState, layouts: np.ndarray, probs: np.ndarray,
                         candidates, bomb_centres, bomb_expected, *,
                         alpha: float, cap: float, bomb_reserve: float,
                         rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Score sampled outcomes in C, preserving FastAnalysis's RNG consumption.

    A cloned stream supplies an upper bound of seeds. Only cache misses consume
    the original stream, in the same first-seen order as the Python evaluator.
    """
    cfg = pub.cfg
    board = np.ascontiguousarray(pub.board, dtype=np.int8)
    layouts = np.ascontiguousarray(layouts, dtype=np.bool_)
    probs = np.ascontiguousarray(probs, dtype=np.float64)
    candidates = np.ascontiguousarray(candidates, dtype=np.int32)
    centres = np.ascontiguousarray(bomb_centres, dtype=np.int32)
    expected = np.ascontiguousarray(bomb_expected, dtype=np.float64)
    scores = np.ascontiguousarray(pub.scores, dtype=np.int32)
    maximum = (len(candidates) + len(centres)) * len(layouts)
    seeds = copy.deepcopy(rng).integers(0, np.iinfo(np.uint64).max,
                                      size=maximum, dtype=np.uint64)
    actions = np.empty(len(candidates) + len(centres), dtype=np.int32)
    values = np.empty(len(actions), dtype=np.float64)
    count, used = ctypes.c_size_t(), ctypes.c_size_t()
    error = ctypes.create_string_buffer(256)
    ok = _LIB.fs_expert_scores(
        cfg.h, cfg.w, cfg.mines, board.ctypes.data, scores.ctypes.data,
        pub.to_move, layouts.ctypes.data, len(layouts), probs.ctypes.data,
        candidates.ctypes.data, len(candidates), centres.ctypes.data,
        expected.ctypes.data, len(centres), alpha, cap, bomb_reserve,
        seeds.ctypes.data, actions.ctypes.data, values.ctypes.data,
        ctypes.byref(count), ctypes.byref(used), error,
    )
    rng.integers(0, np.iinfo(np.uint64).max, size=used.value, dtype=np.uint64)
    if not ok:
        raise ValueError(error.value.decode())
    return actions[:count.value], values[:count.value]
