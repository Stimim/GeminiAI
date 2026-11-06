import numpy as np
import pytest

from msflags.game import CLAIMED, UNKNOWN, Config, Game


def layout(rows):
    return np.array([[ch == "*" for ch in r] for r in rows]).reshape(-1)


def test_mine_scores_and_keeps_turn_safe_passes_turn():
    cfg = Config(3, 3, 1)
    g = Game(cfg, layout(["*..", "...", "..."]))
    result = g.step(0)
    assert result.hit and result.opened == [0] and result.claimed == [0]
    assert g.board[0] == CLAIMED[0] and g.scores == [1, 0] and g.to_move == 0
    # 1 mine -> win_score 1, so player 0 has already won
    assert g.winner == 0
    with pytest.raises(ValueError):
        g.step(1)


def test_flood_fill_opens_zero_region_but_not_mines():
    cfg = Config(3, 4, 2)
    g = Game(cfg, layout(["...*", "....", "*..."]))
    result = g.step(1)  # zero cell floods
    opened = result.opened
    assert not result.hit
    assert g.to_move == 1
    assert g.board[3] == UNKNOWN and g.board[8] == UNKNOWN  # mines stay hidden
    assert sorted(opened) == sorted(np.flatnonzero(g.board != UNKNOWN).tolist())
    assert np.all(g.board[g.board != UNKNOWN] == g.counts[g.board != UNKNOWN])


def test_first_to_majority_wins():
    cfg = Config(2, 2, 3)  # win_score 2
    g = Game(cfg, layout(["**", "*."]))
    g.step(3)  # safe, turn passes
    g.step(0)
    assert g.winner is None
    g.step(1)
    assert g.winner == 1 and g.scores == [0, 2]


@pytest.mark.parametrize("centre, expected", [(0, 9), (3, 15), (24, 25)])
def test_bomb_clips_and_passes_turn(centre, expected):
    cfg = Config(7, 7, 49)
    g = Game(cfg, np.ones(49, dtype=bool))
    result = g.step(cfg.n + centre)
    assert result.kind == "bomb" and not result.hit
    assert len(result.claimed) == expected and result.opened == []
    assert g.scores == [expected, 0] and g.bombs == [0, 1]
    assert g.to_move == (0 if g.done else 1)


def test_bomb_winning_score_ends_game_without_passing():
    cfg = Config(3, 3, 3)
    g = Game(cfg, layout(["***", "...", "..."]))
    result = g.step(cfg.n + 4)
    assert result.claimed == [0, 1, 2]
    assert g.winner == 0 and g.to_move == 0
    assert not g.legal_mask().any()


def test_bomb_unavailable_or_fully_open_area_illegal():
    cfg = Config(7, 7, 7)
    g = Game(cfg, layout(["*******"] + ["......."] * 6))
    g.step(cfg.n + 48)  # zero flood reveals the bottom area, not mines
    assert g.bombs == [0, 1] and g.to_move == 1
    with pytest.raises(ValueError):
        g.step(cfg.n + 48)  # opponent still has bomb, but area is empty
    g.step(0)  # mine retains opponent's turn
    g.to_move = 0
    with pytest.raises(ValueError):
        g.step(cfg.n)  # own bomb already used


def test_bomb_zero_flood_extends_beyond_window_and_clones_availability():
    cfg = Config(7, 7, 7)
    g = Game(cfg, layout(["*******"] + ["......."] * 6))
    result = g.step(cfg.n + 48)
    assert 14 in result.opened  # outside clipped bottom-right window
    assert not result.claimed and len(result.opened) > 9
    assert g.clone().bombs == [0, 1]
    sampled = Game.determinize(g.public(), g.mines)
    assert sampled.bombs == [0, 1]


def test_default_config_always_gives_each_player_one_bomb():
    cfg = Config()
    game = Game.random(cfg, np.random.default_rng(3))
    assert cfg.bombs == 1
    assert game.bombs == [1, 1]
    assert game.legal_mask().shape == (2 * cfg.n,)
    assert game.legal_mask().all()
    with pytest.raises(TypeError):
        Config(bombs=0)


def test_bomb_availability_is_independent_across_public_states_and_clones():
    cfg = Config(7, 7, 7)
    game = Game(cfg, layout(["*******"] + ["......."] * 6))
    clone = game.clone()
    public = game.public()
    clone.step(cfg.n + 48)
    assert clone.bombs == [0, 1]
    assert game.bombs == public.bombs == [1, 1]
