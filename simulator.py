import numpy as np


def _rate(value, fallback, low, high):
    try:
        number = float(value)
        return number if np.isfinite(number) and low <= number <= high else fallback
    except (TypeError, ValueError):
        return fallback


def simulate_tennis_matches(player_a, player_b, surface_name, match_format,
                            simulations=5000, seed=None):
    """Monte Carlo best-of-3/5 simulator with point, game, tiebreak and set scoring."""
    rng = np.random.default_rng(seed)
    serve_a = _rate(player_a.get("serve_win_rate"), 0.62, 0.54, 0.82)
    serve_b = _rate(player_b.get("serve_win_rate"), 0.62, 0.54, 0.82)
    return_a = _rate(player_a.get("return_win_rate"), 0.38, 0.20, 0.47)
    return_b = _rate(player_b.get("return_win_rate"), 0.38, 0.20, 0.47)

    # Estimate point-win chance on each player's serve from both players'
    # observed serve and return profiles, then apply a restrained rating adjustment.
    p_a_serve = 0.65 * serve_a + 0.35 * (1.0 - return_b)
    p_b_serve = 0.65 * serve_b + 0.35 * (1.0 - return_a)
    elo_delta = (
        float(player_a.get("elo", 1500)) - float(player_b.get("elo", 1500))
        + float(player_a.get("surface_elo", {}).get(surface_name, 1500))
        - float(player_b.get("surface_elo", {}).get(surface_name, 1500))
    )
    adjustment = float(np.clip(elo_delta / 4000.0, -0.035, 0.035))
    p_a_on_a_serve = float(np.clip(p_a_serve + adjustment, 0.52, 0.82))
    p_a_on_b_serve = float(np.clip(1.0 - p_b_serve + adjustment, 0.18, 0.48))

    n = int(simulations)
    sets_needed = int(match_format) // 2 + 1
    sets_a = np.zeros(n, dtype=np.int8)
    sets_b = np.zeros(n, dtype=np.int8)
    games_a = np.zeros(n, dtype=np.int8)
    games_b = np.zeros(n, dtype=np.int8)
    next_server_a = rng.integers(0, 2, n).astype(bool)
    set_scores = [[] for _ in range(n)]

    while np.any((sets_a < sets_needed) & (sets_b < sets_needed)):
        active = (sets_a < sets_needed) & (sets_b < sets_needed)
        tied_for_tiebreak = active & (games_a == 6) & (games_b == 6)

        # Play a standard game for every active match not at 6-6.
        regular = active & ~tied_for_tiebreak
        if np.any(regular):
            points_a = np.zeros(n, dtype=np.int16)
            points_b = np.zeros(n, dtype=np.int16)
            game_over = ~regular.copy()
            while np.any(~game_over):
                serving_a = next_server_a
                p = np.where(serving_a, p_a_on_a_serve, p_a_on_b_serve)
                won = rng.random(n) < p
                points_a += ((~game_over) & won).astype(np.int16)
                points_b += ((~game_over) & ~won).astype(np.int16)
                a_wins = (points_a >= 4) & ((points_a - points_b) >= 2)
                b_wins = (points_b >= 4) & ((points_b - points_a) >= 2)
                finished = (~game_over) & (a_wins | b_wins)
                games_a += (finished & a_wins).astype(np.int8)
                games_b += (finished & b_wins).astype(np.int8)
                next_server_a[finished] = ~next_server_a[finished]
                game_over |= finished

        # A 6-6 set uses a 7-point tiebreak, win by two. The first server
        # serves one point; service then alternates in two-point blocks.
        if np.any(tied_for_tiebreak):
            first_server_a = next_server_a.copy()
            tb_a = np.zeros(n, dtype=np.int16)
            tb_b = np.zeros(n, dtype=np.int16)
            tb_done = ~tied_for_tiebreak.copy()
            point_number = np.zeros(n, dtype=np.int16)
            while np.any(~tb_done):
                # Explicit sequence: first server for point 1, then other server
                # for points 2-3, first for 4-5, and so on.
                switch = np.where(point_number == 0, False, ((point_number + 1) // 2) % 2 == 1)
                serving_a = np.logical_xor(first_server_a, switch)
                p = np.where(serving_a, p_a_on_a_serve, p_a_on_b_serve)
                won = rng.random(n) < p
                tb_a += ((~tb_done) & won).astype(np.int16)
                tb_b += ((~tb_done) & ~won).astype(np.int16)
                point_number += (~tb_done).astype(np.int16)
                a_wins = (tb_a >= 7) & ((tb_a - tb_b) >= 2)
                b_wins = (tb_b >= 7) & ((tb_b - tb_a) >= 2)
                finished = (~tb_done) & (a_wins | b_wins)
                games_a += (finished & a_wins).astype(np.int8)
                games_b += (finished & b_wins).astype(np.int8)
                # Receiver of the first tiebreak point serves the next set.
                next_server_a[finished] = ~first_server_a[finished]
                tb_done |= finished

        # Close sets and preserve the actual game score, not just set totals.
        set_a = active & (((games_a >= 6) & ((games_a - games_b) >= 2)) | ((games_a == 7) & (games_b == 6)))
        set_b = active & (((games_b >= 6) & ((games_b - games_a) >= 2)) | ((games_b == 7) & (games_a == 6)))
        finished_set = set_a | set_b
        for idx in np.flatnonzero(finished_set):
            set_scores[idx].append(f"{int(games_a[idx])}-{int(games_b[idx])}")
        sets_a += set_a.astype(np.int8)
        sets_b += set_b.astype(np.int8)
        games_a[finished_set] = 0
        games_b[finished_set] = 0

    won_a = sets_a > sets_b
    scorelines = [" ".join(scores) for scores in set_scores]
    unique, counts = np.unique(scorelines, return_counts=True)
    distribution = sorted(
        [{"score": str(score), "count": int(count), "rate": float(count / n)}
         for score, count in zip(unique, counts)],
        key=lambda item: item["count"], reverse=True
    )
    return {
        "wins_a": int(won_a.sum()),
        "wins_b": int((~won_a).sum()),
        "probability_a": float(won_a.mean()),
        "probability_b": float((~won_a).mean()),
        "score_distribution": distribution,
        "serve_probability_a": p_a_on_a_serve,
        "serve_probability_b": float(1.0 - p_a_on_b_serve),
        "assumptions": "Standard advantage games; 7-point tiebreak at 6-6 in every set; randomized first server.",
    }
