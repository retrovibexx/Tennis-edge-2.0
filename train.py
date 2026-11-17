from pathlib import Path
from datetime import datetime, timezone
import json
import requests
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from xgboost import XGBClassifier

# Historical match data: Jeff Sackmann's tennis_atp / tennis_wta datasets.
# Respect the source repository's license and attribution requirements.
ARCHIVE_URLS = [
    "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/{tour}/{tour}_matches_{year}.csv",
    "https://raw.githubusercontent.com/JeffSackmann/tennis_{tour}/master/{tour}_matches_{year}.csv",
]
DATA_DIR = Path("data")
ARTIFACT_DIR = Path("artifacts")
SURFACES = ("Hard", "Clay", "Grass", "Carpet")
SURFACE_CODE = {name: i for i, name in enumerate(SURFACES)}
FEATURES = [
    "elo_diff", "surface_elo_diff", "rank_advantage",
    "form_diff", "form5_diff", "experience_diff",
    "h2h_diff", "rest_days_diff", "workload30_diff",
    "serve_win_diff", "return_win_diff",
    "surface_code", "best_of"
]



def download_weekly_matches():
    """Download the weekly refreshed ATP/WTA match table from Hugging Face."""
    import io

    endpoint = "https://datasets-server.huggingface.co/parquet"
    response = requests.get(
        endpoint,
        params={"dataset": "minmaxxxer/atp-wta-matches-with-odds"},
        timeout=45,
    )
    response.raise_for_status()
    parquet_files = response.json().get("parquet_files", [])
    if not parquet_files:
        raise RuntimeError("Weekly tennis feed returned no Parquet files.")

    frames = []
    for item in parquet_files:
        url = item.get("url")
        if url:
            part = requests.get(url, timeout=90)
            part.raise_for_status()
            frames.append(pd.read_parquet(io.BytesIO(part.content)))
    if not frames:
        raise RuntimeError("Could not download the weekly tennis Parquet files.")

    latest = pd.concat(frames, ignore_index=True)
    if "status" in latest:
        latest = latest[latest["status"].astype(str).str.lower() == "final"]
    if "draw" in latest:
        latest = latest[latest["draw"].astype(str).str.lower() == "main"]

    required = {"tourney_date", "winner_id", "loser_id", "surface"}
    missing = required.difference(latest.columns)
    if missing:
        raise RuntimeError("Weekly feed missing columns: " + ", ".join(sorted(missing)))

    path = DATA_DIR / "weekly_atp_wta_matches.csv"
    latest.to_csv(path, index=False)
    return path


def download_matches():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "TennisEdgeAI/2.0 (historical research model)"})
    saved = []
    current_year = datetime.now(timezone.utc).year
    for tour in ("atp", "wta"):
        for year in range(2000, current_year + 1):
            path = DATA_DIR / f"{tour}_matches_{year}.csv"
            if path.exists() and path.stat().st_size > 100:
                saved.append(path)
                continue
            for template in ARCHIVE_URLS:
                url = template.format(tour=tour, year=year)
                try:
                    response = session.get(url, timeout=25)
                    if response.ok and "winner_id" in response.text[:5000]:
                        path.write_bytes(response.content)
                        saved.append(path)
                        break
                except requests.RequestException:
                    continue
    try:
        weekly_file = download_weekly_matches()
        saved.append(weekly_file)
        print("Added latest weekly ATP/WTA match results.")
    except Exception as exc:
        print(f"Weekly feed unavailable; continuing with archive data: {exc}")

    if not saved:
        raise RuntimeError(
            "Could not download historical or weekly tennis match data. "
            "Check network access and the upstream data sources, then reload."
        )
    return saved

def _number(value, default=np.nan):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default

def build_training_data(files):
    frames = []
    required = {"tourney_date", "winner_id", "loser_id", "surface"}
    for path in files:
        try:
            frame = pd.read_csv(path, low_memory=False)
            if required.issubset(frame.columns):
                frames.append(frame)
        except Exception:
            continue
    if not frames:
        raise RuntimeError("Downloaded files did not contain usable match records.")

    matches = pd.concat(frames, ignore_index=True)
    matches["date"] = pd.to_datetime(
        matches["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    matches = matches.dropna(subset=["date", "winner_id", "loser_id"])
    # Canonicalize player keys before deduplication so archive/feed duplicates
    # and harmless name-format differences do not fragment head-to-head records.
    if {"winner_name", "loser_name"}.issubset(matches.columns):
        matches["winner_id"] = matches["winner_name"].where(matches["winner_name"].notna(), matches["winner_id"]).astype(str).str.strip()
        matches["loser_id"] = matches["loser_name"].where(matches["loser_name"].notna(), matches["loser_id"]).astype(str).str.strip()
    else:
        matches["winner_id"] = matches["winner_id"].astype(str).str.strip()
        matches["loser_id"] = matches["loser_id"].astype(str).str.strip()
    matches["_winner_key"] = matches["winner_id"].str.casefold().str.replace(r"[^a-z0-9]", "", regex=True)
    matches["_loser_key"] = matches["loser_id"].str.casefold().str.replace(r"[^a-z0-9]", "", regex=True)
    matches = matches.drop_duplicates(subset=["date", "_winner_key", "_loser_key"], keep="last")
    # The weekly feed uses different player IDs than the archived Sackmann
    # CSVs. Use the consistent player-name fields as keys across both sources.
    if {"winner_name", "loser_name"}.issubset(matches.columns):
        matches["winner_id"] = matches["winner_name"].where(
            matches["winner_name"].notna(), matches["winner_id"]
        ).astype(str)
        matches["loser_id"] = matches["loser_name"].where(
            matches["loser_name"].notna(), matches["loser_id"]
        ).astype(str)
    matches["surface"] = matches["surface"].fillna("Hard").astype(str).str.title()
    matches = matches[matches["surface"].isin(SURFACES)].sort_values("date")

    elo, surface_elo, surface_matches, form, played = {}, {}, {}, {}, {}
    names, ranks, ages = {}, {}, {}
    last_played, match_dates, h2h = {}, {}, {}
    h2h_history = {}
    played_since_return = {}
    serve_totals, return_totals = {}, {}
    feature_rows, labels, row_dates = [], [], []

    def rating(pid):
        return elo.get(pid, 1500.0)

    def surface_rating(pid, surface):
        # Shrink small-sample surface ratings toward overall Elo.
        overall = rating(pid)
        raw = surface_elo.get((pid, surface), 1500.0)
        count = surface_matches.get((pid, surface), 0)
        reliability = count / (count + 20.0)
        return overall + reliability * (raw - overall)

    def form_multiplier(winner, loser):
        gap = recent_form(winner, 5) - recent_form(loser, 5)
        return float(np.clip(1.0 + 0.35 * gap, 0.80, 1.20))

    def h2h_score(player, opponent, current_date, surface=None):
        # Exponential two-year half-life; same-surface meetings receive a
        # modest boost. Four pseudo-meetings shrink tiny samples toward zero.
        records = h2h_history.get((player, opponent), [])
        weighted_wins = weighted_losses = 0.0
        for played_date, played_surface in records:
            age_days = max(0, (current_date - played_date).days)
            weight = 0.5 ** (age_days / 730.0)
            if surface and played_surface == surface:
                weight *= 1.25
            weighted_wins += weight
        reverse_records = h2h_history.get((opponent, player), [])
        for played_date, played_surface in reverse_records:
            age_days = max(0, (current_date - played_date).days)
            weight = 0.5 ** (age_days / 730.0)
            if surface and played_surface == surface:
                weight *= 1.25
            weighted_losses += weight
        return (weighted_wins - weighted_losses) / (weighted_wins + weighted_losses + 4.0)

    def recent_form(pid, count=10):
        values = form.get(pid, [])
        return float(np.mean(values[-count:])) if values else 0.5

    def rate_gap(pid, store):
        wins, total = store.get(pid, [0.0, 0.0])
        return wins / total if total else 0.5

    def days_since(pid, current_date):
        previous = last_played.get(pid)
        return min((current_date - previous).days, 30) if previous is not None else 7

    def workload(pid, current_date):
        return sum(1 for played_date in match_dates.get(pid, []) if 0 < (current_date - played_date).days <= 30)

    # Build every match's features from information available before that
    # calendar day. Apply that day's results only after all its rows are scored,
    # preventing same-day outcomes from leaking into later matches that day.
    for match_date, day_matches in matches.groupby("date", sort=True):
        day_updates = []
        for _, row in day_matches.iterrows():
            row = row.to_dict()
            winner, loser, surface = str(row["winner_id"]), str(row["loser_id"]), row["surface"]
            names[winner] = str(row.get("winner_name", winner))
            names[loser] = str(row.get("loser_name", loser))

            winner_rank = _number(row.get("winner_rank"))
            loser_rank = _number(row.get("loser_rank"))
            rank_advantage = (
                np.log1p(loser_rank) - np.log1p(winner_rank)
                if np.isfinite(winner_rank) and np.isfinite(loser_rank)
                and winner_rank > 0 and loser_rank > 0 else 0.0
            )
            best_of = _number(row.get("best_of"), 3)
            if not np.isfinite(best_of):
                best_of = 3

            features = [
                rating(winner) - rating(loser),
                surface_rating(winner, surface) - surface_rating(loser, surface),
                rank_advantage,
                recent_form(winner) - recent_form(loser),
                recent_form(winner, 5) - recent_form(loser, 5),
                np.log1p(played.get(winner, 0)) - np.log1p(played.get(loser, 0)),
                h2h_score(winner, loser, match_date, surface),
                days_since(loser, match_date) - days_since(winner, match_date),
                workload(loser, match_date) - workload(winner, match_date),
                rate_gap(winner, serve_totals) - rate_gap(loser, serve_totals),
                rate_gap(winner, return_totals) - rate_gap(loser, return_totals),
                SURFACE_CODE[surface],
                best_of,
            ]
            reverse = [-value if i < 11 else value for i, value in enumerate(features)]
            feature_rows.extend([features, reverse])
            labels.extend([1, 0])
            row_dates.extend([match_date, match_date])
            day_updates.append((row, winner, loser, surface))

        # Update ratings/form only after all matches on this date have features.
        for row, winner, loser, surface in day_updates:
            # Decay inactive ratings toward 1500 by 1.5% per idle week,
            # capped at 18%, when the player returns to competition.
            for pid in (winner, loser):
                previous = last_played.get(pid)
                if previous is not None:
                    idle_weeks = max(0, (match_date - previous).days // 7)
                    decay = min(idle_weeks * 0.015, 0.18)
                    if decay:
                        elo[pid] = 1500.0 + (rating(pid) - 1500.0) * (1.0 - decay)
                        for surf in SURFACES:
                            key = (pid, surf)
                            if key in surface_elo:
                                surface_elo[key] = 1500.0 + (surface_elo[key] - 1500.0) * (1.0 - decay)

            expected_winner = 1 / (1 + 10 ** ((rating(loser) - rating(winner)) / 400))
            expected_surface = 1 / (
                1 + 10 ** ((surface_rating(loser, surface) - surface_rating(winner, surface)) / 400)
            )
            # Start the comeback counter at zero on the first match after a
            # 21+ day absence. Players with no prior history get no layoff boost.
            for pid in (winner, loser):
                previous = last_played.get(pid)
                if previous is not None and (match_date - previous).days >= 21:
                    played_since_return[pid] = 0
                elif previous is None:
                    played_since_return[pid] = 10
            level = str(row.get("tourney_level", "A"))
            base_k = 40 if level == "G" else 32 if level in ("M", "F") else 24
            shared_k = base_k * form_multiplier(winner, loser)
            k_winner = shared_k * (1.50 if played_since_return.get(winner, 10) < 5 else 1.25 if played_since_return.get(winner, 10) < 10 else 1.0)
            k_loser = shared_k * (1.50 if played_since_return.get(loser, 10) < 5 else 1.25 if played_since_return.get(loser, 10) < 10 else 1.0)
            elo[winner] = rating(winner) + k_winner * (1 - expected_winner)
            elo[loser] = rating(loser) - k_loser * (1 - expected_winner)
            raw_winner = surface_elo.get((winner, surface), 1500.0)
            raw_loser = surface_elo.get((loser, surface), 1500.0)
            surface_elo[(winner, surface)] = raw_winner + k_winner * (1 - expected_surface)
            surface_elo[(loser, surface)] = raw_loser - k_loser * (1 - expected_surface)
            surface_matches[(winner, surface)] = surface_matches.get((winner, surface), 0) + 1
            surface_matches[(loser, surface)] = surface_matches.get((loser, surface), 0) + 1

            for pid in (winner, loser):
                played_since_return[pid] = played_since_return.get(pid, 10) + 1
            form.setdefault(winner, []).append(1.0)
            form.setdefault(loser, []).append(0.0)
            form[winner], form[loser] = form[winner][-10:], form[loser][-10:]
            played[winner] = played.get(winner, 0) + 1
            played[loser] = played.get(loser, 0) + 1
            h2h[(winner, loser)] = h2h.get((winner, loser), 0) + 1
            h2h_history.setdefault((winner, loser), []).append((match_date, surface))
            for pid in (winner, loser):
                last_played[pid] = match_date
                match_dates.setdefault(pid, []).append(match_date)
            # Historical match-box stats are accumulated only when present.
            for pid, prefix in ((winner, "w"), (loser, "l")):
                svpt = _number(row.get(f"{prefix}_svpt"))
                first_in = _number(row.get(f"{prefix}_1stIn"))
                first_won = _number(row.get(f"{prefix}_1stWon"))
                second_won = _number(row.get(f"{prefix}_2ndWon"))
                if np.isfinite(svpt) and svpt > 0 and np.isfinite(first_in) and np.isfinite(first_won) and np.isfinite(second_won):
                    rate = (first_won + second_won) / svpt
                    totals = serve_totals.setdefault(pid, [0.0, 0.0])
                    totals[0] += rate
                    totals[1] += 1
            for pid, prefix in ((winner, "w"), (loser, "l")):
                return_points_won = _number(row.get(f"{prefix}_return_points_won"))
                return_points_total = _number(row.get(f"{prefix}_return_points_total"))
                if np.isfinite(return_points_won) and np.isfinite(return_points_total) and return_points_total > 0:
                    totals = return_totals.setdefault(pid, [0.0, 0.0])
                    totals[0] += return_points_won / return_points_total
                    totals[1] += 1
            winner_rank = _number(row.get("winner_rank"))
            loser_rank = _number(row.get("loser_rank"))
            if np.isfinite(winner_rank):
                ranks[winner] = winner_rank
            if np.isfinite(loser_rank):
                ranks[loser] = loser_rank
            winner_age, loser_age = _number(row.get("winner_age")), _number(row.get("loser_age"))
            if np.isfinite(winner_age):
                ages[winner] = winner_age
            if np.isfinite(loser_age):
                ages[loser] = loser_age

    player_state = {}
    for pid, value in elo.items():
        player_state[pid] = {
            "name": names.get(pid, pid),
            "elo": float(value),
            "surface_elo": {s: float(surface_rating(pid, s)) for s in SURFACES},
            "surface_elo_matches": {s: int(surface_matches.get((pid, s), 0)) for s in SURFACES},
            "form10": recent_form(pid),
            "post_layoff_matches": int(played_since_return.get(pid, 0)),
            "form5": recent_form(pid, 5),
            "serve_win_rate": rate_gap(pid, serve_totals),
            "return_win_rate": rate_gap(pid, return_totals),
            "h2h": {opponent: int(wins) for (player, opponent), wins in h2h.items() if player == pid},
            "h2h_weighted": {
                opponent: float(h2h_score(pid, opponent, matches["date"].max(), None))
                for opponent in set(other for (player, other) in h2h_history if player == pid)
            },
            "h2h_weighted_surface": {
                opponent: {
                    surface: float(h2h_score(pid, opponent, matches["date"].max(), surface))
                    for surface in SURFACES
                }
                for opponent in set(other for (player, other) in h2h_history if player == pid)
            },
            "h2h_counts": {
                opponent: {
                    "wins": int(h2h.get((pid, opponent), 0)),
                    "losses": int(h2h.get((opponent, pid), 0)),
                }
                for opponent in set(other for pair in h2h_history for other in pair if pid in pair and other != pid)
            },
            "last_played": last_played.get(pid).isoformat() if pid in last_played else None,
            "workload30": int(sum(1 for d in match_dates.get(pid, []) if (matches["date"].max() - d).days <= 30)),
            "matches": int(played.get(pid, 0)),
            "rank": ranks.get(pid),
            "age": ages.get(pid),
        }

    X = pd.DataFrame(feature_rows, columns=FEATURES).replace([np.inf, -np.inf], 0).fillna(0)
    y = np.asarray(labels, dtype=np.int8)
    dates = pd.DatetimeIndex(row_dates)
    return X, y, dates, player_state, len(matches)

def _new_models():
    xgb = XGBClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=5,
        min_child_weight=8, objective="binary:logistic",
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.35, max_iter=2000, random_state=42),
    )
    return xgb, logistic


def _score(y_true, probabilities):
    probabilities = np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6)
    return {
        "accuracy": float(accuracy_score(y_true, probabilities >= 0.5)),
        "log_loss": float(log_loss(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    files = download_matches()
    X, y, dates, players, match_count = build_training_data(files)

    # Split by calendar date, never by individual rows. Both player-perspective
    # rows from a match stay together, and the final 15% of dates are untouched
    # until the model choice has been made on the earlier validation period.
    unique_dates = pd.DatetimeIndex(sorted(pd.unique(dates)))
    if len(unique_dates) < 20:
        raise RuntimeError("Not enough distinct match dates for chronological validation.")
    train_end = unique_dates[int(len(unique_dates) * 0.70)]
    validation_end = unique_dates[int(len(unique_dates) * 0.85)]
    train_mask = dates < train_end
    validation_mask = (dates >= train_end) & (dates < validation_end)
    test_mask = dates >= validation_end
    if min(int(train_mask.sum()), int(validation_mask.sum()), int(test_mask.sum())) < 100:
        raise RuntimeError("Not enough matches in chronological train/validation/test periods.")

    xgb, logistic = _new_models()
    xgb.fit(X.loc[train_mask], y[train_mask])
    logistic.fit(X.loc[train_mask], y[train_mask])

    xgb_val = xgb.predict_proba(X.loc[validation_mask])[:, 1]
    logistic_val = logistic.predict_proba(X.loc[validation_mask])[:, 1]

    # Select blend weight using validation data only. The final test period is
    # not consulted when choosing the blend.
    candidate_weights = np.linspace(0.0, 1.0, 21)
    validation_losses = []
    for weight in candidate_weights:
        blended = weight * xgb_val + (1.0 - weight) * logistic_val
        validation_losses.append(log_loss(y[validation_mask], np.clip(blended, 1e-6, 1 - 1e-6)))
    xgb_weight = float(candidate_weights[int(np.argmin(validation_losses))])
    validation_blend = xgb_weight * xgb_val + (1.0 - xgb_weight) * logistic_val
    # Fit a one-dimensional probability calibrator using validation predictions only.
    # The untouched test period remains reserved for final evaluation.
    validation_logit = np.log(
        np.clip(validation_blend, 1e-6, 1 - 1e-6)
        / (1 - np.clip(validation_blend, 1e-6, 1 - 1e-6))
    ).reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    calibrator.fit(validation_logit, y[validation_mask])

    # Refit on train + validation, then evaluate exactly once on the held-out
    # most-recent period.
    development_mask = dates < validation_end
    final_xgb, final_logistic = _new_models()
    final_xgb.fit(X.loc[development_mask], y[development_mask])
    final_logistic.fit(X.loc[development_mask], y[development_mask])
    xgb_test = final_xgb.predict_proba(X.loc[test_mask])[:, 1]
    logistic_test = final_logistic.predict_proba(X.loc[test_mask])[:, 1]
    blended_test = xgb_weight * xgb_test + (1.0 - xgb_weight) * logistic_test
    blended_test_logit = np.log(
        np.clip(blended_test, 1e-6, 1 - 1e-6)
        / (1 - np.clip(blended_test, 1e-6, 1 - 1e-6))
    ).reshape(-1, 1)
    calibrated_test = calibrator.predict_proba(blended_test_logit)[:, 1]

    metrics = {
        "evaluation_method": "Chronological 70% train / 15% validation / 15% untouched test by calendar date",
        "matches": int(match_count),
        "csv_files": len(files),
        "train_start": str(dates[train_mask].min().date()),
        "train_end": str(dates[train_mask].max().date()),
        "validation_start": str(dates[validation_mask].min().date()),
        "validation_end": str(dates[validation_mask].max().date()),
        "test_start": str(dates[test_mask].min().date()),
        "test_end": str(dates[test_mask].max().date()),
        "test_rows": int(test_mask.sum()),
        "xgboost_test": _score(y[test_mask], xgb_test),
        "logistic_test": _score(y[test_mask], logistic_test),
        "selected_blend_test": _score(y[test_mask], blended_test),
        "calibrated_blend_test": _score(y[test_mask], calibrated_test),
        "selected_xgboost_weight": xgb_weight,
        "validation_selected_weight_only": True,
    }

    joblib.dump({
        "xgb": final_xgb,
        "logistic": final_logistic,
        "xgb_weight": xgb_weight,
        "calibrator": calibrator,
        "features": FEATURES,
        "calibration_method": "1D logistic calibration fitted on validation predictions",
    }, ARTIFACT_DIR / "model.joblib")
    (ARTIFACT_DIR / "player_state.json").write_text(
        json.dumps({"players": players}), encoding="utf-8"
    )
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics

if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
