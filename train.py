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
    "form_diff", "experience_diff", "surface_code", "best_of"
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
    identity = [c for c in ("date", "winner_name", "loser_name") if c in matches.columns]
    if len(identity) == 3:
        matches = matches.drop_duplicates(subset=identity, keep="last")
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

    elo, surface_elo, form, played = {}, {}, {}, {}
    names, ranks, ages = {}, {}, {}
    feature_rows, labels, row_dates = [], [], []

    def rating(pid):
        return elo.get(pid, 1500.0)

    def surface_rating(pid, surface):
        return surface_elo.get((pid, surface), 1500.0)

    def recent_form(pid):
        values = form.get(pid, [])
        return float(np.mean(values)) if values else 0.5

    for match in matches.itertuples(index=False):
        row = match._asdict()
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
            np.log1p(played.get(winner, 0)) - np.log1p(played.get(loser, 0)),
            SURFACE_CODE[surface],
            best_of,
        ]
        # The second row reverses player-relative differences only.
        reverse = [-value if i < 5 else value for i, value in enumerate(features)]
        feature_rows.extend([features, reverse])
        labels.extend([1, 0])
        row_dates.extend([row["date"], row["date"]])

        expected_winner = 1 / (1 + 10 ** ((rating(loser) - rating(winner)) / 400))
        expected_surface = 1 / (
            1 + 10 ** ((surface_rating(loser, surface) - surface_rating(winner, surface)) / 400)
        )
        level = str(row.get("tourney_level", "A"))
        k_factor = 40 if level == "G" else 32 if level in ("M", "F") else 24
        elo[winner] = rating(winner) + k_factor * (1 - expected_winner)
        elo[loser] = rating(loser) - k_factor * (1 - expected_winner)
        surface_elo[(winner, surface)] = surface_rating(winner, surface) + k_factor * (1 - expected_surface)
        surface_elo[(loser, surface)] = surface_rating(loser, surface) - k_factor * (1 - expected_surface)

        form.setdefault(winner, []).append(1.0)
        form.setdefault(loser, []).append(0.0)
        form[winner], form[loser] = form[winner][-10:], form[loser][-10:]
        played[winner] = played.get(winner, 0) + 1
        played[loser] = played.get(loser, 0) + 1
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
            "form10": recent_form(pid),
            "matches": int(played.get(pid, 0)),
            "rank": ranks.get(pid),
            "age": ages.get(pid),
        }

    X = pd.DataFrame(feature_rows, columns=FEATURES).replace([np.inf, -np.inf], 0).fillna(0)
    y = np.asarray(labels, dtype=np.int8)
    dates = pd.DatetimeIndex(row_dates)
    return X, y, dates, player_state, len(matches)

def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    files = download_matches()
    X, y, dates, players, match_count = build_training_data(files)

    cutoff = pd.Series(dates).quantile(0.80)
    train_mask, test_mask = dates < cutoff, dates >= cutoff
    if int(train_mask.sum()) < 500 or int(test_mask.sum()) < 100:
        raise RuntimeError("Not enough historical matches for a reliable train/test split.")

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
    xgb.fit(X.loc[train_mask], y[train_mask])
    logistic.fit(X.loc[train_mask], y[train_mask])

    probabilities = (
        0.70 * xgb.predict_proba(X.loc[test_mask])[:, 1]
        + 0.30 * logistic.predict_proba(X.loc[test_mask])[:, 1]
    )
    metrics = {
        "matches": int(match_count),
        "csv_files": len(files),
        "test_start": str(dates[test_mask].min().date()),
        "test_end": str(dates[test_mask].max().date()),
        "accuracy": float(accuracy_score(y[test_mask], probabilities >= 0.5)),
        "log_loss": float(log_loss(y[test_mask], np.clip(probabilities, 1e-6, 1 - 1e-6))),
        "brier_score": float(brier_score_loss(y[test_mask], probabilities)),
    }

    joblib.dump({"xgb": xgb, "logistic": logistic, "features": FEATURES},
                ARTIFACT_DIR / "model.joblib")
    (ARTIFACT_DIR / "player_state.json").write_text(
        json.dumps({"players": players}), encoding="utf-8"
    )
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics

if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
