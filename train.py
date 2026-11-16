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

# Jeff Sackmann historical ATP/WTA archive mirror; original data attribution/license retained.
ARCHIVE = "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/{tour}/{tour}_matches_{year}.csv"
DATA = Path("data")
OUT = Path("artifacts")
SURFACES = ["Hard", "Clay", "Grass", "Carpet"]
SC = {s: i for i, s in enumerate(SURFACES)}
FEATURES = ["elo_diff", "surface_elo_diff", "rank_advantage",
            "form_diff", "experience_diff", "surface_code", "best_of"]

def download_matches():
    DATA.mkdir(exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "TennisEdgeAI/1.0"
    files = []
    current_year = datetime.now(timezone.utc).year
    for tour in ("atp", "wta"):
        for year in range(2000, current_year + 1):
            path = DATA / f"{tour}_matches_{year}.csv"
            if not path.exists() or path.stat().st_size < 100:
                url = ARCHIVE.format(tour=tour, year=year)
                try:
                    response = session.get(url, timeout=30)
                    if response.status_code == 200 and "winner_id" in response.text[:3000]:
                        path.write_bytes(response.content)
                    else:
                        continue
                except requests.RequestException as exc:
                    print(f"Skipping {tour} {year}: {exc}")
                    continue
            files.append(path)
    if not files:
        raise RuntimeError("No historical CSVs downloaded. Check internet access and archive availability.")
    print(f"Loaded {len(files)} yearly CSV files.")
    return files

def num(value, default=np.nan):
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default

def prepare(files):
    frames = []
    for file in files:
        try:
            d = pd.read_csv(file, low_memory=False)
            if {"tourney_date", "winner_id", "loser_id", "surface"}.issubset(d.columns):
                frames.append(d)
        except Exception as e:
            print("Skipping unreadable file", file, e)
    if not frames:
        raise RuntimeError("No usable Sackmann match files found.")
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date", "winner_id", "loser_id"]).copy()
    df["surface"] = df["surface"].fillna("Hard").astype(str).str.title()
    df = df[df.surface.isin(SURFACES)].sort_values("date").reset_index(drop=True)

    elo, selo, form, played = {}, {}, {}, {}
    names, ranks, ages = {}, {}, {}
    X, y, dates = [], [], []

    def rating(pid): return elo.get(pid, 1500.0)
    def surface_rating(pid, s): return selo.get((pid, s), 1500.0)
    def recent_form(pid):
        history = form.get(pid, [])
        return float(np.mean(history)) if history else 0.5

    for _, r in df.iterrows():
        w, l, s = str(r.winner_id), str(r.loser_id), r.surface
        names[w], names[l] = str(r.get("winner_name", w)), str(r.get("loser_name", l))
        wr, lr = num(r.get("winner_rank")), num(r.get("loser_rank"))
        rank_adv = np.log1p(lr) - np.log1p(wr) if np.isfinite(wr) and np.isfinite(lr) and wr > 0 and lr > 0 else 0.0
        bo = num(r.get("best_of"), 3)
        if not np.isfinite(bo): bo = 3
        f = [rating(w)-rating(l),
             surface_rating(w,s)-surface_rating(l,s),
             rank_adv,
             recent_form(w)-recent_form(l),
             np.log1p(played.get(w,0))-np.log1p(played.get(l,0)),
             SC[s], bo]
        X.extend([f, [-v if i != 5 else v for i,v in enumerate(f)]])
        y.extend([1,0])
        dates.extend([r.date,r.date])

        ew = 1/(1+10**((rating(l)-rating(w))/400))
        es = 1/(1+10**((surface_rating(l,s)-surface_rating(w,s))/400))
        level = str(r.get("tourney_level","A"))
        k = 40 if level == "G" else 32 if level in ("M","F") else 24
        elo[w] = rating(w)+k*(1-ew)
        elo[l] = rating(l)-k*(1-ew)
        selo[(w,s)] = surface_rating(w,s)+k*(1-es)
        selo[(l,s)] = surface_rating(l,s)-k*(1-es)
        form.setdefault(w,[]).append(1.0); form.setdefault(l,[]).append(0.0)
        form[w] = form[w][-10:]; form[l] = form[l][-10:]
        played[w] = played.get(w,0)+1; played[l] = played.get(l,0)+1
        if np.isfinite(wr): ranks[w] = wr
        if np.isfinite(lr): ranks[l] = lr
        wa, la = num(r.get("winner_age")), num(r.get("loser_age"))
        if np.isfinite(wa): ages[w] = wa
        if np.isfinite(la): ages[l] = la

    players = {}
    for pid in elo:
        players[pid] = {
            "name": names.get(pid,pid), "elo": float(elo[pid]),
            "surface_elo": {s: float(surface_rating(pid,s)) for s in SURFACES},
            "form10": recent_form(pid), "matches": int(played.get(pid,0)),
            "rank": ranks.get(pid), "age": ages.get(pid)
        }
    return pd.DataFrame(X,columns=FEATURES).replace([np.inf,-np.inf],0).fillna(0), np.array(y), pd.to_datetime(dates), players, len(df)

def main():
    OUT.mkdir(exist_ok=True)
    files = download_matches()
    X,y,dates,players,n_matches = prepare(files)
    cutoff = dates.quantile(.80)
    tr,te = dates < cutoff, dates >= cutoff
    if tr.sum() < 500 or te.sum() < 100:
        raise RuntimeError("Not enough data for chronological training and testing.")
    xgb = XGBClassifier(n_estimators=400,max_depth=3,learning_rate=.03,
        subsample=.85,colsample_bytree=.85,reg_lambda=5,min_child_weight=8,
        objective="binary:logistic",eval_metric="logloss",random_state=42,n_jobs=-1)
    lr = make_pipeline(StandardScaler(),LogisticRegression(C=.35,max_iter=2000))
    xgb.fit(X.loc[tr],y[tr]); lr.fit(X.loc[tr],y[tr])
    p = .70*xgb.predict_proba(X.loc[te])[:,1]+.30*lr.predict_proba(X.loc[te])[:,1]
    metrics = {"matches":int(n_matches),"csv_files":len(files),
        "test_start":str(dates[te].min().date()),"test_end":str(dates[te].max().date()),
        "accuracy":float(accuracy_score(y[te],p>=.5)),
        "log_loss":float(log_loss(y[te],np.clip(p,1e-6,1-1e-6))),
        "brier_score":float(brier_score_loss(y[te],p))}
    joblib.dump({"xgb":xgb,"logistic":lr,"features":FEATURES},OUT/"model.joblib")
    (OUT/"player_state.json").write_text(json.dumps({"players":players}))
    (OUT/"metrics.json").write_text(json.dumps(metrics,indent=2))
    print("TRAINING COMPLETE"); print(json.dumps(metrics,indent=2))
    print("Player records:",len(players))

if __name__ == "__main__":
    main()
