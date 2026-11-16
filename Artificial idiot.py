from pathlib import Path
import json
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ART = Path("artifacts")
FEATURES = [
    "elo_diff", "surface_elo_diff", "rank_advantage",
    "form_diff", "experience_diff", "surface_code", "best_of"
]
SURFACE_CODE = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}

st.set_page_config(page_title="Tennis Edge AI", page_icon="🎾", layout="wide")
st.title("🎾 Tennis Edge AI")
st.caption("Jeff Sackmann historical ATP/WTA data • Match-winner probabilities")

model_path = ART / "model.joblib"
state_path = ART / "player_state.json"

def model_is_ready():
    return model_path.exists() and state_path.exists()

def train_model():
    train_script = Path("train.py")
    if not train_script.exists():
        st.error("Cannot find train.py in the app repository. Make sure train.py is in the same folder as app.py.")
        st.stop()

    status = st.status("Preparing your tennis model...", expanded=True)
    status.write("Downloading historical ATP/WTA match data and training the models. This can take several minutes.")
    log_box = st.empty()
    try:
        process = subprocess.Popen(
            [sys.executable, "-u", str(train_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_lines = []
        for line in process.stdout:
            output_lines.append(line.rstrip())
            log_box.code("\n".join(output_lines[-30:]))
        return_code = process.wait()
        if return_code != 0:
            status.update(label="Model training failed", state="error")
            st.error("Training did not finish. Review the log above for the exact error.")
            st.stop()
        if not model_is_ready():
            status.update(label="Training finished but model files are missing", state="error")
            st.error("train.py ended without creating artifacts/model.joblib and artifacts/player_state.json.")
            st.stop()
        status.update(label="Model trained successfully", state="complete")
        st.success("Training complete. Loading your tennis model...")
        st.rerun()
    except Exception as exc:
        status.update(label="Model training failed", state="error")
        st.exception(exc)
        st.stop()

if not model_is_ready():
    st.warning("No trained model is installed yet. Tennis Edge AI is starting the training process automatically.")
    st.info("The first training run downloads historical match results and may take several minutes. Keep this page open.")
    train_model()

try:
    bundle = joblib.load(model_path)
    if bundle.get("features") != FEATURES:
        st.error("The saved model uses different features. Remove the old artifacts folder and retrain.")
        st.stop()
    players = json.loads(state_path.read_text())["players"]
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

name_ids = {}
for pid, player_data in players.items():
    name_ids.setdefault(player_data["name"], []).append(pid)
names = sorted(name_ids)
if len(names) < 2:
    st.error("Not enough player records in the trained model.")
    st.stop()

st.sidebar.header("Match setup")
surface = st.sidebar.selectbox("Surface", list(SURFACE_CODE))
best_of = st.sidebar.selectbox("Match format", [3, 5])
a_name = st.sidebar.selectbox("Player A", names, index=0)
b_name = st.sidebar.selectbox("Player B", names, index=1)
odds_a = st.sidebar.number_input("Player A decimal odds", min_value=1.01, value=1.80, step=0.01)
odds_b = st.sidebar.number_input("Player B decimal odds", min_value=1.01, value=2.10, step=0.01)

def select_player(name):
    return max((players[pid] for pid in name_ids[name]), key=lambda item: item["matches"])

if st.button("ANALYZE MATCH", type="primary"):
    if a_name == b_name:
        st.error("Select two different players.")
        st.stop()
    a, b = select_player(a_name), select_player(b_name)
    ra = a["rank"] if a["rank"] and a["rank"] > 0 else 300
    rb = b["rank"] if b["rank"] and b["rank"] > 0 else 300
    row = pd.DataFrame([[
        a["elo"] - b["elo"],
        a["surface_elo"].get(surface, 1500) - b["surface_elo"].get(surface, 1500),
        np.log1p(rb) - np.log1p(ra),
        a["form10"] - b["form10"],
        np.log1p(a["matches"]) - np.log1p(b["matches"]),
        SURFACE_CODE[surface],
        best_of,
    ]], columns=FEATURES)
    pa = float(
        0.70 * bundle["xgb"].predict_proba(row)[0, 1]
        + 0.30 * bundle["logistic"].predict_proba(row)[0, 1]
    )
    pa = float(np.clip(pa, 0.01, 0.99))
    pb = 1 - pa
    ma = (1 / odds_a) / ((1 / odds_a) + (1 / odds_b))
    mb = 1 - ma
    ea, eb = pa - ma, pb - mb
    eva, evb = pa * odds_a - 1, pb * odds_b - 1

    st.subheader("Estimated win probabilities")
    c1, c2 = st.columns(2)
    c1.metric(a_name, f"{pa:.1%}")
    c2.metric(b_name, f"{pb:.1%}")
    st.subheader("Odds value analysis")
    table = pd.DataFrame({
        "Player": [a_name, b_name],
        "AI probability": [pa, pb],
        "No-vig market probability": [ma, mb],
        "Edge": [ea, eb],
        "EV per $1": [eva, evb],
        "Decimal odds": [odds_a, odds_b],
    })
    st.dataframe(
        table.style.format({
            "AI probability": "{:.1%}",
            "No-vig market probability": "{:.1%}",
            "Edge": "{:+.1%}",
            "EV per $1": "{:+.3f}",
            "Decimal odds": "{:.2f}",
        }),
        use_container_width=True,
    )
    best = max([(a_name, ea, eva), (b_name, eb, evb)], key=lambda item: item[1])
    if best[1] >= 0.03 and best[2] > 0:
        st.success(f"Potential value candidate: {best[0]} | edge {best[1]:+.1%} | EV {best[2]:+.3f} per $1")
    else:
        st.info("No qualifying value edge. Passing is a valid decision.")
    st.caption("Research estimates only, not guaranteed outcomes. No live injury/news or live odds feed is included. Verify current player status and prices.")

with st.expander("Validation metrics"):
    metrics_path = ART / "metrics.json"
    if metrics_path.exists():
        st.json(json.loads(metrics_path.read_text()))
    else:
        st.info("Metrics appear after training.")
