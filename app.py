from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import train

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
PLAYERS_PATH = ARTIFACT_DIR / "player_state.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
FEATURES = [
    "elo_diff", "surface_elo_diff", "rank_advantage",
    "form_diff", "experience_diff", "surface_code", "best_of"
]
SURFACE_CODE = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}

st.set_page_config(page_title="Tennis Edge AI", page_icon="🎾", layout="wide")

# TENNIS EDGE NEON STADIUM THEME
st.markdown("""<style># MainMenu{visibility:hidden}
[data-testid="stAppViewContainer"]{background:radial-gradient(ellipse at 50% 75%,rgba(38,255,0,.2),transparent 26%),linear-gradient(180deg,rgba(1,7,7,.96),rgba(2,9,13,.78) 42%,rgba(1,8,7,.94)),linear-gradient(160deg,#020807,#17232b 54%,#06120b);background-attachment:fixed}
[data-testid="stAppViewContainer"]::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.7;background:linear-gradient(180deg,transparent 52%,rgba(0,0,0,.5)),linear-gradient(90deg,transparent 9%,rgba(180,255,190,.13) 9.2%,transparent 9.5%,transparent 90%,rgba(180,255,190,.13) 90.2%,transparent 90.5%),linear-gradient(180deg,transparent 73%,rgba(160,255,170,.2) 73.2%,transparent 73.5%),radial-gradient(ellipse at 50% 58%,rgba(255,117,38,.16),transparent 36%)}
[data-testid="stHeader"],[data-testid="stMain"]{background:transparent}
.block-container{position:relative;z-index:1;padding-top:1.4rem}
h1,h2,h3{color:#f5fff5!important;text-shadow:0 0 18px rgba(135,255,0,.26)}
[data-testid="stMetric"],[data-testid="stExpander"],[data-testid="stVerticalBlockBorderWrapper"]{background:rgba(4,13,13,.86);border:1px solid rgba(145,255,50,.25);border-radius:16px}
.stButton>button[kind="primary"]{background:linear-gradient(100deg,#8dff00,#c0ff42);color:#071000;border:0;font-weight:800;border-radius:12px;box-shadow:0 0 18px rgba(130,255,0,.34)}
.stButton>button{border-radius:12px}</style>""", unsafe_allow_html=True)
st.title("🎾 Tennis Edge AI")
st.caption("Historical ATP/WTA match data • Estimated match-winner probabilities")

@st.cache_resource(show_spinner=False)
def load_or_train_model():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if not (MODEL_PATH.exists() and PLAYERS_PATH.exists()):
        # First launch: train automatically. No manual terminal command required.
        train.main()
    if not (MODEL_PATH.exists() and PLAYERS_PATH.exists()):
        raise RuntimeError("Training finished without creating the required model files.")
    model_bundle = joblib.load(MODEL_PATH)
    if model_bundle.get("features") != FEATURES:
        raise RuntimeError("Saved model features do not match this app. Clear artifacts and reload.")
    player_data = json.loads(PLAYERS_PATH.read_text(encoding="utf-8"))["players"]
    return model_bundle, player_data

if st.sidebar.button("Update tennis data & retrain", use_container_width=True):
    for artifact in (MODEL_PATH, PLAYERS_PATH, METRICS_PATH):
        if artifact.exists():
            artifact.unlink()
    load_or_train_model.clear()
    st.rerun()

try:
    with st.spinner("Preparing Tennis Edge AI — training the model automatically on first launch…"):
        bundle, players = load_or_train_model()
except Exception as exc:
    st.error("The automatic model setup did not complete.")
    st.exception(exc)
    st.info("Fix the reported download or training error, then reload this page. You do not need to run a terminal command.")
    st.stop()

name_to_ids = {}
for player_id, record in players.items():
    name_to_ids.setdefault(record["name"], []).append(player_id)
player_names = sorted(name_to_ids)
if len(player_names) < 2:
    st.error("The historical dataset did not produce enough player records.")
    st.stop()

def get_player(name):
    return max((players[pid] for pid in name_to_ids[name]),
               key=lambda record: record.get("matches", 0))

st.subheader("Match setup")
surface = st.selectbox("Surface", list(SURFACE_CODE), key="surface_v2")
best_of = st.selectbox("Match format", [3, 5], key="best_of_v2")

def matching_players(query):
    query = query.strip().casefold()
    if not query:
        return player_names
    return [name for name in player_names if query in name.casefold()]

# Keep search and selection on the main page so they are visible on iPhone.
search_a = st.text_input("Search Player A", placeholder="Type a name, e.g. Jannik Sinner", key="search_player_a_v2")
options_a = matching_players(search_a)
if not options_a:
    st.warning("No Player A matches. Try a shorter spelling.")
    player_a_name = None
else:
    player_a_name = st.selectbox("Choose Player A", options_a, key="choose_player_a_v2")

search_b = st.text_input("Search Player B", placeholder="Type a name, e.g. Carlos Alcaraz", key="search_player_b_v2")
options_b = [name for name in matching_players(search_b) if name != player_a_name]
if not options_b:
    st.warning("No different Player B matches. Search for another player.")
    player_b_name = None
else:
    default_b_index = 0
    player_b_name = st.selectbox("Choose Player B", options_b, index=default_b_index, key="choose_player_b_v2")

odds_a = st.number_input("Player A decimal odds", min_value=1.01, value=1.80, step=0.01, key="odds_a_v2")
odds_b = st.number_input("Player B decimal odds", min_value=1.01, value=2.10, step=0.01, key="odds_b_v2")

if st.button("ANALYZE MATCH", type="primary", use_container_width=True):
    if not player_a_name or not player_b_name:
        st.error("Search for and select two players first.")
        st.stop()
    if player_a_name == player_b_name:
        st.error("Choose two different players.")
        st.stop()

    a, b = get_player(player_a_name), get_player(player_b_name)
    rank_a = a.get("rank") if a.get("rank") and a["rank"] > 0 else 300
    rank_b = b.get("rank") if b.get("rank") and b["rank"] > 0 else 300
    features = pd.DataFrame([[
        a["elo"] - b["elo"],
        a["surface_elo"].get(surface, 1500) - b["surface_elo"].get(surface, 1500),
        np.log1p(rank_b) - np.log1p(rank_a),
        a["form10"] - b["form10"],
        np.log1p(a["matches"]) - np.log1p(b["matches"]),
        SURFACE_CODE[surface],
        best_of,
    ]], columns=FEATURES)

    probability_a = float(
        0.70 * bundle["xgb"].predict_proba(features)[0, 1]
        + 0.30 * bundle["logistic"].predict_proba(features)[0, 1]
    )
    probability_a = float(np.clip(probability_a, 0.01, 0.99))
    probability_b = 1 - probability_a
    market_a = (1 / odds_a) / ((1 / odds_a) + (1 / odds_b))
    market_b = 1 - market_a
    edge_a, edge_b = probability_a - market_a, probability_b - market_b
    ev_a, ev_b = probability_a * odds_a - 1, probability_b * odds_b - 1

    st.subheader("Estimated win probabilities")
    col_a, col_b = st.columns(2)
    col_a.metric(player_a_name, f"{probability_a:.1%}")
    col_b.metric(player_b_name, f"{probability_b:.1%}")
    st.subheader("Odds value comparison")
    results = pd.DataFrame({
        "Player": [player_a_name, player_b_name],
        "AI win probability": [probability_a, probability_b],
        "No-vig market probability": [market_a, market_b],
        "Model edge": [edge_a, edge_b],
        "Expected value per $1": [ev_a, ev_b],
        "Decimal odds": [odds_a, odds_b],
    })
    st.dataframe(
        results.style.format({
            "AI win probability": "{:.1%}",
            "No-vig market probability": "{:.1%}",
            "Model edge": "{:+.1%}",
            "Expected value per $1": "{:+.3f}",
            "Decimal odds": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    best = max(
        [(player_a_name, edge_a, ev_a), (player_b_name, edge_b, ev_b)],
        key=lambda item: item[1],
    )
    if best[1] >= 0.03 and best[2] > 0:
        st.success(
            f"Potential value candidate: {best[0]} | "
            f"edge {best[1]:+.1%} | EV {best[2]:+.3f} per $1"
        )
    else:
        st.info("No qualifying value edge. Passing is a valid decision.")
    st.caption(
        "Estimates are not guarantees. This app does not include live injury, "
        "withdrawal, weather, or live-odds feeds. Check current match information."
    )

with st.expander("Model validation metrics"):
    if METRICS_PATH.exists():
        st.json(json.loads(METRICS_PATH.read_text(encoding="utf-8")))
    else:
        st.info("Validation metrics will appear after automatic training completes.")
