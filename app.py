from pathlib import Path
import json
import uuid
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import train
from simulator import simulate_tennis_matches
import prediction_tracker

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
PLAYERS_PATH = ARTIFACT_DIR / "player_state.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
FEATURES = [
    "elo_diff", "surface_elo_diff", "rank_advantage",
    "form_diff", "form5_diff", "experience_diff",
    "h2h_diff", "rest_days_diff", "workload30_diff",
    "serve_win_diff", "return_win_diff",
    "surface_code", "best_of"
]
SURFACE_CODE = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}

st.set_page_config(page_title="Tennis Edge AI", page_icon="🎾", layout="wide")

# TENNIS EDGE NEON STADIUM THEME
st.markdown("""<style>
#MainMenu{visibility:hidden}
[data-testid="stAppViewContainer"]{background:
radial-gradient(ellipse at 50% 42%,rgba(255,126,44,.34),transparent 26%),
linear-gradient(180deg,rgba(2,8,12,.40) 0%,rgba(2,8,10,.20) 38%,rgba(0,8,5,.48) 100%),
linear-gradient(180deg,#101d2b 0%,#273847 42%,#081c12 68%,#031007 100%);background-attachment:fixed}
[data-testid="stAppViewContainer"]::before{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;opacity:.95;background:
radial-gradient(ellipse at 50% 44%,rgba(255,153,66,.46),transparent 22%),
linear-gradient(180deg,transparent 0 48%,rgba(0,0,0,.24) 60%,rgba(0,0,0,.72) 100%),
repeating-linear-gradient(90deg,transparent 0 8%,rgba(205,237,255,.18) 8.2% 8.45%,transparent 8.7% 18%),
linear-gradient(180deg,transparent 0 68%,rgba(28,125,48,.52) 68.2% 68.8%,transparent 69%),
linear-gradient(180deg,transparent 0 73%,rgba(220,255,224,.85) 73.2% 73.5%,transparent 73.8%),
linear-gradient(90deg,transparent 0 8%,rgba(220,255,224,.72) 8.2% 8.5%,transparent 8.7% 91%,rgba(220,255,224,.72) 91.2% 91.5%,transparent 91.7%),
linear-gradient(180deg,transparent 0 86%,rgba(120,255,0,.32) 86.2% 86.6%,transparent 87%)}
[data-testid="stHeader"],[data-testid="stMain"]{background:transparent}
.block-container{position:relative;z-index:1;padding-top:1.4rem}
h1,h2,h3{color:#f5fff5!important;text-shadow:0 0 18px rgba(135,255,0,.38)}
[data-testid="stMetric"],[data-testid="stExpander"],[data-testid="stVerticalBlockBorderWrapper"]{background:rgba(3,10,9,.82);border:1px solid rgba(145,255,50,.35);border-radius:16px}
.stButton>button[kind="primary"]{background:linear-gradient(100deg,#8dff00,#c0ff42);color:#071000;border:0;font-weight:800;border-radius:12px;box-shadow:0 0 18px rgba(130,255,0,.4)}
.stButton>button{border-radius:12px}
</style>""", unsafe_allow_html=True)
st.title("🎾 Tennis Edge AI")
st.caption("Historical ATP/WTA match data • Chronologically validated match-winner probabilities")

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

if st.button("🔄 UPDATE TENNIS DATA & RETRAIN MODEL", type="primary", use_container_width=True):
    if st.session_state.get("tennis_training_active", False):
        st.warning("A model update is already running. Please wait for it to finish.")
    else:
        st.session_state["tennis_training_active"] = True
        try:
            with st.spinner("Updating historical tennis data and retraining the model… Keep this page open until it finishes."):
                # Keep the last working model in place until train.main() successfully
                # writes its replacement. The training script overwrites artifacts at
                # the end, so deleting them first only risks leaving the app without a model.
                train.main()
                load_or_train_model.clear()
            st.session_state["tennis_training_active"] = False
            st.success("Tennis data updated and model retrained successfully.")
            st.rerun()
        except Exception as exc:
            st.session_state["tennis_training_active"] = False
            st.error("The data update or model retraining failed. Your previous model files were not deliberately deleted.")
            st.exception(exc)

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

def get_player_id(name):
    return max(name_to_ids[name], key=lambda pid: players[pid].get("matches", 0))

# LIVE MATCH CENTER — separate from the historical training data
@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_slate(status, api_key):
    response = requests.get(
        "https://api.livetennisapi.com/api/public/v1/matches",
        params={"status": status, "limit": 100},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("data", [])

def live_player_name(match, side):
    player = (match.get("players") or {}).get(side) or {}
    return player.get("name") or player.get("full_name")

@st.cache_data(ttl=45, show_spinner=False)
def fetch_rapid_live(api_key):
    response = requests.get(
        "https://tennis-api-atp-wta-itf.p.rapidapi.com/tennis/v2/extend/api/events/live",
        headers={"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": "tennis-api-atp-wta-itf.p.rapidapi.com"},
        timeout=20,
    )
    response.raise_for_status()
    return response.json().get("results", [])

@st.cache_data(ttl=30, show_spinner=False)
def fetch_draftkings_live_odds(event_id, api_key):
    response = requests.get(
        f"https://tennis-api-atp-wta-itf.p.rapidapi.com/tennis/v2/extend/api/event/recent-odds/get/{event_id}",
        params={"inplay": "true"},
        headers={"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": "tennis-api-atp-wta-itf.p.rapidapi.com"},
        timeout=20,
    )
    response.raise_for_status()
    return response.json().get("result", {})

@st.cache_data(ttl=30, show_spinner=False)
def fetch_odds_movement(event_id, api_key):
    response = requests.get(
        f"https://tennis-api-atp-wta-itf.p.rapidapi.com/tennis/v2/extend/api/odds/summary/movements/last-10/{event_id}",
        headers={"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": "tennis-api-atp-wta-itf.p.rapidapi.com"},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()

@st.cache_data(ttl=45, show_spinner=False)
def fetch_match_momentum(match_id, match_type, api_key):
    response = requests.get(
        f"https://tennis-api-atp-wta-itf.p.rapidapi.com/tennis/v2/matches/{match_id}/momentum",
        params={"type": match_type},
        headers={"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": "tennis-api-atp-wta-itf.p.rapidapi.com"},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()

def find_quote_records(payload):
    records = []
    def walk(node):
        if isinstance(node, dict):
            if node.get("od1") is not None and node.get("od2") is not None:
                records.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(payload)
    return records

def find_draftkings(payload):
    matches = []
    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                next_path = f"{path}/{key}"
                if "draftkings" in str(key).casefold():
                    matches.extend(find_quote_records(value))
                walk(value, next_path)
        elif isinstance(node, list):
            for value in node:
                walk(value, path)
    walk(payload)
    return matches

@st.cache_data(ttl=30, show_spinner=False)
def fetch_named_match_odds(player_one, player_two, match_date, api_key):
    from urllib.parse import quote
    p1 = quote(player_one, safe="")
    p2 = quote(player_two, safe="")
    url = f"https://tennis-api-atp-wta-itf.p.rapidapi.com/tennis/v2/extend/api/event/get/{p1}/{p2}/{match_date}"
    response = requests.get(
        url, params={"odds": "yes"},
        headers={"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": "tennis-api-atp-wta-itf.p.rapidapi.com"},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()

def extract_draftkings_quotes(payload):
    found = []
    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if "draftkings" in str(key).casefold():
                    if isinstance(value, dict):
                        for market, quote in value.items():
                            if isinstance(quote, dict) and quote.get("od1") and quote.get("od2"):
                                found.append((str(market), quote))
                            elif isinstance(quote, list):
                                for item in quote:
                                    if isinstance(item, dict) and item.get("od1") and item.get("od2"):
                                        found.append((str(market), item))
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict) and item.get("od1") and item.get("od2"):
                                found.append(("Match winner", item))
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
    walk(payload.get("result") or {})
    return found

def live_score_text(match):
    score = match.get("score") or {}
    sets = score.get("sets") or []
    games = score.get("games") or []
    bits = []
    if sets:
        bits.append("Sets " + "–".join(str(x) for x in sets))
    if games and len(games) == 2:
        per_set = []
        for a, b in zip(games[0], games[1]):
            per_set.append(f"{a}–{b}")
        if per_set:
            bits.append("Games " + "  ".join(per_set))
    points = score.get("points") or []
    if points:
        bits.append("Points " + "–".join(str(x) for x in points))
    return " · ".join(bits) if bits else "Score not posted yet"

st.subheader("⚡ Live Match Center")
live_key = st.secrets.get("LIVETENNISAPI_KEY", "")
if not live_key:
    st.info("Connect your Live Tennis API key in Streamlit → App settings → Secrets as LIVETENNISAPI_KEY.")
else:
    live_tab, upcoming_tab = st.tabs(["🔴 Live now", "📅 Upcoming"])
    for tab, slate_status in [(live_tab, "live"), (upcoming_tab, "upcoming")]:
        with tab:
            try:
                rapid_key = st.secrets.get("TENNIS_RAPIDAPI_KEY", "")
                if slate_status == "live" and rapid_key:
                    rapid_rows = fetch_rapid_live(rapid_key)
                    matches = []
                    for row in rapid_rows:
                        p1 = row.get("participant1") or (row.get("player1") or {}).get("name")
                        p2 = row.get("participant2") or (row.get("player2") or {}).get("name")
                        if p1 and p2:
                            matches.append({
                                "match_id": row.get("matchId") or row.get("match_id") or row.get("fixtureId"),
                                "rapid_event_id": row.get("id") or row.get("eventId"),
                                "rapid_raw": row,
                                "match_type": str(row.get("type") or "atp").lower(),
                                "players": {"p1": {"name": p1}, "p2": {"name": p2}},
                                "tournament": row.get("league") or (row.get("tournament") or {}).get("name"),
                                "surface": (row.get("tournament") or {}).get("surface"),
                                "score": {"raw": row.get("score"), "points": row.get("points")},
                                "status": row.get("status"),
                            })
                else:
                    matches = fetch_live_slate(slate_status, live_key)
                singles = []
                for match in matches:
                    p1, p2 = live_player_name(match, "p1"), live_player_name(match, "p2")
                    if p1 and p2 and match.get("draw") != "doubles":
                        singles.append(match)
                if not singles:
                    st.info("No singles matches are listed in this feed right now.")
                else:
                    labels, match_by_label = [], {}
                    for match in singles:
                        p1, p2 = live_player_name(match, "p1"), live_player_name(match, "p2")
                        tournament = match.get("tournament") or "Tournament not listed"
                        score_text = live_score_text(match) if slate_status == "live" else (match.get("scheduled_time") or "Start time TBA")
                        label = f"{p1} vs {p2}  |  {tournament}  |  {score_text}"
                        labels.append(label)
                        match_by_label[label] = match
                    selected_label = st.selectbox("Choose a match", labels, key=f"{slate_status}_match_select")
                    selected_match = match_by_label[selected_label]
                    p1, p2 = live_player_name(selected_match, "p1"), live_player_name(selected_match, "p2")
                    surface_live = selected_match.get("surface")
                    best_of_live = selected_match.get("best_of")
                    c1, c2 = st.columns(2)
                    c1.metric("Player 1", p1)
                    c2.metric("Player 2", p2)
                    st.caption(
                        f"Tournament: {selected_match.get('tournament') or 'Not listed'}"
                        f" · Surface: {surface_live or 'Not reported'}"
                        f" · Round: {selected_match.get('round') or 'Not reported'}"
                    )
                    raw_live_score = (selected_match.get("score") or {}).get("raw")
                    score_display = raw_live_score or live_score_text(selected_match)
                    raw_points = (selected_match.get("score") or {}).get("points")
                    if raw_points:
                        score_display += f" · Points {raw_points}"
                    st.write("**Current score:** " + score_display)
                    rapid_key = st.secrets.get("TENNIS_RAPIDAPI_KEY", "")
                    rapid_event_id = selected_match.get("rapid_event_id")
                    if rapid_key and rapid_event_id:
                        st.markdown("#### 📈 Live odds & momentum")
                        refresh_live_panel = st.button("REFRESH LIVE ODDS + MOMENTUM", key=f"refresh_live_panel_{slate_status}", use_container_width=True)
                        try:
                            dk_payload = fetch_draftkings_live_odds(rapid_event_id, rapid_key)
                            dk_quotes = find_draftkings(dk_payload)
                            if dk_quotes:
                                quote = dk_quotes[-1]
                                st.success(f"DraftKings latest quote — {p1}: {quote.get('od1', '—')} · {p2}: {quote.get('od2', '—')}")
                                st.caption(f"Quote timestamp: {quote.get('addTime') or quote.get('timestamp') or 'Not supplied by provider'}")
                            else:
                                st.info("Live odds endpoint responded, but no DraftKings match-winner quote was found in its response.")
                        except requests.HTTPError as exc:
                            code = exc.response.status_code if exc.response is not None else "unknown"
                            st.warning(f"Live odds unavailable (HTTP {code}). Check RapidAPI subscription/plan access.")
                        except requests.RequestException as exc:
                            st.warning(f"Live odds request failed: {exc}")
                        try:
                            movement_payload = fetch_odds_movement(rapid_event_id, rapid_key)
                            movement_quotes = find_draftkings(movement_payload)
                            if movement_quotes:
                                movement_rows = []
                                for item in movement_quotes:
                                    movement_rows.append({
                                        "Updated": item.get("addTime") or item.get("timestamp") or "—",
                                        p1: item.get("od1"), p2: item.get("od2")
                                    })
                                st.write("**DraftKings recent odds history**")
                                st.dataframe(pd.DataFrame(movement_rows), use_container_width=True, hide_index=True)
                                if len(movement_rows) >= 2:
                                    st.caption("Compare earliest and latest returned quotes to see whether prices shortened or drifted.")
                            else:
                                st.caption("No DraftKings movement history was present in the returned data.")
                        except requests.HTTPError as exc:
                            code = exc.response.status_code if exc.response is not None else "unknown"
                            st.caption(f"Odds movement endpoint unavailable (HTTP {code}).")
                        except requests.RequestException as exc:
                            st.caption(f"Odds movement request failed: {exc}")
                        raw_event = selected_match.get("rapid_raw") or {}
                        match_id = selected_match.get("match_id") or raw_event.get("id")
                        match_type = selected_match.get("match_type") or "atp"
                        if match_id:
                            try:
                                momentum_payload = fetch_match_momentum(match_id, match_type, rapid_key)
                                st.write("**Match momentum feed**")
                                momentum_result = momentum_payload.get("result", momentum_payload)
                                st.json(momentum_result)
                            except requests.HTTPError as exc:
                                code = exc.response.status_code if exc.response is not None else "unknown"
                                st.caption(f"Momentum endpoint unavailable (HTTP {code}); this event may not expose a compatible match ID.")
                            except requests.RequestException as exc:
                                st.caption(f"Momentum request failed: {exc}")
                        else:
                            st.caption("Momentum will appear when the live event includes its separate match ID.")
                    elif slate_status == "live":
                        st.caption("Add TENNIS_RAPIDAPI_KEY in Streamlit Secrets to enable live odds, movement, and momentum.")
                    if st.button("ANALYZE THIS MATCH", key=f"analyze_{slate_status}_match", type="primary", use_container_width=True):
                        missing = [name for name in (p1, p2) if name not in name_to_ids]
                        if missing:
                            st.warning("The historical model does not have a matching player record for: " + ", ".join(missing) + ". Use Manual Entry to search the closest stored name.")
                        else:
                            st.session_state["search_player_a_v2"] = p1
                            st.session_state["search_player_b_v2"] = p2
                            st.session_state["choose_player_a_v2"] = p1
                            st.session_state["choose_player_b_v2"] = p2
                            if surface_live and surface_live.title() in SURFACE_CODE:
                                st.session_state["surface_v2"] = surface_live.title()
                            if best_of_live in (3, 5):
                                st.session_state["best_of_v2"] = best_of_live
                            st.session_state["live_match_context"] = {
                                "match_id": selected_match.get("match_id") or selected_match.get("id"),
                                "status": slate_status,
                                "score": live_score_text(selected_match),
                                "tournament": selected_match.get("tournament"),
                                "surface": surface_live,
                            }
                            st.rerun()
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else "unknown"
                detail = ""
                if exc.response is not None:
                    try:
                        detail = exc.response.text[:700]
                    except Exception:
                        detail = ""
                st.error(f"Live match feed returned HTTP {status_code}.")
                if detail:
                    st.code(detail, language="text")
                st.info("For this feed, the key must be saved as LIVETENNISAPI_KEY in Streamlit Secrets. The provider's FREE tier includes live and upcoming match lists; 403 response details will tell us whether the key, account, or access was refused.")
            except requests.RequestException as exc:
                st.warning(f"Could not load the live match feed right now: {exc}")
            except (ValueError, TypeError) as exc:
                st.warning(f"The live match feed returned an unexpected response: {exc}")
    st.caption("Live Tennis API match lists are cached for 5 minutes to reduce free-tier rate-limit usage.")

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

st.subheader("🏷️ DraftKings odds lookup")
st.caption("This uses RapidAPI independently of the Live Tennis API match feed.")
rapid_odds_key = st.secrets.get("TENNIS_RAPIDAPI_KEY", "")
odds_match_date = st.date_input("Match date for DraftKings lookup", value=__import__("datetime").date.today(), key="dk_match_date")
if st.button("FETCH DRAFTKINGS ODDS", use_container_width=True, disabled=not bool(rapid_odds_key) or not bool(player_a_name and player_b_name)):
    try:
        with st.spinner("Looking up this matchup and DraftKings prices…"):
            odds_payload = fetch_named_match_odds(player_a_name, player_b_name, odds_match_date.isoformat(), rapid_odds_key)
        dk_quotes = extract_draftkings_quotes(odds_payload)
        if dk_quotes:
            st.session_state["dk_lookup_quotes"] = [
                {"market": market, "player_a": quote.get("od1"), "player_b": quote.get("od2"),
                 "updated": quote.get("addTime"), "type": quote.get("oddType")}
                for market, quote in dk_quotes
            ]
            first = dk_quotes[0][1]
            try:
                st.session_state["odds_a_v2"] = float(first.get("od1"))
                st.session_state["odds_b_v2"] = float(first.get("od2"))
            except (TypeError, ValueError):
                pass
            st.success("DraftKings prices found. The first quote has been loaded into the odds fields below.")
        else:
            st.session_state["dk_lookup_quotes"] = []
            st.info("No DraftKings quote was returned for this matchup/date.")
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "unknown"
        detail = exc.response.text[:500] if exc.response is not None else ""
        st.error(f"RapidAPI odds lookup returned HTTP {code}.")
        if detail:
            st.code(detail, language="text")
    except requests.RequestException as exc:
        st.error(f"Could not retrieve DraftKings odds: {exc}")

for quote in st.session_state.get("dk_lookup_quotes", []):
    st.write(f"**{quote['market']}** — Player A: {quote['player_a']} · Player B: {quote['player_b']} · {quote.get('type') or 'odds'}")

odds_a = st.number_input("Player A decimal odds", min_value=1.01, value=1.80, step=0.01, key="odds_a_v2")
odds_b = st.number_input("Player B decimal odds", min_value=1.01, value=2.10, step=0.01, key="odds_b_v2")


st.subheader("🎾 Head-to-Head Match Simulator")
st.caption("Run virtual point-by-point matches using each player's stored serve/return profile, overall Elo, surface Elo, and selected match format.")
simulation_count = st.selectbox("Number of simulated matches", [1000, 5000, 10000], index=1, key="simulation_count_v1")
if st.button("▶ SIMULATE MATCHES", type="primary", use_container_width=True, disabled=not bool(player_a_name and player_b_name)):
    if player_a_name == player_b_name:
        st.error("Choose two different players.")
    else:
        sim_a = get_player(player_a_name)
        sim_b = get_player(player_b_name)
        with st.spinner(f"Simulating {simulation_count:,} virtual matches…"):
            simulation = simulate_tennis_matches(
                sim_a, sim_b, surface, best_of, simulations=int(simulation_count)
            )
        st.session_state["last_match_simulation"] = {
            "player_a": player_a_name, "player_b": player_b_name,
            "surface": surface, "best_of": best_of,
            "count": int(simulation_count), "result": simulation,
        }

last_simulation = st.session_state.get("last_match_simulation")
if last_simulation and last_simulation.get("player_a") == player_a_name and last_simulation.get("player_b") == player_b_name and last_simulation.get("surface") == surface and last_simulation.get("best_of") == best_of:
    sim_result = last_simulation["result"]
    st.markdown(f"**Results from {last_simulation['count']:,} simulated best-of-{best_of} matches**")
    sim_col_a, sim_col_b = st.columns(2)
    sim_col_a.metric(player_a_name, f"{sim_result['probability_a']:.1%}", f"{sim_result['wins_a']:,} wins")
    sim_col_b.metric(player_b_name, f"{sim_result['probability_b']:.1%}", f"{sim_result['wins_b']:,} wins")
    st.write("**Most common match scores (sets)**")
    st.dataframe(
        pd.DataFrame([{"Set-by-set score": row["score"], "Simulations": row["count"], "Share": row["rate"]} for row in sim_result["score_distribution"][:8]]),
        use_container_width=True, hide_index=True
    )
    st.caption("Point-by-point Monte Carlo using advantage games and a 7-point tiebreak at 6-6 in every set. First server is randomized. Fatigue, injuries, weather and player-specific clutch effects are not modeled.")

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
        a.get("form5", a["form10"]) - b.get("form5", b["form10"]),
        np.log1p(a["matches"]) - np.log1p(b["matches"]),
        (a.get("h2h", {}).get(get_player_id(player_b_name), 0) - b.get("h2h", {}).get(get_player_id(player_a_name), 0)) / max(1, a.get("h2h", {}).get(get_player_id(player_b_name), 0) + b.get("h2h", {}).get(get_player_id(player_a_name), 0)),
        (pd.Timestamp(a["last_played"]) - pd.Timestamp(b["last_played"])).days if a.get("last_played") and b.get("last_played") else 0.0,
        float(b.get("workload30", 0)) - float(a.get("workload30", 0)),
        a.get("serve_win_rate", 0.5) - b.get("serve_win_rate", 0.5),
        a.get("return_win_rate", 0.5) - b.get("return_win_rate", 0.5),
        SURFACE_CODE[surface],
        best_of,
    ]], columns=FEATURES)

    xgb_weight = float(bundle.get("xgb_weight", 0.70))
    logistic_weight = 1.0 - xgb_weight
    raw_probability_a = float(
        xgb_weight * bundle["xgb"].predict_proba(features)[0, 1]
        + logistic_weight * bundle["logistic"].predict_proba(features)[0, 1]
    )
    calibrator = bundle.get("calibrator")
    if calibrator is not None:
        clipped_raw = float(np.clip(raw_probability_a, 1e-6, 1 - 1e-6))
        raw_logit = np.log(clipped_raw / (1 - clipped_raw)).reshape(1, 1)
        probability_a = float(calibrator.predict_proba(raw_logit)[0, 1])
    else:
        probability_a = raw_probability_a
    probability_a = float(np.clip(probability_a, 0.01, 0.99))
    probability_b = 1 - probability_a
    tracker_token = st.secrets.get("GITHUB_TOKEN", "")
    tracker_repository = st.secrets.get("GITHUB_REPOSITORY", "retrovibexx/Tennis-edge-2.0")
    prediction_record = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "player_a": player_a_name,
        "player_b": player_b_name,
        "surface": surface,
        "best_of": int(best_of),
        "probability_a": probability_a,
        "probability_b": probability_b,
        "predicted_winner": player_a_name if probability_a >= 0.5 else player_b_name,
        "odds_a": float(odds_a),
        "odds_b": float(odds_b),
        "actual_winner": None,
        "correct": None,
    }
    try:
        prediction_tracker.append_prediction(tracker_token, prediction_record, tracker_repository)
        st.success("Prediction saved to the persistent accuracy tracker.")
    except Exception as tracker_error:
        st.warning(f"Prediction analyzed but not saved to the tracker: {tracker_error}")
    market_a = (1 / odds_a) / ((1 / odds_a) + (1 / odds_b))
    market_b = 1 - market_a
    edge_a, edge_b = probability_a - market_a, probability_b - market_b
    ev_a, ev_b = probability_a * odds_a - 1, probability_b * odds_b - 1

    st.subheader("Calibrated estimated win probabilities")
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

with st.expander("📊 Persistent prediction accuracy tracker", expanded=True):
    tracker_token = st.secrets.get("GITHUB_TOKEN", "")
    tracker_repository = st.secrets.get("GITHUB_REPOSITORY", "retrovibexx/Tennis-edge-2.0")
    if not tracker_token:
        st.warning("To keep records across app restarts, add GITHUB_TOKEN to Streamlit Secrets. It needs Contents: Read and write access to this repository.")
        st.code('GITHUB_TOKEN = "your_fine_grained_token"\nGITHUB_REPOSITORY = "retrovibexx/Tennis-edge-2.0"', language="toml")
    else:
        try:
            tracker_data, _ = prediction_tracker.load_tracker(tracker_token, tracker_repository)
            tracked = tracker_data.get("predictions", [])
            resolved = [item for item in tracked if item.get("actual_winner")]
            correct = sum(bool(item.get("correct")) for item in resolved)
            m1, m2, m3 = st.columns(3)
            m1.metric("Predictions logged", len(tracked))
            m2.metric("Graded predictions", len(resolved))
            m3.metric("Accuracy", f"{correct / len(resolved):.1%}" if resolved else "—")
            if resolved:
                st.dataframe(pd.DataFrame([{"Date": item.get("created_at", "")[:10], "Match": f"{item.get('player_a')} vs {item.get('player_b')}", "Pick": item.get("predicted_winner"), "Actual": item.get("actual_winner"), "Correct": "Yes" if item.get("correct") else "No"} for item in resolved[-50:][::-1]]), use_container_width=True, hide_index=True)
            pending = [item for item in tracked if not item.get("actual_winner")]
            if pending:
                options = {f"{item.get('created_at', '')[:10]} · {item.get('player_a')} vs {item.get('player_b')} · pick: {item.get('predicted_winner')}": item for item in pending}
                selected_label = st.selectbox("Grade a completed match", list(options), key="tracker_pending_select")
                selected = options[selected_label]
                actual = st.radio("Actual winner", [selected["player_a"], selected["player_b"]], horizontal=True, key="tracker_actual_winner")
                if st.button("SAVE MATCH RESULT", key="save_tracker_result", use_container_width=True):
                    prediction_tracker.grade_prediction(tracker_token, selected["id"], actual, tracker_repository)
                    st.success("Match result saved.")
                    st.rerun()
            else:
                st.info("No ungraded predictions yet. Analyze a match to log one.")
        except Exception as tracker_error:
            st.error(f"Could not load tracker: {tracker_error}")

with st.expander("Model validation metrics"):
    if METRICS_PATH.exists():
        st.json(json.loads(METRICS_PATH.read_text(encoding="utf-8")))
    else:
        st.info("Validation metrics will appear after automatic training completes.")
