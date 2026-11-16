from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ART = Path("artifacts")
FEATURES = ["elo_diff","surface_elo_diff","rank_advantage","form_diff","experience_diff","surface_code","best_of"]
SURFACE_CODE = {"Hard":0,"Clay":1,"Grass":2,"Carpet":3}

st.set_page_config(page_title="Tennis Edge AI",page_icon="🎾",layout="wide")
st.title("🎾 Tennis Edge AI")
st.caption("Jeff Sackmann historical ATP/WTA data • Match-winner probabilities")

if not (ART/"model.joblib").exists() or not (ART/"player_state.json").exists():
    st.warning("Model not trained yet. Run `python train.py` first to download historical data and train it.")
    st.stop()

bundle = joblib.load(ART/"model.joblib")
if bundle.get("features") != FEATURES:
    st.error("Feature mismatch. Retrain with the train.py included in this project.")
    st.stop()
players = json.loads((ART/"player_state.json").read_text())["players"]
name_ids = {}
for pid,p in players.items(): name_ids.setdefault(p["name"],[]).append(pid)
names = sorted(name_ids)
if len(names)<2:
    st.error("Not enough player records."); st.stop()

st.sidebar.header("Match setup")
surface = st.sidebar.selectbox("Surface",list(SURFACE_CODE))
best_of = st.sidebar.selectbox("Match format",[3,5])
a_name = st.sidebar.selectbox("Player A",names,index=0)
b_name = st.sidebar.selectbox("Player B",names,index=1)
odds_a = st.sidebar.number_input("Player A decimal odds",min_value=1.01,value=1.80,step=.01)
odds_b = st.sidebar.number_input("Player B decimal odds",min_value=1.01,value=2.10,step=.01)

def player(name):
    return max((players[i] for i in name_ids[name]),key=lambda p:p["matches"])

if st.button("ANALYZE MATCH",type="primary"):
    if a_name == b_name:
        st.error("Select two different players."); st.stop()
    a,b=player(a_name),player(b_name)
    ra=a["rank"] if a["rank"] and a["rank"]>0 else 300
    rb=b["rank"] if b["rank"] and b["rank"]>0 else 300
    row=pd.DataFrame([[
        a["elo"]-b["elo"],
        a["surface_elo"].get(surface,1500)-b["surface_elo"].get(surface,1500),
        np.log1p(rb)-np.log1p(ra),
        a["form10"]-b["form10"],
        np.log1p(a["matches"])-np.log1p(b["matches"]),
        SURFACE_CODE[surface],best_of
    ]],columns=FEATURES)
    pa=float(.70*bundle["xgb"].predict_proba(row)[0,1]+.30*bundle["logistic"].predict_proba(row)[0,1])
    pa=float(np.clip(pa,.01,.99)); pb=1-pa
    ma=(1/odds_a)/((1/odds_a)+(1/odds_b)); mb=1-ma
    ea,eb=pa-ma,pb-mb
    eva,evb=pa*odds_a-1,pb*odds_b-1
    st.subheader("Estimated win probabilities")
    c1,c2=st.columns(2); c1.metric(a_name,f"{pa:.1%}"); c2.metric(b_name,f"{pb:.1%}")
    st.subheader("Odds value analysis")
    table=pd.DataFrame({"Player":[a_name,b_name],"AI probability":[pa,pb],
        "No-vig market probability":[ma,mb],"Edge":[ea,eb],
        "EV per $1":[eva,evb],"Decimal odds":[odds_a,odds_b]})
    st.dataframe(table.style.format({"AI probability":"{:.1%}","No-vig market probability":"{:.1%}","Edge":"{:+.1%}","EV per $1":"{:+.3f}","Decimal odds":"{:.2f}"}),use_container_width=True)
    best=max([(a_name,ea,eva),(b_name,eb,evb)],key=lambda z:z[1])
    if best[1]>=.03 and best[2]>0: st.success(f"Potential value candidate: {best[0]} | edge {best[1]:+.1%} | EV {best[2]:+.3f} per $1")
    else: st.info("No qualifying value edge. Passing is a valid decision.")
    st.caption("Research estimates only, not guaranteed outcomes. This dashboard has no live injury/news or live odds feed. Verify current player status and prices.")
with st.expander("Validation metrics"):
    m=ART/"metrics.json"
    if m.exists(): st.json(json.loads(m.read_text()))
    else: st.info("Metrics appear after training.")
