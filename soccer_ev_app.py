
# soccer_ev_app.py (v3)
# Full version with Compute Only + Compute & Save across matches

import math
import numpy as np
import pandas as pd
import streamlit as st

def american_to_decimal(odds: float) -> float:
    if odds > 0:
        return 1.0 + (odds / 100.0)
    else:
        return 1.0 + (100.0 / abs(odds))

def parse_odds(odds_input: str):
    s = str(odds_input).strip()
    if s.startswith(('+','-')):
        am = float(s)
        dec = american_to_decimal(am)
        return 1.0/dec, dec
    dec = float(s)
    return 1.0/dec, dec

def roi_per_dollar(true_p: float, dec_odds: float) -> float:
    return true_p * (dec_odds - 1.0) - (1.0 - true_p)

def pct(x): return f"{x*100:.2f}%"

def poisson_pmf(k, lam): return math.exp(-lam) * lam**k / math.factorial(k)

def goal_matrix(lam_home, lam_away, max_goals=10):
    h = np.array([poisson_pmf(i, lam_home) for i in range(max_goals+1)])
    a = np.array([poisson_pmf(j, lam_away) for j in range(max_goals+1)])
    M = np.outer(h,a)
    return M/M.sum()

def market_probs_from_matrix(M):
    over15 = over25 = btts = 0.0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            p = M[i,j]
            if i+j>=2: over15+=p
            if i+j>=3: over25+=p
            if i>=1 and j>=1: btts+=p
    return {"O1.5": over15, "O2.5": over25, "BTTS": btts}

def init_state():
    st.session_state.setdefault("matches", [])
    st.session_state.setdefault("saved_bets", [])
    st.session_state.setdefault("id_counter", 1)
def next_id():
    st.session_state["id_counter"] += 1
    return st.session_state["id_counter"]

st.set_page_config(page_title="Soccer EV v3", layout="wide")
st.title("⚽ Moneyball Phil — Soccer EV App (v3)")
init_state()

colA, colB, colC = st.columns(3)
with colA:
    home_team = st.text_input("Home Team","Team A")
    home_xg_for = st.number_input("Home xG For",1.6)
    home_xga = st.number_input("Home xGA Against",1.2)
with colB:
    away_team = st.text_input("Away Team","Team B")
    away_xg_for = st.number_input("Away xG For",1.4)
    away_xga = st.number_input("Away xGA Against",1.3)
with colC:
    odds_o15 = st.text_input("Odds Over 1.5","-250")
    odds_o25 = st.text_input("Odds Over 2.5","+110")
    odds_btts = st.text_input("Odds BTTS","+100")
    compute_only = st.button("Compute Only")
    compute_and_save = st.button("Compute & Save Match")

lam_home = (home_xg_for*away_xga)/1.2
lam_away = (away_xg_for*home_xga)/1.2

def show_results(label, lam_home, lam_away, odds_dict, save=False):
    st.markdown(f"### Results for {label}")
    M = goal_matrix(lam_home, lam_away)
    probs = market_probs_from_matrix(M)
    for mk,label in [("O1.5","Over 1.5"),("O2.5","Over 2.5"),("BTTS","BTTS")]:
        imp,dec=parse_odds(odds_dict[mk])
        true_p=probs[mk]; ev=roi_per_dollar(true_p,dec)
        st.write(f"{label}: True {pct(true_p)}, Implied {pct(imp)}, EV {pct(ev)}")
    if save:
        match_id=next_id()
        st.session_state.matches.append({
            "id":match_id,"label":label,"lambda_home":lam_home,"lambda_away":lam_away,
            "probs":probs,"odds":odds_dict
        })
        st.success("Match saved.")
    return probs

odds_dict={"O1.5":odds_o15,"O2.5":odds_o25,"BTTS":odds_btts}

if compute_only:
    show_results(f"{home_team} vs {away_team}",lam_home,lam_away,odds_dict,save=False)

if compute_and_save:
    show_results(f"{home_team} vs {away_team}",lam_home,lam_away,odds_dict,save=True)

st.subheader("Saved Matches")
st.write(st.session_state.matches)
