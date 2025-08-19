
# soccer_ev_app.py
# Moneyball Phil — Soccer EV App (v3.1)
# Full app: Compute Only + Compute & Save, Multi-match, Saved Bets, 2-Leg Parlay Builder
# Markets: Over 1.5, Over 2.5, BTTS — True % vs Implied %, EV (ROI per $), with NaN/input guards

import math
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# --------- Utils ---------
# =========================

def american_to_decimal(odds: float) -> float:
    """Convert American odds to Decimal odds."""
    if odds == 0:
        raise ValueError("American odds cannot be 0.")
    return 1.0 + (odds / 100.0) if odds > 0 else 1.0 + (100.0 / abs(odds))

def parse_odds(odds_input: str) -> Tuple[float, float]:
    """
    Accept American like '-120' / '+110' or Decimal like '1.83'.
    Returns (implied_prob [0..1], decimal_odds). Raises on invalid input.
    """
    s = str(odds_input).strip()
    if not s:
        raise ValueError("Odds field is empty.")
    # American?
    if s.startswith(('+', '-')):
        am = float(s)
        dec = american_to_decimal(am)
        if dec <= 1.0 or not np.isfinite(dec):
            raise ValueError("Invalid American odds → decimal ≤ 1.")
        return 1.0 / dec, dec
    # Decimal
    dec = float(s)
    if dec <= 1.0 or not np.isfinite(dec):
        raise ValueError("Decimal odds must be > 1.0")
    return 1.0 / dec, dec

def roi_per_dollar(true_p: float, dec_odds: float) -> float:
    """
    EV per $1 stake. If >0 you're +EV.
    EV_per_$ = true_p*(dec - 1) - (1 - true_p)
    """
    return true_p * (dec_odds - 1.0) - (1.0 - true_p)

def pct(x: float) -> str:
    return f"{x*100:.2f}%"

# ---- EV → Tier helper (soccer, same colors as your baseball apps) ----
def tier_from_ev(ev_roi: float):
    """Map EV (ROI per $1) to tier + color."""
    if ev_roi >= 0.20:   # 20%+
        return "Elite", "🟩"
    if ev_roi >= 0.10:   # 10–19.99%
        return "Strong", "🟨"
    if ev_roi >= 0.05:   # 5–9.99%
        return "Moderate", "🟧"
    return "Risky", "🟥"

def poisson_pmf(k: int, lam: float) -> float:
    # Guard: lam must be finite and >= 0
    if not np.isfinite(lam) or lam < 0:
        return np.nan

def poisson_pmf(k: int, lam: float) -> float:
    # Guard: lam must be finite and >= 0
    if not np.isfinite(lam) or lam < 0:
        return np.nan
    try:
        return math.exp(-lam) * lam**k / math.factorial(k)
    except OverflowError:
        return np.nan

def safe_goal_matrix(lam_home: float, lam_away: float, max_goals: int = 10) -> np.ndarray:
    """
    Build independent Poisson joint distribution with guards + normalization.
    """
    for name, lam in [("λ_home", lam_home), ("λ_away", lam_away)]:
        if not np.isfinite(lam):
            raise ValueError(f"{name} is not finite. Check your inputs.")
        if lam < 0:
            raise ValueError(f"{name} is negative. xG/xGA must be ≥ 0.")
    # Clamp to sane range to avoid tails exploding
    lam_home = float(np.clip(lam_home, 0.0, 4.0))
    lam_away = float(np.clip(lam_away, 0.0, 4.0))

    h = np.array([poisson_pmf(i, lam_home) for i in range(max_goals + 1)], dtype=float)
    a = np.array([poisson_pmf(j, lam_away) for j in range(max_goals + 1)], dtype=float)

    if np.isnan(h).any() or np.isnan(a).any():
        raise ValueError("Poisson PMF produced NaN. Check your λ values and inputs.")

    M = np.outer(h, a)
    total = M.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Distribution could not be normalized (sum ≤ 0).")
    return M / total

def market_probs_from_matrix(M: np.ndarray) -> Dict[str, float]:
    if np.isnan(M).any():
        raise ValueError("Probability matrix contains NaN.")
    over15 = over25 = btts = 0.0
    H, A = M.shape
    for i in range(H):
        for j in range(A):
            p = M[i, j]
            if i + j >= 2: over15 += p
            if i + j >= 3: over25 += p
            if i >= 1 and j >= 1: btts += p
    return {"O1.5": over15, "O2.5": over25, "BTTS": btts}

def is_num(x) -> bool:
    try:
        return np.isfinite(float(x))
    except:
        return False

# =========================
# ---- Qualification -------
# =========================

def qualify_market(market_key: str, true_p: float, ev_roi: float):
    """
    Returns (qualified, tier, reason). Tiers by true%:
      Elite ≥80%, Strong ≥70%, Moderate ≥60%, else Risky
    Gates:
      O1.5: true≥75% and EV≥+5%
      O2.5: true≥60% and EV≥+10%
      BTTS: true≥55% and EV≥+5%
    """
    tier = "Risky"
    if true_p >= 0.80: tier = "Elite"
    elif true_p >= 0.70: tier = "Strong"
    elif true_p >= 0.60: tier = "Moderate"

    if market_key == "O1.5":
        ok = (true_p >= 0.75) and (ev_roi >= 0.05)
        reason = "Needs ≥75% true and ≥+5% EV"
    elif market_key == "O2.5":
        ok = (true_p >= 0.60) and (ev_roi >= 0.10)
        reason = "Needs ≥60% true and ≥+10% EV"
    elif market_key == "BTTS":
        ok = (true_p >= 0.55) and (ev_roi >= 0.05)
        reason = "Needs ≥55% true and ≥+5% EV"
    else:
        ok, reason = False, "Unknown market"
    return ok, tier, reason

# =========================
# ----- Session Setup -----
# =========================

def init_state():
    st.session_state.setdefault("matches", [])    # saved matches
    st.session_state.setdefault("saved_bets", []) # saved bets (from matches)
    st.session_state.setdefault("id_counter", 1)

def next_id():
    st.session_state["id_counter"] += 1
    return st.session_state["id_counter"]

# =========================
# --------- App -----------
# =========================

st.set_page_config(page_title="Moneyball Phil — Soccer EV", layout="wide")
st.title("⚽ Moneyball Phil — Soccer EV App (v3.1)")
st.caption("Over 1.5 • Over 2.5 • BTTS — True % vs Implied % with EV/ROI • Multi-match • Saved Bets • 2-Leg Parlay Builder")
init_state()

# ---------------- Inputs ----------------
st.subheader("➕ Add / Compute a Match (Season totals only)")

# Seed used to force-clear inputs by changing widget keys
if "reset_seed" not in st.session_state:
    st.session_state["reset_seed"] = 0

def reset_inputs():
    st.session_state["reset_seed"] += 1   # change all widget keys
    st.rerun()

seed = st.session_state["reset_seed"]  # shorthand

# --- Helper to convert season totals -> per match (float) ---
def per_match(total_str, games_str):
    try:
        total = float(str(total_str).strip())
        games = float(str(games_str).strip())
        if games <= 0:
            return None
        return total / games
    except:
        return None

# ---- HOME TEAM (Season totals) ----
st.markdown("### Home Team — Season Totals")
hcol1, hcol2, hcol3 = st.columns(3)
with hcol1:
    home_team = st.text_input("Home Team", value="", key=f"home_team_name_{seed}", placeholder="Team name")
with hcol2:
    home_xg_total  = st.text_input("Home xG (SEASON TOTAL)",  value="", key=f"home_xg_total_{seed}",  placeholder="e.g., 23.1")
    home_xga_total = st.text_input("Home xGA (SEASON TOTAL)", value="", key=f"home_xga_total_{seed}", placeholder="e.g., 28.5")
with hcol3:
    home_season_matches = st.text_input("Matches Played (SEASON TOTAL)", value="", key=f"home_matches_total_{seed}", placeholder="e.g., 19")

# Compute per-match (floats) and strings for compute pipeline
_home_xg_for_val = per_match(home_xg_total, home_season_matches)
_home_xga_ag_val = per_match(home_xga_total, home_season_matches)
home_xg_for = f"{_home_xg_for_val:.3f}" if _home_xg_for_val is not None else ""
home_xga_ag = f"{_home_xga_ag_val:.3f}" if _home_xga_ag_val is not None else ""

st.caption(
    f"Home per-match → xG = {(_home_xg_for_val or 0):.3f} • xGA = {(_home_xga_ag_val or 0):.3f}"
    if (_home_xg_for_val is not None and _home_xga_ag_val is not None)
    else "Enter season totals + matches to auto-compute Home per-match values."
)

# ---- AWAY TEAM (Season totals) ----
st.markdown("### Away Team — Season Totals")
acol1, acol2, acol3 = st.columns(3)
with acol1:
    away_team = st.text_input("Away Team", value="", key=f"away_team_name_{seed}", placeholder="Team name")
with acol2:
    away_xg_total  = st.text_input("Away xG (SEASON TOTAL)",  value="", key=f"away_xg_total_{seed}",  placeholder="e.g., 25.4")
    away_xga_total = st.text_input("Away xGA (SEASON TOTAL)", value="", key=f"away_xga_total_{seed}", placeholder="e.g., 21.9")
with acol3:
    away_season_matches = st.text_input("Matches Played (SEASON TOTAL)", value="", key=f"away_matches_total_{seed}", placeholder="e.g., 19")

_away_xg_for_val = per_match(away_xg_total, away_season_matches)
_away_xga_ag_val = per_match(away_xga_total, away_season_matches)
away_xg_for = f"{_away_xg_for_val:.3f}" if _away_xg_for_val is not None else ""
away_xga_ag = f"{_away_xga_ag_val:.3f}" if _away_xga_ag_val is not None else ""

st.caption(
    f"Away per-match → xG = {(_away_xg_for_val or 0):.3f} • xGA = {(_away_xga_ag_val or 0):.3f}"
    if (_away_xg_for_val is not None and _away_xga_ag_val is not None)
    else "Enter season totals + matches to auto-compute Away per-match values."
)

# ---- ODDS INPUTS ----
st.markdown("### Odds (American or Decimal)")
ocol1, ocol2, ocol3 = st.columns(3)
with ocol1:
    odds_o15  = st.text_input("Over 1.5 Odds", value="", key=f"odds_o15_{seed}",  placeholder="-190 or 1.53")
with ocol2:
    odds_o25  = st.text_input("Over 2.5 Odds", value="", key=f"odds_o25_{seed}",  placeholder="+160 or 2.60")
with ocol3:
    odds_btts = st.text_input("BTTS Odds",     value="", key=f"odds_btts_{seed}", placeholder="+135 or 2.35")

# Actions
btn_cols = st.columns([1,1,2])
with btn_cols[0]:
    compute_only = st.button("Compute Only", key=f"btn_compute_only_{seed}")
with btn_cols[1]:
    compute_and_save = st.button("Compute & Save Match", key=f"btn_compute_save_{seed}")
with btn_cols[2]:
    if st.button("Reset Inputs", key=f"btn_reset_inputs_{seed}"):
        reset_inputs()






def compute_match(label: str,
                  home_xg_for_s: str, away_xga_s: str,
                  away_xg_for_s: str, home_xga_s: str,
                  odds_dict: Dict[str, str]):
    # 1) Validate numeric inputs
    fields = {
        "Home xG For": home_xg_for_s, "Away xGA Against": away_xga_s,
        "Away xG For": away_xg_for_s, "Home xGA Against": home_xga_s
    }
    for name, val in fields.items():
        if not is_num(val):
            raise ValueError(f"{name} is not a valid number: '{val}'")
        if float(val) < 0:
            raise ValueError(f"{name} cannot be negative.")

    hxf = float(home_xg_for_s)
    axga = float(away_xga_s)
    axf  = float(away_xg_for_s)
    hxga = float(home_xga_s)

    # 2) Build lambdas (transparent, same logic as before)
    lam_home = (hxf * axga) / 1.2
    lam_away = (axf * hxga) / 1.2

    # 3) Poisson matrix → market probabilities
    M = safe_goal_matrix(lam_home, lam_away, max_goals=10)
    probs = market_probs_from_matrix(M)

    # 4) Parse odds (raises on invalid)
    imp15, dec15 = parse_odds(odds_dict["O1.5"])
    imp25, dec25 = parse_odds(odds_dict["O2.5"])
    impBT, decBT = parse_odds(odds_dict["BTTS"])

    st.markdown(f"### Results for {label}")

    # 5) Compute and display each market with Tier
    results = []
    markets = [
        ("O1.5", "Over 1.5", imp15, dec15, odds_dict["O1.5"]),
        ("O2.5", "Over 2.5", imp25, dec25, odds_dict["O2.5"]),
        ("BTTS", "BTTS",     impBT, decBT, odds_dict["BTTS"]),
    ]

    for key, label_mkt, imp, dec, odds_str in markets:
        true_p = probs[key]
        ev = roi_per_dollar(true_p, dec)  # ROI per $1 (e.g., 0.1632 => 16.32%)
        tier, badge = tier_from_ev(ev)    # <-- uses the helper you added
        results.append({
            "key": key, "label": label_mkt, "true": true_p, "imp": imp,
            "ev": ev, "dec": dec, "odds_str": odds_str, "tier": tier, "badge": badge
        })
        st.write(
            f"{label_mkt}: True {pct(true_p)}, Implied {pct(imp)}, EV {pct(ev)} → "
            f"Tier: **{tier}** {badge}"
        )

    # 6) Recommended Play (highest positive EV, threshold 5%)
    best = max(results, key=lambda r: r["ev"])
    st.markdown("---")
    if best["ev"] >= 0.05:
        st.success(
            f"**Recommended Play (straight): {best['label']}** "
            f"({best['odds_str']}) • EV {pct(best['ev'])} • True {pct(best['true'])} • Tier: {best['tier']} {best['badge']}"
        )
    else:
        st.warning("No recommended straight bet (all EV < 5%).")

    # 7) Return odds parsed and lambdas so the rest of the app works the same
    odds_parsed = {
        "O1.5": {"imp": imp15, "dec": dec15, "str": odds_dict["O1.5"]},
        "O2.5": {"imp": imp25, "dec": dec25, "str": odds_dict["O2.5"]},
        "BTTS": {"imp": impBT, "dec": decBT, "str": odds_dict["BTTS"]},
    }
    return probs, odds_parsed, (lam_home, lam_away)


odds_dict = {"O1.5": odds_o15, "O2.5": odds_o25, "BTTS": odds_btts}
label = f"{home_team} vs {away_team}"

if compute_only or compute_and_save:
    try:
        probs, odds_parsed, (lam_h, lam_a) = compute_match(
            label, home_xg_for, away_xga_ag, away_xg_for, home_xga_ag, odds_dict
        )
        if compute_and_save:
            rec = {
                "id": next_id(),
                "label": label,
                "lambda_home": lam_h,
                "lambda_away": lam_a,
                "probs": probs,
                "odds": {
                    "O1.5": odds_parsed["O1.5"],
                    "O2.5": odds_parsed["O2.5"],
                    "BTTS": odds_parsed["BTTS"],
                }
            }
            st.session_state["matches"].append(rec)
            st.success("Match saved.")
    except Exception as e:
        st.error(f"⚠️ Compute error: {e}")

# ---------------- Saved Matches (EV-consistent tiers + Save buttons) ----------------
st.markdown("---")
st.subheader("📚 Saved Matches")
if not st.session_state["matches"]:
    st.info("No matches saved yet. Add one above.")
else:
    for match in st.session_state["matches"]:
        st.markdown(f"#### {match['label']}")
        lam_cols = st.columns(4)
        lam_cols[0].metric("λ Home",  f"{match['lambda_home']:.2f}")
        lam_cols[1].metric("λ Away",  f"{match['lambda_away']:.2f}")
        lam_cols[2].metric("Total λ", f"{match['lambda_home']+match['lambda_away']:.2f}")
        lam_cols[3].markdown("&nbsp;")

        # Header row (7 columns — last col is Save)
        head = st.columns([1.2,1,1,1,1,1,1])
        head[0].write("**Market**")
        head[1].write("**True %**")
        head[2].write("**Implied %**")
        head[3].write("**Edge (pp)**")
        head[4].write("**EV % (ROI/$)**")
        head[5].write("**Tier (by EV)**")
        head[6].write("**Save**")

        for mkt_key, mkt_label in [("O1.5", "Over 1.5"), ("O2.5", "Over 2.5"), ("BTTS", "BTTS")]:
            true_p = match["probs"][mkt_key]
            imp    = match["odds"][mkt_key]["imp"]
            dec    = match["odds"][mkt_key]["dec"]
            ev     = roi_per_dollar(true_p, dec)                 # EV per $1 (consistent)
            edge_pp = (true_p - imp)
            tier, badge = tier_from_ev(ev)                       # <-- EV-based tier everywhere

            row = st.columns([1.2,1,1,1,1,1,1])
            row[0].write(mkt_label)
            row[1].write(pct(true_p))
            row[2].write(pct(imp))
            row[3].write(f"{edge_pp*100:.2f} pp")
            row[4].write(pct(ev))
            row[5].write(f"**{tier}** {badge}")

            # Save button in the 7th column
            if row[6].button(f"💾 Save {mkt_label}", key=f"save_{match['id']}_{mkt_key}"):
                bet_id = next_id()
                st.session_state["saved_bets"].append({
                    "id": bet_id,
                    "match_id": match["id"],
                    "match_label": match["label"],
                    "market": mkt_key,
                    "market_label": mkt_label,
                    "true_p": true_p,
                    "implied_p": imp,
                    "dec": dec,
                    "odds_str": match["odds"][mkt_key]["str"],
                })
                st.success(f"Saved bet: {match['label']} — {mkt_label} ({match['odds'][mkt_key]['str']})")


# ---------------- Saved Bets ----------------
st.markdown("---")
st.subheader("💾 Saved Bets (ready for Parlay Builder or straight plays)")
if not st.session_state["saved_bets"]:
    st.info("No bets saved yet. Save from the matches above.")
else:
    sb_df = pd.DataFrame([
        {
            "Bet ID": b["id"],
            "Match": b["match_label"],
            "Market": b["market_label"],
            "True %": f"{b['true_p']*100:.2f}%",
            "Implied %": f"{b['implied_p']*100:.2f}%",
            "EV % (ROI/$)": f"{roi_per_dollar(b['true_p'], b['dec'])*100:.2f}%",
            "Odds": b["odds_str"],
        } for b in st.session_state["saved_bets"]
    ])
    st.dataframe(sb_df, use_container_width=True)

    # Deletion controls
    del_cols = st.columns([2,1,2])
    del_id = del_cols[0].number_input("Delete Bet by Bet ID", min_value=0, value=0, step=1)
    if del_cols[1].button("Delete Bet"):
        before = len(st.session_state["saved_bets"])
        st.session_state["saved_bets"] = [b for b in st.session_state["saved_bets"] if b["id"] != del_id]
        after = len(st.session_state["saved_bets"])
        st.success(f"Deleted Bet ID {del_id}.") if after < before else st.warning("No bet deleted (Bet ID not found).")

# ---------------- 2-Leg Parlay Builder ----------------
st.markdown("---")
st.subheader("🧮 2-Leg Parlay Builder")
st.caption("Pick any two saved bets (ideally from different matches). Assumes independence across matches.")

if len(st.session_state["saved_bets"]) < 2:
    st.info("Save at least two bets to build a parlay.")
else:
    options = {f"{b['id']} | {b['match_label']} | {b['market_label']} ({b['odds_str']})": b
               for b in st.session_state["saved_bets"]}
    pcols = st.columns(2)
    opt_list = list(options.keys())
    sel1 = pcols[0].selectbox("Leg 1", opt_list, index=0, key="parlay_leg1")
    sel2 = pcols[1].selectbox("Leg 2", opt_list, index=1, key="parlay_leg2")

    b1 = options[sel1]
    b2 = options[sel2]

    if b1["match_id"] == b2["match_id"]:
        st.warning("Both legs are from the SAME match. Use different matches for a cleaner independence assumption.")

    true_parlay = b1["true_p"] * b2["true_p"]
    dec_parlay = b1["dec"] * b2["dec"]
    imp_parlay = 1.0 / dec_parlay
    edge_pp = true_parlay - imp_parlay
    ev_parlay = roi_per_dollar(true_parlay, dec_parlay)

    colp = st.columns(6)
    colp[0].metric("Parlay Decimal", f"{dec_parlay:.3f}")
    colp[1].metric("True Parlay %", pct(true_parlay))
    colp[2].metric("Implied Parlay %", pct(imp_parlay))
    colp[3].metric("Edge (pp)", f"{edge_pp*100:.2f} pp")
    colp[4].metric("EV % (ROI/$)", pct(ev_parlay))
    colp[5].metric("Tier", "Elite" if true_parlay>=0.65 else ("Strong" if true_parlay>=0.58 else "Moderate"))

    st.markdown("**Copy-ready Tracker Row**")
    st.code(
        f"{b1['match_label']} — {b1['market_label']} ({b1['odds_str']})  +  "
        f"{b2['match_label']} — {b2['market_label']} ({b2['odds_str']}) | "
        f"True {pct(true_parlay)} | Implied {pct(imp_parlay)} | EV {pct(ev_parlay)} | Edge {edge_pp*100:.2f} pp",
        language="text"
    )

