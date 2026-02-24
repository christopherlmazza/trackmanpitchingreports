"""
TrackMan Pitching Report — Streamlit App
==========================================
Install:   pip install streamlit
Run with:  streamlit run trackman_app.py

Flow:
  1. Pick date range in sidebar
  2. App fetches sessions, shows team dropdown
  3. Pick team -> app finds pitchers
  4. Pick pitcher(s) -> Generate report
  5. Preview in browser + PDF download button

Requires: D1_percentiles.json in same folder (for color grading)
"""

import streamlit as st
import requests, json, os, io, math, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from datetime import date, timedelta, datetime
warnings.filterwarnings("ignore")

# ===========================================================================
# PAGE CONFIG
# ===========================================================================
st.set_page_config(page_title="TrackMan Pitching Report", layout="wide", page_icon="⚾")

# ===========================================================================
# CREDENTIALS & CONSTANTS
# ===========================================================================
CLIENT_ID = st.secrets["CLIENT_ID"]
CLIENT_SECRET = st.secrets["CLIENT_SECRET"]
BASE_URL  = "https://dataapi.trackmanbaseball.com"
TOKEN_URL = "https://login.trackman.com/connect/token"

STRIKE_CALLS = {"StrikeCalled", "StrikeSwinging", "FoulBallNotFieldable", "InPlay"}
SWING_CALLS  = {"StrikeSwinging", "FoulBallNotFieldable", "InPlay"}
PITCH_COLORS = {
    "Fastball": "#D32F2F", "FourSeamFastBall": "#D32F2F",
    "Sinker": "#E65100", "TwoSeamFastBall": "#E65100",
    "Cutter": "#B8A000", "Slider": "#00897B", "Curveball": "#1565C0",
    "ChangeUp": "#F9A825", "Changeup": "#F9A825",
    "Splitter": "#00796B", "Sweeper": "#7B1FA2", "Other": "#888888",
}
BG_COLOR = "#FFFFFF"
PANEL_COLOR = "#F7F8FA"
GRID_COLOR = "#D5D8DC"
TEXT_COLOR = "#1A1A2E"
ACCENT_COLOR = "#1565C0"
MUTED_TEXT = "#6B7280"
AUTO_CORRECT_PITCHES = True
MIN_CLUSTER_SIZE = 3

# ===========================================================================
# D1 PERCENTILE COLOR GRADING
# ===========================================================================
GRADE_CMAP = LinearSegmentedColormap.from_list("grade", [
    (0.0, "#4575B4"), (0.25, "#91BFDB"), (0.5, "#FFFFFF"),
    (0.75, "#FDB863"), (1.0, "#E66101"),
])

@st.cache_data(ttl=3600)
def load_percentiles():
    paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "D1_percentiles.json"),
        os.path.join(os.path.expanduser("~"), "Downloads", "D1_percentiles.json"),
        "D1_percentiles.json",
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return {}

D1_PCTLS = load_percentiles()

def get_percentile(pitch_type, stat_name, value):
    if not D1_PCTLS: return None
    pt_map = {"FourSeamFastBall": "Fastball", "TwoSeamFastBall": "Sinker", "Changeup": "ChangeUp"}
    pt = pt_map.get(pitch_type, pitch_type)
    pt_data = D1_PCTLS.get(pt, {}).get(stat_name, {})
    pctls = pt_data.get("percentiles", {})
    if not pctls: return None
    pts_list = sorted([(int(k), v) for k, v in pctls.items()], key=lambda x: x[0])
    if value <= pts_list[0][1]: return pts_list[0][0]
    if value >= pts_list[-1][1]: return pts_list[-1][0]
    for i in range(len(pts_list) - 1):
        p0, v0 = pts_list[i]; p1, v1 = pts_list[i + 1]
        if v0 <= value <= v1:
            if v1 == v0: return (p0 + p1) / 2
            return p0 + (value - v0) / (v1 - v0) * (p1 - p0)
        elif v0 >= value >= v1:
            if v0 == v1: return (p0 + p1) / 2
            return p0 + (v0 - value) / (v0 - v1) * (p1 - p0)
    return None

def grade_color(pitch_type, stat_name, value, higher_is_better=True):
    pctile = get_percentile(pitch_type, stat_name, value)
    if pctile is None: return None
    norm = pctile / 100.0
    if not higher_is_better: norm = 1.0 - norm
    return GRADE_CMAP(norm)

# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================
def pc(pt): return PITCH_COLORS.get(pt, "#C8C8C8")

def sg(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict): d = d.get(k, default)
        else: return default
    return d

def sf(v):
    if v is None: return np.nan
    try: return float(v)
    except: return np.nan

def resolve_pt(row):
    t = row.get("TaggedPitchType", ""); a = row.get("AutoPitchType", "")
    if t and t not in ("", "Undefined"): return t
    if a and a not in ("", "Undefined"): return a
    return "Other"

def calc_xwoba(ev, la):
    if pd.isna(ev) or pd.isna(la): return np.nan
    if ev < 50: return 0.05
    if la > 60: return 0.05
    if la > 50: return 0.05 + max(0, (ev - 90)) * 0.01
    if la < -10: return 0.05
    if la < 10:
        if ev >= 100: return 0.50
        if ev >= 90: return 0.30
        if ev >= 80: return 0.20
        return 0.10
    if ev >= 105: base = 1.8
    elif ev >= 100: base = 1.4
    elif ev >= 95: base = 0.9
    elif ev >= 90: base = 0.55
    elif ev >= 85: base = 0.35
    elif ev >= 80: base = 0.25
    elif ev >= 70: base = 0.15
    else: base = 0.08
    la_opt = 24.0
    la_penalty = ((la - la_opt) / 20.0) ** 2
    modifier = max(0.3, 1.0 - la_penalty * 0.5)
    return min(base * modifier, 2.1)

def calc_ip(pd_):
    total = 0
    for (_, _), grp in pd_.groupby(["Inning", "PAofInning"]):
        last = grp.loc[grp["PitchNo"].idxmax()]
        oop = last["OutsOnPlay"]
        if pd.notna(oop) and oop > 0: total += int(oop)
        elif last["KorBB"] == "Strikeout": total += 1
    return f"{total // 3}.{total % 3}"

def calc_pa(pd_): return pd_.groupby(["Inning", "PAofInning"]).ngroups

def calc_er(pd_):
    d = int(pd_["RunsScored"].fillna(0).sum())
    hm = int(((pd_["PlayResult"] == "HomeRun") & (pd_["RunsScored"].fillna(0) == 0)).sum())
    return d + hm

def in_zone(s):
    return (s["PlateLocSide"].notna() & s["PlateLocHeight"].notna() &
            (s["PlateLocSide"].abs() <= 0.95) &
            (s["PlateLocHeight"] >= 1.6) & (s["PlateLocHeight"] <= 3.5))

def auto_correct_pitch_types(pitcher_df):
    if not AUTO_CORRECT_PITCHES: return pitcher_df, 0
    df = pitcher_df.copy(); corrections = 0
    features = ["RelSpeed", "InducedVertBreak", "HorzBreak"]
    for pname in df["Pitcher"].unique():
        pmask = df["Pitcher"] == pname; pdf = df[pmask].copy()
        valid = pdf[features].notna().all(axis=1)
        if valid.sum() < 5: continue
        centroids, stds = {}, {}
        for pt in pdf.loc[valid, "PitchType"].unique():
            if pt in ("Other", "Undefined", ""): continue
            ptmask = (pdf["PitchType"] == pt) & valid
            if ptmask.sum() >= MIN_CLUSTER_SIZE:
                centroids[pt] = pdf.loc[ptmask, features].mean().values
                stds[pt] = pdf.loc[ptmask, features].std().values
                stds[pt] = np.where(stds[pt] < 0.5, 2.0, stds[pt])
        if len(centroids) < 2: continue
        for idx in pdf[valid].index:
            row_vals = pdf.loc[idx, features].values.astype(float)
            tagged = pdf.loc[idx, "PitchType"]
            if tagged not in centroids: continue
            own_max_sd = (np.abs(row_vals - centroids[tagged]) / stds[tagged]).max()
            if own_max_sd > 2.5:
                best_type, best_dist = tagged, own_max_sd
                for other_pt, other_cent in centroids.items():
                    if other_pt == tagged: continue
                    other_max_sd = (np.abs(row_vals - other_cent) / stds[other_pt]).max()
                    if other_max_sd < best_dist and other_max_sd < 1.5:
                        best_dist = other_max_sd; best_type = other_pt
                if best_type != tagged:
                    df.loc[idx, "PitchType"] = best_type; corrections += 1
    return df, corrections

def fmt(s, fn="mean", d=1):
    v = s.dropna()
    if v.empty: return "—"
    r = v.mean() if fn == "mean" else v.max()
    return f"{r:.{d}f}"

# ===========================================================================
# DRAWING FUNCTIONS
# ===========================================================================
def draw_zone(ax, data, title, pts):
    ax.set_facecolor(PANEL_COLOR)
    ax.add_patch(Rectangle((-0.95, 1.6), 1.9, 1.9, fill=False, ec="#333333", lw=1.5, alpha=0.8, zorder=3))
    ax.add_patch(Rectangle((-0.95, 1.6), 1.9, 1.9, fill=True, fc="#E8EDF2", alpha=0.3, zorder=2))
    ax.add_patch(Rectangle((-1.4, 1.2), 2.8, 2.7, fill=False, ec="#AAAAAA", lw=0.7, ls="--", alpha=0.4, zorder=2))
    ax.add_patch(Polygon([(-.708, .15), (.708, .15), (.708, .35), (0, .55), (-.708, .35)],
                         closed=True, fc="none", ec=MUTED_TEXT, lw=.7, alpha=0.4))
    outcome_markers = {
        "BallCalled": ("o", False), "BallinDirt": ("o", False), "BallIntentional": ("o", False),
        "StrikeCalled": ("o", True), "StrikeSwinging": ("X", True),
        "FoulBallNotFieldable": ("^", True), "FoulBallFieldable": ("^", True),
        "InPlay": ("s", True), "HitByPitch": ("D", True),
    }
    for pt in pts:
        s = data[data["PitchType"] == pt]
        if s.empty: continue
        color = pc(pt)
        for call, grp in s.groupby("PitchCall"):
            marker, filled = outcome_markers.get(call, ("o", False))
            x = grp["PlateLocSide"]; y = grp["PlateLocHeight"]
            valid = x.notna() & y.notna()
            if not valid.any(): continue
            if filled:
                ax.scatter(x[valid], y[valid], marker=marker, c=color, s=45, alpha=0.9,
                           edgecolors="black", linewidths=0.3, zorder=5)
            else:
                ax.scatter(x[valid], y[valid], marker=marker, c="none", s=45, alpha=0.9,
                           edgecolors=color, linewidths=1.2, zorder=5)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(0, 5); ax.set_aspect("equal")
    ax.set_title(title, fontsize=9, fontweight="bold", color=TEXT_COLOR, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

def draw_mov(ax, data, pts):
    ax.set_facecolor(PANEL_COLOR)
    ax.axhline(0, color=GRID_COLOR, ls="-", lw=1, zorder=1)
    ax.axvline(0, color=GRID_COLOR, ls="-", lw=1, zorder=1)
    for r in [5, 10, 15, 20]:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ec=GRID_COLOR, lw=0.3, ls="--", alpha=0.3))
    for pt in pts:
        s = data[data["PitchType"] == pt]
        if not s.empty:
            ax.scatter(s["HorzBreak"], s["InducedVertBreak"],
                       c=pc(pt), label=pt, s=40, alpha=.9, edgecolors="black", linewidths=0.3, zorder=5)
    ax.set_xlim(-25, 25); ax.set_ylim(-25, 25)
    ax.set_xlabel("HB (in)", fontsize=7, color=MUTED_TEXT)
    ax.set_ylabel("IVB (in)", fontsize=7, color=MUTED_TEXT)
    ax.set_title("Pitch Movement", fontsize=9, fontweight="bold", color=TEXT_COLOR, pad=6)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.06), ncol=min(len(pts), 5),
              fontsize=6, frameon=False, labelcolor=TEXT_COLOR)
    ax.tick_params(labelsize=6, colors=MUTED_TEXT)
    for sp in ax.spines.values(): sp.set_color(GRID_COLOR)

def draw_release(ax, data, pts):
    ax.set_facecolor(PANEL_COLOR)
    all_rs = data["RelSide"].dropna(); all_rh = data["RelHeight"].dropna()
    for pt in pts:
        valid = data.loc[data["PitchType"] == pt, ["RelSide", "RelHeight"]].dropna()
        if not valid.empty:
            ax.scatter(valid["RelSide"].mean(), valid["RelHeight"].mean(),
                       c=pc(pt), s=80, alpha=0.95, edgecolors="black", linewidths=0.8, zorder=5)
    if not all_rs.empty and not all_rh.empty:
        avg_rs, avg_rh = all_rs.mean(), all_rh.mean()
        ax.axhline(avg_rh, color=ACCENT_COLOR, lw=1, alpha=0.5, zorder=3)
        ax.axvline(avg_rs, color=ACCENT_COLOR, lw=1, alpha=0.5, zorder=3)
        ax.text(avg_rs, avg_rh - 0.3, f"Avg ({avg_rs:.1f}, {avg_rh:.1f})",
                ha="center", va="top", fontsize=6, color=ACCENT_COLOR, family="monospace",
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc=BG_COLOR,
                ec=ACCENT_COLOR, alpha=0.9, lw=0.5))
    ax.axvline(0, color=MUTED_TEXT, lw=0.5, ls="--", alpha=0.3)
    ax.set_xlabel("Release Side (ft)", fontsize=6, color=MUTED_TEXT)
    ax.set_ylabel("Release Height (ft)", fontsize=6, color=MUTED_TEXT)
    ax.set_title("Release Point", fontsize=9, fontweight="bold", color=TEXT_COLOR, pad=6)
    ax.tick_params(labelsize=6, colors=MUTED_TEXT)
    for sp in ax.spines.values(): sp.set_color(GRID_COLOR)
    if not all_rs.empty and not all_rh.empty:
        rs_c, rh_c = all_rs.mean(), all_rh.mean()
        pad = max(all_rs.std(), all_rh.std(), 0.3) * 4 + 0.3
        ax.set_xlim(rs_c - pad, rs_c + pad); ax.set_ylim(rh_c - pad, rh_c + pad)
    ax.set_aspect("equal")

# ===========================================================================
# API FUNCTIONS (cached)
# ===========================================================================
@st.cache_data(ttl=300)
def get_auth_token():
    resp = requests.post(TOKEN_URL, data={
        "grant_type": "client_credentials", "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET})
    if resp.status_code != 200:
        return None
    return resp.json()["access_token"]

def get_headers():
    token = get_auth_token()
    if not token: return None
    return {"Authorization": f"Bearer {token}", "Accept": "application/json", "Content-Type": "application/json"}

@st.cache_data(ttl=3600)
def fetch_sessions(date_from_str, date_to_str):
    headers = get_headers()
    if not headers: return []
    resp = requests.post(f"{BASE_URL}/api/v1/discovery/game/sessions", headers=headers,
                         json={"sessionType": "All", "utcDateFrom": date_from_str, "utcDateTo": date_to_str})
    if resp.status_code != 200: return []
    return resp.json()

@st.cache_data(ttl=3600)
def fetch_game_data(session_id):
    headers = get_headers()
    if not headers: return [], []
    resp = requests.get(f"{BASE_URL}/api/v1/data/game/plays/{session_id}", headers=headers)
    if resp.status_code != 200: return [], []
    plays = resp.json()
    if not isinstance(plays, list): return [], []
    resp = requests.get(f"{BASE_URL}/api/v1/data/game/balls/{session_id}", headers=headers)
    if resp.status_code != 200: return plays, []
    balls = resp.json()
    return plays, balls

def extract_teams_from_sessions(sessions):
    """Get unique team names from sessions."""
    teams = set()
    for s in sessions:
        h = s.get("homeTeam", {}).get("name", "")
        a = s.get("awayTeam", {}).get("name", "")
        if h: teams.add(h)
        if a: teams.add(a)
    return sorted(teams)

def get_sessions_for_team(sessions, team_name):
    """Filter sessions where team_name is home or away."""
    matching = []
    for s in sessions:
        h = s.get("homeTeam", {}).get("name", "")
        a = s.get("awayTeam", {}).get("name", "")
        if team_name.lower() in h.lower() or team_name.lower() in a.lower():
            matching.append(s)
    return matching

def flatten_game(plays_raw, balls_raw):
    """Flatten raw API plays + balls into a merged DataFrame."""
    rows = []
    for p in plays_raw:
        rows.append({
            "PlayID": p.get("playID"),
            "PitchNo": sg(p, "taggerBehavior", "pitchNo"),
            "PAofInning": sg(p, "taggerBehavior", "pAofinning"),
            "PitchofPA": sg(p, "taggerBehavior", "pitchofPA"),
            "Pitcher": sg(p, "pitcher", "name", default=""),
            "PitcherTeam": sg(p, "pitcher", "team", default=""),
            "Batter": sg(p, "batter", "name", default=""),
            "BatterSide": sg(p, "batter", "side", default=""),
            "Inning": sg(p, "gameState", "inning"),
            "TopBottom": sg(p, "gameState", "topBottom"),
            "Outs": sg(p, "gameState", "outs"),
            "Balls": sg(p, "gameState", "balls"),
            "Strikes": sg(p, "gameState", "strikes"),
            "TaggedPitchType": sg(p, "pitchTag", "taggedPitchType", default=""),
            "PitchCall": sg(p, "pitchTag", "pitchCall", default=""),
            "AutoPitchType": sg(p, "pitchTag", "autoPitchType", default=""),
            "KorBB": p.get("korBB", ""),
            "PlayResult": sg(p, "playResult", "playResult", default=""),
            "OutsOnPlay": sf(sg(p, "playResult", "outsOnPlay", default=0)),
            "RunsScored": sf(sg(p, "playResult", "runsScored", default=0)),
        })
    plays_df = pd.DataFrame(rows)
    if plays_df.empty: return plays_df
    plays_df["PitchNo"] = pd.to_numeric(plays_df["PitchNo"], errors="coerce")
    plays_df = plays_df.sort_values("PitchNo").reset_index(drop=True)

    pr, hr_ = [], []
    for b in balls_raw:
        kind = b.get("kind", ""); pid = b.get("playId")
        if kind == "Pitch":
            pr.append({"PlayID": pid,
                "RelSpeed": sf(sg(b, "pitch", "release", "relSpeed")),
                "SpinRate": sf(sg(b, "pitch", "release", "spinRate")),
                "Extension": sf(sg(b, "pitch", "release", "extension")),
                "RelHeight": sf(sg(b, "pitch", "release", "relHeight")),
                "RelSide": sf(sg(b, "pitch", "release", "relSide")),
                "HorzBreak": sf(sg(b, "pitch", "movement", "horzBreak")),
                "InducedVertBreak": sf(sg(b, "pitch", "movement", "inducedVertBreak")),
                "PlateLocHeight": sf(sg(b, "pitch", "location", "plateLocHeight")),
                "PlateLocSide": sf(sg(b, "pitch", "location", "plateLocSide")),
                "VertApprAngle": sf(sg(b, "pitch", "location", "vertApprAngle")),
            })
        elif kind == "Hit":
            hr_.append({"PlayID": pid,
                "ExitSpeed": sf(sg(b, "hit", "launch", "exitSpeed")),
                "LaunchAngle": sf(sg(b, "hit", "launch", "angle")),
            })

    pbdf = pd.DataFrame(pr).drop_duplicates("PlayID", keep="first") if pr else pd.DataFrame()
    hbdf = pd.DataFrame(hr_).drop_duplicates("PlayID", keep="first") if hr_ else pd.DataFrame()

    df = plays_df
    if not pbdf.empty: df = df.merge(pbdf, on="PlayID", how="left")
    else:
        for c in ["RelSpeed","SpinRate","Extension","RelHeight","RelSide",
                   "HorzBreak","InducedVertBreak","PlateLocHeight","PlateLocSide","VertApprAngle"]:
            df[c] = np.nan
    if not hbdf.empty: df = df.merge(hbdf, on="PlayID", how="left")
    else:
        for c in ["ExitSpeed", "LaunchAngle"]: df[c] = np.nan
    df = df.drop_duplicates("PlayID", keep="first")
    return df

def identify_team_code(df, team_name, ht, at):
    """Figure out the team code for our team from play data."""
    if team_name.lower() in ht.lower():
        top_pitchers = df[df["TopBottom"] == "Top"]["PitcherTeam"].value_counts()
        if not top_pitchers.empty: return top_pitchers.index[0]
    elif team_name.lower() in at.lower():
        bot_pitchers = df[df["TopBottom"] == "Bottom"]["PitcherTeam"].value_counts()
        if not bot_pitchers.empty: return bot_pitchers.index[0]
    return None

# ===========================================================================
# GENERATE ONE PITCHER PAGE (returns matplotlib figure)
# ===========================================================================
def generate_pitcher_page(p, pname, gdate, opp):
    """Generate a single pitcher report page. Returns a matplotlib Figure."""
    N = len(p)
    if N == 0: return None

    ip = calc_ip(p); pa = calc_pa(p); er = calc_er(p)
    hits = int(p["PlayResult"].isin(["Single", "Double", "Triple"]).sum())
    hr = int((p["PlayResult"] == "HomeRun").sum())
    k = int((p["KorBB"] == "Strikeout").sum())
    bb = int((p["KorBB"] == "Walk").sum())
    hbp = int((p["PitchCall"] == "HitByPitch").sum())
    spct = round(p["PitchCall"].isin(STRIKE_CALLS).sum() / N * 100, 1)

    wh = p["PitchCall"] == "StrikeSwinging"
    sw = p["PitchCall"].isin(SWING_CALLS)
    iz = p["InZone"]
    ooz = ~iz

    zpct = round(iz.sum() / N * 100, 1)
    wpct = round(wh.sum() / sw.sum() * 100, 1) if sw.sum() else 0
    cpct = round((sw & ooz).sum() / ooz.sum() * 100, 1) if ooz.sum() else 0
    iz_sw = (sw & iz).sum()
    iz_wh_ct = (wh & iz).sum()
    izwp = round(iz_wh_ct / iz_sw * 100, 1) if iz_sw else 0

    pts = p["PitchType"].value_counts().index.tolist()

    fig = plt.figure(figsize=(17, 10), facecolor=BG_COLOR)
    gs = GridSpec(4, 4, figure=fig,
                  height_ratios=[.07, .03, .46, .44],
                  width_ratios=[1, 1, 1, 0.65],
                  hspace=.20, wspace=.20,
                  top=0.95, bottom=0.03, left=0.04, right=0.96)

    # Header
    ax = fig.add_subplot(gs[0, :]); ax.set_facecolor(BG_COLOR); ax.axis("off")
    ax.text(.5, .75, pname.upper(), ha="center", va="center", fontsize=22,
            fontweight="bold", color=TEXT_COLOR, family="monospace")
    ax.text(.5, .25, f"{gdate:%B %d, %Y}   ·   vs {opp}",
            ha="center", va="center", fontsize=11, color=ACCENT_COLOR, family="monospace")

    # Stats bar
    ax = fig.add_subplot(gs[1, :]); ax.set_facecolor(BG_COLOR); ax.axis("off")
    stats_str = (f"IP {ip}   ·   PA {pa}   ·   P {N}   ·   ER {er}   ·   "
                 f"H {hits}   ·   K {k}   ·   BB {bb}   ·   HBP {hbp}   ·   HR {hr}   ·   "
                 f"STR% {spct}%")
    ax.text(.5, .6, stats_str, ha="center", va="center", fontsize=9,
            color=TEXT_COLOR, family="monospace")
    legend_str = "○ Ball    ● Called Strike    ✕ Swinging Strike    ▲ Foul    ■ In Play"
    ax.text(.5, .05, legend_str, ha="center", va="center", fontsize=7,
            color=TEXT_COLOR, family="monospace")

    # Plots
    lhb = p[p["BatterSide"] == "Left"]
    ax_l = fig.add_subplot(gs[2, 0]); draw_zone(ax_l, lhb, f"vs LHB ({len(lhb)})", pts)
    rhb = p[p["BatterSide"] == "Right"]
    ax_r = fig.add_subplot(gs[2, 1]); draw_zone(ax_r, rhb, f"vs RHB ({len(rhb)})", pts)
    ax_m = fig.add_subplot(gs[2, 2]); draw_mov(ax_m, p, pts)
    ax_rp = fig.add_subplot(gs[2, 3]); draw_release(ax_rp, p, pts)

    # Table
    ax_t = fig.add_subplot(gs[3, :]); ax_t.set_facecolor(BG_COLOR); ax_t.axis("off")
    trows = []
    grade_cells = {}
    for ri, pt in enumerate(pts):
        s = p[p["PitchType"] == pt]; n = len(s)
        s_iz = in_zone(s); _sw = s["PitchCall"].isin(SWING_CALLS)
        _wh = s["PitchCall"] == "StrikeSwinging"
        _ooz = ~s_iz
        _ooz_sw = (_sw & _ooz).sum(); _ooz_n = _ooz.sum()
        _iz_sw = (_sw & s_iz).sum(); _iz_wh = (_wh & s_iz).sum()
        iz_whiff_str = f"{_iz_wh / _iz_sw * 100:.1f}%" if _iz_sw else "—"
        _sw_ct = _sw.sum()
        whiff_val = _wh.sum() / _sw_ct * 100 if _sw_ct else None
        whiff_str = f"{whiff_val:.1f}%" if whiff_val is not None else "—"
        chase_val = _ooz_sw / _ooz_n * 100 if _ooz_n else None
        chase_str = f"{chase_val:.1f}%" if chase_val is not None else "—"
        xw = s["xwOBA"].dropna()
        xwoba_val = xw.mean() if not xw.empty else None
        xwoba_str = f"{xwoba_val:.3f}" if xwoba_val is not None else "—"
        avg_velo_raw = s["RelSpeed"].dropna()
        avg_velo_val = avg_velo_raw.mean() if not avg_velo_raw.empty else None
        zone_val = s_iz.sum() / n * 100 if n else None
        zone_str = f"{zone_val:.1f}%" if zone_val is not None else "—"

        trows.append([pt, n, f"{n / N * 100:.1f}%",
                      fmt(s["RelSpeed"]), fmt(s["RelSpeed"], "max"),
                      fmt(s["SpinRate"], d=0),
                      fmt(s["InducedVertBreak"]), fmt(s["HorzBreak"]),
                      fmt(s["Extension"]), fmt(s["RelHeight"]), fmt(s["RelSide"]),
                      fmt(s["VertApprAngle"]),
                      xwoba_str, zone_str, whiff_str, chase_str, iz_whiff_str])

        data_row = ri + 1
        if avg_velo_val is not None:
            grade_cells[(data_row, 2)] = (pt, "velo", avg_velo_val, True)
        if xwoba_val is not None:
            grade_cells[(data_row, 11)] = (pt, "xwoba", xwoba_val, False)
        if zone_val is not None:
            grade_cells[(data_row, 12)] = (pt, "zone_pct", zone_val, True)
        if whiff_val is not None:
            grade_cells[(data_row, 13)] = (pt, "whiff_pct", whiff_val, True)
        if chase_val is not None:
            grade_cells[(data_row, 14)] = (pt, "chase_pct", chase_val, True)

    # All row
    all_sw_ct = sw.sum()
    all_whiff = f"{wh.sum() / all_sw_ct * 100:.1f}%" if all_sw_ct else "0%"
    all_xw = p["xwOBA"].dropna()
    all_xwoba = f"{all_xw.mean():.3f}" if not all_xw.empty else "—"
    trows.append(["All", N, "100%", "—", "—", "—", "—", "—",
                  fmt(p["Extension"]), "—", "—", "—",
                  all_xwoba, f"{zpct}%", all_whiff, f"{cpct}%", f"{izwp}%"])

    cols = ["Count", "Usage%", "Avg\nVelo", "Max\nVelo", "Avg\nSpin",
            "IVB", "HB", "Ext", "RelH", "RelS", "VAA",
            "xwOBA", "Zone%", "Whiff%", "Chase%", "IZ\nWhiff%"]

    tbl = ax_t.table(cellText=[r[1:] for r in trows], rowLabels=[r[0] for r in trows],
                     colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1, 1.4)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID_COLOR); cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#2E2E2E")
            cell.set_text_props(fontweight="bold", color="white", fontfamily="monospace", fontsize=6.5)
        elif col == -1:
            pitch_name = cell.get_text().get_text()
            if pitch_name == "All":
                cell.set_facecolor("#F0F0F0")
                cell.set_text_props(fontweight="bold", color=TEXT_COLOR, fontfamily="monospace")
            else:
                cell.set_facecolor(pc(pitch_name))
                cell.set_text_props(fontweight="bold", color="white", fontfamily="monospace")
        else:
            graded = False
            if (row, col) in grade_cells and row <= len(pts):
                pt_name, stat_name, raw_val, higher_better = grade_cells[(row, col)]
                gc = grade_color(pt_name, stat_name, raw_val, higher_better)
                if gc is not None:
                    cell.set_facecolor(gc)
                    cell.set_text_props(color=TEXT_COLOR, fontfamily="monospace", fontweight="bold")
                    graded = True
            if not graded:
                if row == len(trows):
                    cell.set_facecolor("#F0F0F0")
                    cell.set_text_props(color=TEXT_COLOR, fontweight="bold", fontfamily="monospace")
                elif row % 2 == 0:
                    cell.set_facecolor("#F7F8FA")
                    cell.set_text_props(color=TEXT_COLOR, fontfamily="monospace")
                else:
                    cell.set_facecolor("#FFFFFF")
                    cell.set_text_props(color=TEXT_COLOR, fontfamily="monospace")

    return fig

# ===========================================================================
# STREAMLIT UI
# ===========================================================================
st.title("⚾ TrackMan Pitching Report")

# Sidebar inputs
with st.sidebar:
    st.header("Report Settings")

    # Date range
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("From", value=date.today() - timedelta(days=7))
    with col2:
        date_to = st.date_input("To", value=date.today())

    if date_from > date_to:
        st.error("'From' date must be before 'To' date")
        st.stop()

    date_from_str = f"{date_from}T00:00:00Z"
    date_to_str = f"{date_to + timedelta(days=1)}T00:00:00Z"

    # Auto-fetch sessions when dates change
    date_key = f"{date_from}_{date_to}"
    if st.session_state.get("_date_key") != date_key:
        with st.spinner("Fetching sessions..."):
            sessions = fetch_sessions(date_from_str, date_to_str)
            if sessions:
                st.session_state["sessions"] = sessions
                st.session_state["teams"] = extract_teams_from_sessions(sessions)
                st.session_state["_date_key"] = date_key
                # Clear pitcher data since dates changed
                st.session_state.pop("pitcher_outings", None)
                st.session_state.pop("pitcher_names", None)
                st.session_state.pop("_team_key", None)
                st.success(f"Found {len(sessions)} session(s)")
            else:
                st.error("No sessions found for this date range")

    # Manual refresh button
    if st.button("🔄 Refresh Sessions", use_container_width=True):
        st.cache_data.clear()
        st.session_state.pop("_date_key", None)
        st.session_state.pop("pitcher_outings", None)
        st.session_state.pop("pitcher_names", None)
        st.session_state.pop("_team_key", None)
        st.rerun()

    # Team dropdown — persist selection
    if "teams" in st.session_state and st.session_state["teams"]:
        teams_list = st.session_state["teams"]
        # Restore previous selection if it still exists in the list
        prev_team = st.session_state.get("selected_team", None)
        if prev_team and prev_team in teams_list:
            default_idx = teams_list.index(prev_team)
        else:
            default_idx = 0
        selected_team = st.selectbox("Select Team", teams_list, index=default_idx, key="team_select")

        if selected_team:
            st.session_state["selected_team"] = selected_team

    # D1 percentiles status
    st.divider()
    if D1_PCTLS:
        meta = D1_PCTLS.get("_meta", {})
        st.success(f"D1 Percentiles loaded\n\n{meta.get('sessions_scanned', 0)} sessions · {meta.get('generated', '?')[:10]}")
    else:
        st.warning("No D1_percentiles.json found.\nPlace it next to this script or in Downloads.\nColor grading disabled.")

# Main area — find pitchers and generate
if "sessions" in st.session_state and "selected_team" in st.session_state:
    team_name = st.session_state["selected_team"]
    sessions = st.session_state["sessions"]
    team_sessions = get_sessions_for_team(sessions, team_name)

    if not team_sessions:
        st.warning(f"No games found for {team_name}")
        st.stop()

    st.subheader(f"{team_name} — {len(team_sessions)} game(s) found")

    # Show game list
    game_info = []
    for s in team_sessions:
        ht = s.get("homeTeam", {}).get("name", "")
        at = s.get("awayTeam", {}).get("name", "")
        gd = None
        for field in ["gameDateLocal", "gameDateUtc", "gameDate", "startDateTimeLocal", "startDateTimeUtc"]:
            val = s.get(field, "")
            if val and isinstance(val, str):
                try:
                    gd = datetime.fromisoformat(val.replace("Z", "").split("+")[0]).date()
                    break
                except:
                    pass
        if gd is None: gd = date_from
        game_info.append(f"{gd} — {ht} vs {at}")

    for gi in game_info:
        st.text(f"  📅 {gi}")

    # Debug: show raw date fields from first session (remove after confirming dates work)
    with st.expander("🔧 Debug: Raw session date fields (click to check)"):
        s0 = team_sessions[0]
        date_fields = {}
        for k, v in s0.items():
            kl = k.lower()
            if "date" in kl or "time" in kl or "start" in kl:
                date_fields[k] = v
        st.json(date_fields)

    # Fetch all game data and find pitchers
    if "pitcher_outings" not in st.session_state or st.session_state.get("_team_key") != team_name:
        with st.spinner("Loading game data and finding pitchers..."):
            # pitcher_outings: { "Munn, Drew": [ (df, gdate, opp), (df, gdate2, opp2), ... ] }
            pitcher_outings = {}

            for session in team_sessions:
                sid = session["sessionId"]
                ht = session.get("homeTeam", {}).get("name", "")
                at = session.get("awayTeam", {}).get("name", "")
                if team_name.lower() in ht.lower():
                    opp = at
                elif team_name.lower() in at.lower():
                    opp = ht
                else:
                    opp = "Opponent"

                gdl = session.get("gameDateLocal", "") or session.get("gameDateUtc", "") or session.get("gameDate", "") or ""
                gdate = None
                # Try multiple date sources and formats
                date_candidates = [gdl]
                # Also check other possible fields
                for field in ["startDateTimeLocal", "startDateTimeUtc", "sessionDate"]:
                    val = session.get(field, "")
                    if val: date_candidates.append(val)

                for dc in date_candidates:
                    if gdate: break
                    if not dc or not isinstance(dc, str): continue
                    for fmt_str in [None, "%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S"]:
                        try:
                            if fmt_str is None:
                                gdate = datetime.fromisoformat(dc.replace("Z", "").split("+")[0]).date()
                            else:
                                gdate = datetime.strptime(dc[:len(fmt_str.replace("%","x"))], fmt_str).date()
                            break
                        except:
                            continue

                if gdate is None:
                    # Last resort: try to find any date-like string in session
                    for k, v in session.items():
                        if gdate: break
                        if isinstance(v, str) and len(v) >= 10:
                            try:
                                gdate = datetime.fromisoformat(v.replace("Z", "").split("+")[0]).date()
                            except:
                                pass
                if gdate is None:
                    gdate = date_from  # absolute last fallback

                plays_raw, balls_raw = fetch_game_data(sid)
                if not plays_raw: continue

                df = flatten_game(plays_raw, balls_raw)
                if df.empty: continue

                # Identify team code
                tc = identify_team_code(df, team_name, ht, at)
                if tc:
                    tdf = df[df["PitcherTeam"] == tc].copy()
                else:
                    continue
                if tdf.empty: continue

                tdf["PitchType"] = tdf.apply(resolve_pt, axis=1)
                tdf, _ = auto_correct_pitch_types(tdf)
                tdf["xwOBA"] = tdf.apply(
                    lambda r: calc_xwoba(r["ExitSpeed"], r["LaunchAngle"]) if r["PitchCall"] == "InPlay" else np.nan,
                    axis=1)
                tdf["InZone"] = in_zone(tdf)

                pitchers = (tdf.sort_values("PitchNo").groupby("Pitcher")["PitchNo"].min()
                            .sort_values().index.tolist())

                for pn in pitchers:
                    p_df = tdf[tdf["Pitcher"] == pn].copy().sort_values("PitchNo").reset_index(drop=True)
                    if pn not in pitcher_outings:
                        pitcher_outings[pn] = []
                    pitcher_outings[pn].append((p_df, gdate, opp))

            # Sort outings by date within each pitcher
            for pn in pitcher_outings:
                pitcher_outings[pn].sort(key=lambda x: x[1])

            st.session_state["pitcher_outings"] = pitcher_outings
            st.session_state["pitcher_names"] = sorted(pitcher_outings.keys())
            st.session_state["_team_key"] = team_name

    # Pitcher selection — just unique names, each generates all outings
    if "pitcher_names" in st.session_state and st.session_state["pitcher_names"]:
        st.divider()

        # Show outing counts next to names
        outing_labels = []
        for pn in st.session_state["pitcher_names"]:
            n_outings = len(st.session_state["pitcher_outings"][pn])
            if n_outings > 1:
                outing_labels.append(f"{pn}  ({n_outings} outings)")
            else:
                gdate, opp = st.session_state["pitcher_outings"][pn][0][1], st.session_state["pitcher_outings"][pn][0][2]
                outing_labels.append(f"{pn}  ({gdate} vs {opp})")

        selected_labels = st.multiselect(
            "Select Pitcher(s) to Generate Reports",
            outing_labels,
            default=None
        )

        # Map labels back to pitcher names
        label_to_name = dict(zip(outing_labels, st.session_state["pitcher_names"]))
        selected_names = [label_to_name[lbl] for lbl in selected_labels]

        if selected_names and st.button("⚾ Generate Reports", type="primary", use_container_width=True):
            figures = []
            with st.spinner("Generating reports..."):
                for pname in selected_names:
                    outings = st.session_state["pitcher_outings"][pname]
                    for p_data, gdate, opp in outings:
                        if len(p_data) == 0: continue
                        fig = generate_pitcher_page(p_data, pname, gdate, opp)
                        if fig:
                            figures.append((f"{pname} ({gdate} vs {opp})", fig))

            if figures:
                # Show preview
                st.divider()
                st.subheader("📋 Report Preview")
                for label, fig in figures:
                    st.pyplot(fig, use_container_width=True)
                    st.divider()

                # Build PDF for download
                pdf_buffer = io.BytesIO()
                with PdfPages(pdf_buffer) as pdf:
                    for label, fig in figures:
                        pdf.savefig(fig, bbox_inches="tight", facecolor=BG_COLOR)
                        plt.close(fig)
                pdf_buffer.seek(0)

                # Generate filename
                safe_team = team_name.replace(" ", "")[:15]
                fname = f"PitchingReport_{safe_team}_{date_from}_to_{date_to}.pdf"

                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_buffer,
                    file_name=fname,
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
            else:
                st.error("No reports could be generated")
    elif "pitcher_names" in st.session_state:
        st.warning("No pitchers found for this team in the selected games")
else:
    st.info("👈 Select a date range and click **Fetch Sessions** to get started")


