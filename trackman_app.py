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
from scipy.stats import gaussian_kde
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        korbb = last.get("KorBB", "")
        result = last.get("PlayResult", "")

        if pd.notna(oop) and oop > 0:
            total += int(oop)
        elif korbb == "Strikeout":
            # Dropped 3rd strike: batter reaches — no out recorded
            # If PlayResult indicates batter reached base, skip
            reached = result in ("Single", "Double", "Triple", "HomeRun",
                                 "Error", "FieldersChoice", "CaughtStealing",
                                 "ReachedOnError")
            if not reached:
                total += 1
    return f"{total // 3}.{total % 3}"

def calc_pa(pd_): return pd_.groupby(["Inning", "PAofInning"]).ngroups

def calc_er(pd_):
    """Count earned runs — only from the LAST pitch of each plate appearance to avoid double counting."""
    total = 0
    for (_, _), grp in pd_.groupby(["Inning", "PAofInning"]):
        last = grp.loc[grp["PitchNo"].idxmax()]
        rs = last["RunsScored"]
        if pd.notna(rs) and rs > 0:
            total += int(rs)
        # Also count solo HR where RunsScored might be 0 due to data quirk
        elif last["PlayResult"] == "HomeRun":
            total += 1
    return total

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

    ip = calc_ip(p); pa = calc_pa(p)
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
    stats_str = (f"IP {ip}   ·   PA {pa}   ·   P {N}   ·   "
                 f"H {hits + hr}   ·   K {k}   ·   BB {bb}   ·   HBP {hbp}   ·   HR {hr}   ·   "
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
# SEASON SUMMARY FUNCTIONS
# ===========================================================================
def calc_fip(k, bb, hbp_ct, hr_ct, ip_str):
    """FIP = (13*HR + 3*(BB+HBP) - 2*K) / IP + 3.10"""
    parts = ip_str.split(".")
    ip = int(parts[0]) + int(parts[1]) / 3 if len(parts) == 2 else float(ip_str)
    if ip == 0: return 0.0
    return (13 * hr_ct + 3 * (bb + hbp_ct) - 2 * k) / ip + 3.10

def generate_season_summary(pitcher_name, outings, date_from, date_to):
    """Generate a season summary figure for a pitcher across all outings."""
    all_dfs = []
    for p_df, gdate, opp in outings:
        df_copy = p_df.copy()
        df_copy["_game_date"] = gdate
        df_copy["_opp"] = opp
        all_dfs.append(df_copy)
    p = pd.concat(all_dfs, ignore_index=True)
    N = len(p)
    if N == 0: return None

    # Season counting stats
    total_ip_outs = 0; total_k = 0; total_bb = 0
    total_hbp = 0; total_hr = 0; total_hits = 0; total_pa = 0
    for p_df, gdate, opp in outings:
        ip_s = calc_ip(p_df)
        parts = ip_s.split(".")
        total_ip_outs += int(parts[0]) * 3 + int(parts[1])
        total_k += int((p_df["KorBB"] == "Strikeout").sum())
        total_bb += int((p_df["KorBB"] == "Walk").sum())
        total_hbp += int((p_df["PitchCall"] == "HitByPitch").sum())
        total_hr += int((p_df["PlayResult"] == "HomeRun").sum())
        total_hits += int(p_df["PlayResult"].isin(["Single", "Double", "Triple"]).sum())
        total_pa += calc_pa(p_df)

    ip_full = total_ip_outs // 3; ip_rem = total_ip_outs % 3
    ip_str = f"{ip_full}.{ip_rem}"
    ip_float = ip_full + ip_rem / 3.0
    whip = ((total_bb + total_hits + total_hr) / ip_float) if ip_float > 0 else 0  # H includes HR
    k_pct = (total_k / total_pa * 100) if total_pa > 0 else 0
    bb_pct = (total_bb / total_pa * 100) if total_pa > 0 else 0
    fip = calc_fip(total_k, total_bb, total_hbp, total_hr, ip_str)

    wh = p["PitchCall"] == "StrikeSwinging"
    sw = p["PitchCall"].isin(SWING_CALLS)
    iz = p["InZone"]; ooz = ~iz
    zpct = round(iz.sum() / N * 100, 1)
    wpct = round(wh.sum() / sw.sum() * 100, 1) if sw.sum() else 0
    cpct = round((sw & ooz).sum() / ooz.sum() * 100, 1) if ooz.sum() else 0
    iz_sw = (sw & iz).sum(); iz_wh_ct = (wh & iz).sum()
    izwp = round(iz_wh_ct / iz_sw * 100, 1) if iz_sw else 0
    pts = p["PitchType"].value_counts().index.tolist()

    fig = plt.figure(figsize=(17, 11), facecolor=BG_COLOR)
    gs = GridSpec(4, 3, figure=fig,
                  height_ratios=[.06, .04, .38, .52],
                  width_ratios=[1, 1.2, 1.2],
                  hspace=.25, wspace=.25,
                  top=0.96, bottom=0.03, left=0.05, right=0.96)

    # ---- Row 0: Header ----
    ax = fig.add_subplot(gs[0, :]); ax.set_facecolor(BG_COLOR); ax.axis("off")
    ax.text(.5, .7, pitcher_name.upper(), ha="center", va="center", fontsize=22,
            fontweight="bold", color=TEXT_COLOR, family="monospace")
    ax.text(.5, .1, "Season Pitching Summary", ha="center", va="center",
            fontsize=12, color=ACCENT_COLOR, family="monospace")

    # ---- Row 1: Season stats banner ----
    ax = fig.add_subplot(gs[1, :]); ax.set_facecolor(BG_COLOR); ax.axis("off")

    # Build per-outing detail string
    outing_details = []
    for p_df, gdate, opp in outings:
        o_ip = calc_ip(p_df)
        o_k = int((p_df["KorBB"] == "Strikeout").sum())
        o_bb = int((p_df["KorBB"] == "Walk").sum())
        outing_details.append(f"{gdate} vs {opp}: {o_ip}IP {o_k}K {o_bb}BB")

    banner = (f"IP {ip_str}   ·   FIP {fip:.2f}   ·   WHIP {whip:.2f}   ·   "
              f"K% {k_pct:.1f}%   ·   BB% {bb_pct:.1f}%   ·   K-BB% {k_pct - bb_pct:.1f}%   ·   "
              f"PA {total_pa}   ·   P {N}   ·   H {total_hits + total_hr}   ·   HR {total_hr}   ·   "
              f"K {total_k}   ·   BB {total_bb}   ·   {len(outings)} outing(s)")
    ax.text(.5, .7, banner, ha="center", va="center", fontsize=8.5, color=TEXT_COLOR, family="monospace")
    ax.text(.5, .2, f"{date_from} to {date_to}     |     " + "  /  ".join(outing_details),
            ha="center", va="center", fontsize=6.5, color=MUTED_TEXT, family="monospace")

    # ---- Row 2, Col 0: Velocity Distribution ----
    ax_velo = fig.add_subplot(gs[2, 0]); ax_velo.set_facecolor(PANEL_COLOR)
    pt_velo_sorted = sorted(pts, key=lambda x: p[p["PitchType"] == x]["RelSpeed"].median()
                            if not p[p["PitchType"] == x]["RelSpeed"].dropna().empty else 0, reverse=True)
    for i, pt in enumerate(pt_velo_sorted):
        velos = p.loc[p["PitchType"] == pt, "RelSpeed"].dropna()
        if len(velos) < 3: continue
        try:
            kde = gaussian_kde(velos, bw_method=0.3)
            x_range = np.linspace(velos.min() - 3, velos.max() + 3, 200)
            density = kde(x_range)
            density = density / density.max() * 0.38
            ax_velo.fill_betweenx(i + density, x_range, i - density, alpha=0.7, color=pc(pt))
            ax_velo.plot(x_range, i + density, color="black", lw=0.4)
            ax_velo.plot(x_range, i - density, color="black", lw=0.4)
            med = velos.median()
            ax_velo.plot([med, med], [i - 0.3, i + 0.3], color="black", lw=1, ls="--", alpha=0.5)
        except:
            pass
    ax_velo.set_yticks(range(len(pt_velo_sorted)))
    ax_velo.set_yticklabels(pt_velo_sorted, fontsize=7, fontfamily="monospace")
    ax_velo.set_xlabel("Velocity (mph)", fontsize=7, color=MUTED_TEXT)
    ax_velo.set_title("Velocity Distribution", fontsize=9, fontweight="bold", color=TEXT_COLOR, pad=6)
    ax_velo.tick_params(labelsize=6, colors=MUTED_TEXT)
    for sp in ax_velo.spines.values(): sp.set_color(GRID_COLOR)

    # ---- Row 2, Col 1: Movement (avg dots only) ----
    ax_mov = fig.add_subplot(gs[2, 1]); ax_mov.set_facecolor(PANEL_COLOR)
    ax_mov.axhline(0, color=GRID_COLOR, ls="-", lw=1, zorder=1)
    ax_mov.axvline(0, color=GRID_COLOR, ls="-", lw=1, zorder=1)
    for r in [5, 10, 15, 20]:
        ax_mov.add_patch(plt.Circle((0, 0), r, fill=False, ec=GRID_COLOR, lw=0.3, ls="--", alpha=0.3))
    for pt in pts:
        s = p[p["PitchType"] == pt]
        hb = s["HorzBreak"].dropna(); ivb = s["InducedVertBreak"].dropna()
        if not hb.empty and not ivb.empty:
            ax_mov.scatter(hb.mean(), ivb.mean(), c=pc(pt), label=pt, s=120, alpha=0.95,
                          edgecolors="black", linewidths=1, zorder=5, marker="o")
    ax_mov.set_xlim(-25, 25); ax_mov.set_ylim(-25, 25)
    ax_mov.set_xlabel("HB (in)", fontsize=7, color=MUTED_TEXT)
    ax_mov.set_ylabel("IVB (in)", fontsize=7, color=MUTED_TEXT)
    ax_mov.set_title("Avg Pitch Movement", fontsize=9, fontweight="bold", color=TEXT_COLOR, pad=6)
    ax_mov.legend(loc="upper center", bbox_to_anchor=(.5, -.06), ncol=min(len(pts), 5),
                  fontsize=6, frameon=False, labelcolor=TEXT_COLOR)
    ax_mov.tick_params(labelsize=6, colors=MUTED_TEXT)
    for sp in ax_mov.spines.values(): sp.set_color(GRID_COLOR)

    # ---- Row 2, Col 2: Pitch Usage vs Batter Handedness ----
    ax_usage = fig.add_subplot(gs[2, 2]); ax_usage.set_facecolor(PANEL_COLOR)
    lhb_data = p[p["BatterSide"] == "Left"]
    rhb_data = p[p["BatterSide"] == "Right"]
    lhb_total = len(lhb_data); rhb_total = len(rhb_data)

    bar_pts = sorted(pts, key=lambda x: p[p["PitchType"] == x]["RelSpeed"].median()
                     if not p[p["PitchType"] == x]["RelSpeed"].dropna().empty else 0, reverse=True)
    y_pos = np.arange(len(bar_pts))
    bar_h = 0.35

    for i, pt in enumerate(bar_pts):
        lhb_pct = len(lhb_data[lhb_data["PitchType"] == pt]) / lhb_total * 100 if lhb_total > 0 else 0
        rhb_pct = len(rhb_data[rhb_data["PitchType"] == pt]) / rhb_total * 100 if rhb_total > 0 else 0
        # LHB bars go left (negative), RHB go right (positive)
        ax_usage.barh(i + bar_h / 2, -lhb_pct, bar_h, color=pc(pt), alpha=0.8, edgecolor="black", lw=0.3)
        ax_usage.barh(i - bar_h / 2, rhb_pct, bar_h, color=pc(pt), alpha=0.8, edgecolor="black", lw=0.3)
        if lhb_pct > 2:
            ax_usage.text(-lhb_pct / 2, i + bar_h / 2, f"{lhb_pct:.1f}%", ha="center", va="center",
                         fontsize=6, fontweight="bold", color="white")
        if rhb_pct > 2:
            ax_usage.text(rhb_pct / 2, i - bar_h / 2, f"{rhb_pct:.1f}%", ha="center", va="center",
                         fontsize=6, fontweight="bold", color="white")

    ax_usage.set_yticks(y_pos)
    ax_usage.set_yticklabels(bar_pts, fontsize=7, fontfamily="monospace")
    ax_usage.axvline(0, color="black", lw=1)
    max_pct = max(60, max(
        [len(lhb_data[lhb_data["PitchType"] == pt]) / max(lhb_total, 1) * 100 for pt in bar_pts] +
        [len(rhb_data[rhb_data["PitchType"] == pt]) / max(rhb_total, 1) * 100 for pt in bar_pts]
    ) + 10)
    ax_usage.set_xlim(-max_pct, max_pct)
    ticks = ax_usage.get_xticks()
    ax_usage.set_xticklabels([f"{abs(t):.0f}%" for t in ticks], fontsize=6)
    ax_usage.set_title("Pitch Usage", fontsize=9, fontweight="bold", color=TEXT_COLOR, pad=6)
    ax_usage.text(-max_pct * 0.5, len(bar_pts) + 0.3, f"vs LHB ({lhb_total})", ha="center", fontsize=7,
                 color=MUTED_TEXT, fontweight="bold")
    ax_usage.text(max_pct * 0.5, len(bar_pts) + 0.3, f"vs RHB ({rhb_total})", ha="center", fontsize=7,
                 color=MUTED_TEXT, fontweight="bold")
    ax_usage.tick_params(labelsize=6, colors=MUTED_TEXT)
    for sp in ax_usage.spines.values(): sp.set_color(GRID_COLOR)

    # ---- Row 3: Table ----
    ax_t = fig.add_subplot(gs[3, :]); ax_t.set_facecolor(BG_COLOR); ax_t.axis("off")
    trows = []
    grade_cells = {}
    for ri, pt in enumerate(pts):
        s = p[p["PitchType"] == pt]; n = len(s)
        s_iz = in_zone(s); _sw = s["PitchCall"].isin(SWING_CALLS)
        _wh = s["PitchCall"] == "StrikeSwinging"; _ooz = ~s_iz
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
# HEATMAP FUNCTIONS
# ===========================================================================
# Run value mapping for pitch outcomes (from pitcher perspective, negative = good for pitcher)
RUN_VALUES = {
    "StrikeSwinging": -0.065, "StrikeCalled": -0.038, "FoulBallNotFieldable": -0.025,
    "BallCalled": 0.032, "BallinDirt": 0.032, "BallIntentional": 0.032,
    "HitByPitch": 0.035,
}
# For InPlay: use xwOBA converted to run value scale (approx wOBA/1.15 - league_avg_rv)
# We'll compute per-pitch run values on the fly

def compute_pitch_run_value(row):
    """Estimate per-pitch run value. Negative = good for pitcher."""
    call = row.get("PitchCall", "")
    if call in RUN_VALUES:
        return RUN_VALUES[call]
    if call == "InPlay":
        xw = row.get("xwOBA", np.nan)
        if pd.notna(xw):
            # Convert xwOBA to run value scale: (xwOBA - league_avg) / scale
            return (xw - 0.320) / 1.15  # ~centered around 0
        return 0.0
    return 0.0

def generate_heatmap(p, pitch_type, metric="run_value"):
    """
    Generate a side-by-side heatmap (LHB / RHB) for a given pitch type.
    metric: 'run_value', 'whiff', or 'xwoba'
    Returns a matplotlib Figure.
    """
    sub = p[p["PitchType"] == pitch_type].copy()
    sub = sub[sub["PlateLocSide"].notna() & sub["PlateLocHeight"].notna()]
    if len(sub) < 5:
        return None

    # Compute metric values per pitch
    is_density_only = False
    if metric == "location":
        # Pure density heatmap — no per-pitch value needed
        is_density_only = True
        cmap_name = "YlOrRd"
        title_label = "Pitch Location Density"
        vmin, vmax = 0, 1  # will be normalized
    elif metric == "run_value":
        sub["_val"] = sub.apply(compute_pitch_run_value, axis=1)
        cmap_name = "RdBu_r"  # red=bad for pitcher, blue=good
        title_label = "Run Value"
        vmin, vmax = -0.08, 0.08
    elif metric == "whiff":
        sub["_val"] = (sub["PitchCall"] == "StrikeSwinging").astype(float)
        cmap_name = "YlOrRd"
        title_label = "Whiff Rate"
        vmin, vmax = 0, 0.6
    elif metric == "xwoba":
        # Only BIP pitches have xwOBA; for non-BIP assign outcome-based values
        def xw_val(row):
            if row["PitchCall"] == "InPlay" and pd.notna(row.get("xwOBA")):
                return row["xwOBA"]
            elif row["PitchCall"] == "StrikeSwinging":
                return 0.0
            elif row["PitchCall"] == "StrikeCalled":
                return 0.05
            elif row["PitchCall"] in ("BallCalled", "BallinDirt"):
                return 0.4  # neutral-ish
            return np.nan
        sub["_val"] = sub.apply(xw_val, axis=1)
        sub = sub[sub["_val"].notna()]
        cmap_name = "RdYlBu_r"  # red=high xwOBA (bad for pitcher)
        title_label = "xwOBA"
        vmin, vmax = 0, 0.8
    else:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor=BG_COLOR)

    for idx, (side, label) in enumerate([("Left", "vs LHB"), ("Right", "vs RHB")]):
        ax = axes[idx]
        ax.set_facecolor(PANEL_COLOR)
        side_data = sub[sub["BatterSide"] == side]

        # Draw zone
        ax.add_patch(Rectangle((-0.95, 1.6), 1.9, 1.9, fill=False, ec="black", lw=1.5, zorder=10))
        ax.add_patch(Polygon([(-.708, .15), (.708, .15), (.708, .35), (0, .55), (-.708, .35)],
                             closed=True, fc="#CCCCCC", ec="black", lw=.5, alpha=0.5, zorder=10))

        if len(side_data) >= 5:
            x = side_data["PlateLocSide"].values
            y = side_data["PlateLocHeight"].values

            # Create grid
            xi = np.linspace(-2.5, 2.5, 80)
            yi = np.linspace(-0.5, 5.0, 80)
            Xi, Yi = np.meshgrid(xi, yi)

            try:
                positions = np.vstack([x, y])
                kde = gaussian_kde(positions, bw_method=0.4)
                density = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

                if is_density_only:
                    # Pure density — normalize to 0-1
                    Zi = density / density.max() if density.max() > 0 else density
                    density_thresh = 0.05
                    Zi[Zi < density_thresh] = np.nan
                else:
                    vals = side_data["_val"].values

                    # Weighted value surface
                    Zi = np.zeros_like(Xi)
                    for px, py, pv in zip(x, y, vals):
                        dist2 = (Xi - px) ** 2 + (Yi - py) ** 2
                        weights = np.exp(-dist2 / (2 * 0.3 ** 2))
                        Zi += weights * pv
                    weight_sum = np.zeros_like(Xi)
                    for px, py in zip(x, y):
                        dist2 = (Xi - px) ** 2 + (Yi - py) ** 2
                        weight_sum += np.exp(-dist2 / (2 * 0.3 ** 2))
                    weight_sum[weight_sum == 0] = 1
                    Zi = Zi / weight_sum

                    # Mask low-density areas
                    density_thresh = density.max() * 0.05
                    Zi[density < density_thresh] = np.nan

                im = ax.pcolormesh(Xi, Yi, Zi, cmap=cmap_name, vmin=vmin, vmax=vmax,
                                   shading="gouraud", zorder=1)
            except:
                pass

            # Scatter actual pitch locations
            ax.scatter(x, y, c="black", s=8, alpha=0.5, zorder=6)

        n_side = len(side_data)
        ax.set_title(f"{pitch_type} {label} ({n_side})", fontsize=11, fontweight="bold", color=TEXT_COLOR, pad=8)
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-0.5, 5.0)
        ax.set_xlabel("PlateLocSide", fontsize=8, color=MUTED_TEXT)
        if idx == 0:
            ax.set_ylabel("PlateLocHeight", fontsize=8, color=MUTED_TEXT)
        ax.tick_params(labelsize=7, colors=MUTED_TEXT)
        for sp in ax.spines.values(): sp.set_color(GRID_COLOR)

    # Colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(title_label, fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(f"{title_label} Heatmap — {pitch_type}", fontsize=14, fontweight="bold",
                 color=TEXT_COLOR, y=0.98)

    return fig

# ===========================================================================
# HELPER: Parse game date from session
# ===========================================================================
def parse_session_date(session, fallback_date):
    """Extract game date from session metadata."""
    gdate = None
    for field in ["gameDateLocal", "gameDateUtc", "gameDate", "startDateTimeLocal", "startDateTimeUtc"]:
        val = session.get(field, "")
        if val and isinstance(val, str):
            try:
                gdate = datetime.fromisoformat(val.replace("Z", "").split("+")[0]).date()
                return gdate
            except:
                pass
    # Scan all string fields for a date
    for k, v in session.items():
        if isinstance(v, str) and len(v) >= 10:
            try:
                gdate = datetime.fromisoformat(v.replace("Z", "").split("+")[0]).date()
                return gdate
            except:
                pass
    return fallback_date

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
                st.session_state.pop("pitcher_outings", None)
                st.session_state.pop("pitcher_outings_meta", None)
                st.session_state.pop("pitcher_names", None)
                st.session_state.pop("_team_key", None)
            else:
                st.error("No sessions found for this date range")

    # Manual refresh
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        for k in ["_date_key", "pitcher_outings", "pitcher_outings_meta", "pitcher_names", "_team_key"]:
            st.session_state.pop(k, None)
        st.rerun()

    # Team dropdown — persist selection, searchable
    if "teams" in st.session_state and st.session_state["teams"]:
        teams_list = st.session_state["teams"]
        prev_team = st.session_state.get("selected_team", None)
        if prev_team and prev_team in teams_list:
            default_idx = teams_list.index(prev_team)
        else:
            default_idx = 0
        selected_team = st.selectbox("Select Team", teams_list, index=default_idx)
        if selected_team:
            st.session_state["selected_team"] = selected_team

    # D1 percentiles status
    st.divider()
    if D1_PCTLS:
        meta = D1_PCTLS.get("_meta", {})
        st.success(f"D1 Percentiles loaded\n\n{meta.get('sessions_scanned', 0)} sessions · {meta.get('generated', '?')[:10]}")
    else:
        st.warning("No D1_percentiles.json found.\nColor grading disabled.")

# ===========================================================================
# MAIN CONTENT
# ===========================================================================
if "sessions" in st.session_state and "selected_team" in st.session_state:
    team_name = st.session_state["selected_team"]
    sessions = st.session_state["sessions"]
    team_sessions = get_sessions_for_team(sessions, team_name)

    if not team_sessions:
        st.warning(f"No games found for {team_name}")
        st.stop()

    st.subheader(f"{team_name} — {len(team_sessions)} game(s) found")

    # Show game list
    for s in team_sessions:
        ht = s.get("homeTeam", {}).get("name", "")
        at = s.get("awayTeam", {}).get("name", "")
        gd = parse_session_date(s, date_from)
        st.text(f"  📅 {gd} — {ht} vs {at}")

    # Load pitcher NAMES only (lightweight — just plays, no balls)
    if "pitcher_outings_meta" not in st.session_state or st.session_state.get("_team_key") != team_name:
        with st.spinner("Finding pitchers..."):
            pitcher_meta = {}

            def fetch_plays_only(session):
                """Fetch just plays for one session (no balls)."""
                sid = session["sessionId"]
                headers = get_headers()
                if not headers: return None
                try:
                    resp = requests.get(f"{BASE_URL}/api/v1/data/game/plays/{sid}", headers=headers)
                    if resp.status_code != 200: return None
                    plays_raw = resp.json()
                    if not isinstance(plays_raw, list): return None
                    return (session, plays_raw)
                except:
                    return None

            # Parallel fetch all sessions at once
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(fetch_plays_only, s): s for s in team_sessions}
                for future in as_completed(futures):
                    result = future.result()
                    if result is None: continue
                    session, plays_raw = result

                    ht = session.get("homeTeam", {}).get("name", "")
                    at = session.get("awayTeam", {}).get("name", "")
                    opp = at if team_name.lower() in ht.lower() else (ht if team_name.lower() in at.lower() else "Opponent")
                    gdate = parse_session_date(session, date_from)

                    # Quick pitcher discovery from plays only
                    pitcher_teams = {}
                    for p in plays_raw:
                        pname = sg(p, "pitcher", "name", default="")
                        pteam = sg(p, "pitcher", "team", default="")
                        tb = sg(p, "gameState", "topBottom", default="")
                        if pname and pteam:
                            pitcher_teams[pname] = (pteam, tb)

                    tc = None
                    if team_name.lower() in ht.lower():
                        for pn, (pt, tb) in pitcher_teams.items():
                            if tb == "Top": tc = pt; break
                    elif team_name.lower() in at.lower():
                        for pn, (pt, tb) in pitcher_teams.items():
                            if tb == "Bottom": tc = pt; break

                    if not tc: continue

                    for pn, (pt, tb) in pitcher_teams.items():
                        if pt == tc:
                            if pn not in pitcher_meta:
                                pitcher_meta[pn] = []
                            pitcher_meta[pn].append((session, gdate, opp))

            for pn in pitcher_meta:
                pitcher_meta[pn].sort(key=lambda x: x[1])

            st.session_state["pitcher_outings_meta"] = pitcher_meta
            st.session_state["pitcher_names"] = sorted(pitcher_meta.keys())
            st.session_state["_team_key"] = team_name
            # Clear any previously loaded full data
            st.session_state.pop("pitcher_outings", None)

    # Pitcher selection
    if "pitcher_names" in st.session_state and st.session_state["pitcher_names"]:
        st.divider()

        outing_labels = []
        for pn in st.session_state["pitcher_names"]:
            meta_list = st.session_state["pitcher_outings_meta"][pn]
            n_outings = len(meta_list)
            if n_outings > 1:
                outing_labels.append(f"{pn}  ({n_outings} outings)")
            else:
                gd = meta_list[0][1]
                op = meta_list[0][2]
                outing_labels.append(f"{pn}  ({gd} vs {op})")

        selected_labels = st.multiselect("Select Pitcher(s)", outing_labels, default=None)
        label_to_name = dict(zip(outing_labels, st.session_state["pitcher_names"]))
        selected_names = [label_to_name[lbl] for lbl in selected_labels]

        if selected_names:
            # ========== TABS ==========
            tab_reports, tab_summary, tab_heatmaps, tab_debug = st.tabs(
                ["📄 Game Reports", "📊 Season Summary", "🔥 Heatmaps", "🔧 Debug"])

            def load_full_pitcher_data(pitcher_names_to_load):
                """Load full play+ball data for selected pitchers. Caches in session_state."""
                if "pitcher_outings" not in st.session_state:
                    st.session_state["pitcher_outings"] = {}

                needed = [pn for pn in pitcher_names_to_load if pn not in st.session_state["pitcher_outings"]]
                if not needed:
                    return  # All already loaded

                # Gather which sessions we need
                sessions_to_fetch = {}  # sid -> (session, gdate, opp)
                for pn in needed:
                    for session, gdate, opp in st.session_state["pitcher_outings_meta"][pn]:
                        sid = session["sessionId"]
                        if sid not in sessions_to_fetch:
                            sessions_to_fetch[sid] = (session, gdate, opp)

                # Parallel fetch all sessions
                def fetch_one_session(sid_info):
                    sid, (session, gdate, opp) = sid_info
                    plays_raw, balls_raw = fetch_game_data(sid)
                    if not plays_raw: return None
                    return (sid, session, gdate, opp, plays_raw, balls_raw)

                results = []
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = {executor.submit(fetch_one_session, item): item
                               for item in sessions_to_fetch.items()}
                    for future in as_completed(futures):
                        r = future.result()
                        if r: results.append(r)

                # Process results
                for sid, session, gdate, opp, plays_raw, balls_raw in results:
                    ht = session.get("homeTeam", {}).get("name", "")
                    at = session.get("awayTeam", {}).get("name", "")

                    df = flatten_game(plays_raw, balls_raw)
                    if df.empty: continue
                    tc = identify_team_code(df, team_name, ht, at)
                    if not tc: continue
                    tdf = df[df["PitcherTeam"] == tc].copy()
                    if tdf.empty: continue

                    tdf["PitchType"] = tdf.apply(resolve_pt, axis=1)
                    tdf, _ = auto_correct_pitch_types(tdf)
                    tdf["xwOBA"] = tdf.apply(
                        lambda r: calc_xwoba(r["ExitSpeed"], r["LaunchAngle"]) if r["PitchCall"] == "InPlay" else np.nan, axis=1)
                    tdf["InZone"] = in_zone(tdf)

                    for pn in tdf["Pitcher"].unique():
                        p_df = tdf[tdf["Pitcher"] == pn].copy().sort_values("PitchNo").reset_index(drop=True)
                        if pn in needed:
                            if pn not in st.session_state["pitcher_outings"]:
                                st.session_state["pitcher_outings"][pn] = []
                            st.session_state["pitcher_outings"][pn].append((p_df, gdate, opp))

                # Sort outings by date
                for pn in needed:
                    if pn in st.session_state["pitcher_outings"]:
                        st.session_state["pitcher_outings"][pn].sort(key=lambda x: x[1])

            # ========== TAB 1: GAME REPORTS ==========
            with tab_reports:
                if st.button("⚾ Generate Game Reports", type="primary", use_container_width=True, key="btn_reports"):
                    figures = []
                    with st.spinner("Fetching data & generating reports..."):
                        load_full_pitcher_data(selected_names)
                        for pname in selected_names:
                            if pname not in st.session_state.get("pitcher_outings", {}): continue
                            for p_data, gdate, opp in st.session_state["pitcher_outings"][pname]:
                                if len(p_data) == 0: continue
                                fig = generate_pitcher_page(p_data, pname, gdate, opp)
                                if fig:
                                    figures.append((f"{pname} ({gdate} vs {opp})", fig))

                    if figures:
                        for label, fig in figures:
                            st.pyplot(fig, use_container_width=True)
                            st.divider()

                        pdf_buffer = io.BytesIO()
                        with PdfPages(pdf_buffer) as pdf:
                            for label, fig in figures:
                                pdf.savefig(fig, bbox_inches="tight", facecolor=BG_COLOR)
                                plt.close(fig)
                        pdf_buffer.seek(0)
                        safe_team = team_name.replace(" ", "")[:15]
                        st.download_button("📥 Download Game Reports PDF", data=pdf_buffer,
                                           file_name=f"GameReports_{safe_team}_{date_from}_to_{date_to}.pdf",
                                           mime="application/pdf", type="primary", use_container_width=True)
                    else:
                        st.error("No reports generated")

            # ========== TAB 2: SEASON SUMMARY ==========
            with tab_summary:
                if st.button("📊 Generate Season Summaries", type="primary", use_container_width=True, key="btn_summary"):
                    figures = []
                    with st.spinner("Fetching full 2026 season data..."):
                        # ---- Fetch ALL sessions for the full season ----
                        season_start = "2026-01-01T00:00:00Z"
                        season_end = f"{date.today() + timedelta(days=1)}T00:00:00Z"

                        # Check if we already have season data cached for this team
                        season_cache_key = f"_season_outings_{team_name}"
                        if season_cache_key not in st.session_state:
                            season_sessions = fetch_sessions(season_start, season_end)
                            if season_sessions:
                                season_team_sessions = get_sessions_for_team(season_sessions, team_name)
                            else:
                                season_team_sessions = []

                            if season_team_sessions:
                                # Parallel fetch all game data
                                season_outings = {}  # pitcher_name -> [(df, gdate, opp), ...]

                                def fetch_season_session(session):
                                    sid = session["sessionId"]
                                    plays_raw, balls_raw = fetch_game_data(sid)
                                    if not plays_raw: return None
                                    return (session, plays_raw, balls_raw)

                                results = []
                                with ThreadPoolExecutor(max_workers=8) as executor:
                                    futures = {executor.submit(fetch_season_session, s): s for s in season_team_sessions}
                                    for future in as_completed(futures):
                                        r = future.result()
                                        if r: results.append(r)

                                for session, plays_raw, balls_raw in results:
                                    ht = session.get("homeTeam", {}).get("name", "")
                                    at = session.get("awayTeam", {}).get("name", "")
                                    opp = at if team_name.lower() in ht.lower() else (ht if team_name.lower() in at.lower() else "Opponent")
                                    gdate = parse_session_date(session, date.today())

                                    df = flatten_game(plays_raw, balls_raw)
                                    if df.empty: continue
                                    tc = identify_team_code(df, team_name, ht, at)
                                    if not tc: continue
                                    tdf = df[df["PitcherTeam"] == tc].copy()
                                    if tdf.empty: continue

                                    tdf["PitchType"] = tdf.apply(resolve_pt, axis=1)
                                    tdf, _ = auto_correct_pitch_types(tdf)
                                    tdf["xwOBA"] = tdf.apply(
                                        lambda r: calc_xwoba(r["ExitSpeed"], r["LaunchAngle"]) if r["PitchCall"] == "InPlay" else np.nan, axis=1)
                                    tdf["InZone"] = in_zone(tdf)

                                    for pn in tdf["Pitcher"].unique():
                                        p_df = tdf[tdf["Pitcher"] == pn].copy().sort_values("PitchNo").reset_index(drop=True)
                                        if pn not in season_outings:
                                            season_outings[pn] = []
                                        season_outings[pn].append((p_df, gdate, opp))

                                for pn in season_outings:
                                    season_outings[pn].sort(key=lambda x: x[1])

                                st.session_state[season_cache_key] = season_outings

                        season_outings = st.session_state.get(season_cache_key, {})

                        # Generate summaries for selected pitchers
                        season_start_date = date(2026, 1, 1)
                        season_end_date = date.today()
                        for pname in selected_names:
                            if pname not in season_outings:
                                st.warning(f"No season data found for {pname}")
                                continue
                            outings = season_outings[pname]
                            fig = generate_season_summary(pname, outings, season_start_date, season_end_date)
                            if fig:
                                figures.append((pname, fig))

                    if figures:
                        for label, fig in figures:
                            st.pyplot(fig, use_container_width=True)
                            st.divider()

                        pdf_buffer = io.BytesIO()
                        with PdfPages(pdf_buffer) as pdf:
                            for label, fig in figures:
                                pdf.savefig(fig, bbox_inches="tight", facecolor=BG_COLOR)
                                plt.close(fig)
                        pdf_buffer.seek(0)
                        safe_team = team_name.replace(" ", "")[:15]
                        st.download_button("📥 Download Season Summary PDF", data=pdf_buffer,
                                           file_name=f"SeasonSummary_{safe_team}_2026.pdf",
                                           mime="application/pdf", type="primary", use_container_width=True)
                    else:
                        st.error("No summaries generated")

            # ========== TAB 3: HEATMAPS ==========
            with tab_heatmaps:
                if len(selected_names) == 1:
                    hm_pitcher = selected_names[0]
                else:
                    hm_pitcher = st.selectbox("Select pitcher for heatmaps", selected_names, key="hm_pitcher")

                # Load data button
                if st.button("📂 Load Pitch Data", type="primary", use_container_width=True, key="btn_load_hm"):
                    with st.spinner("Fetching pitch data..."):
                        load_full_pitcher_data([hm_pitcher])
                    st.session_state["_hm_loaded"] = hm_pitcher

                # Show controls if data is loaded
                if (st.session_state.get("_hm_loaded") == hm_pitcher and
                    hm_pitcher in st.session_state.get("pitcher_outings", {})):

                    all_outing_dfs = [p_df for p_df, _, _ in st.session_state["pitcher_outings"][hm_pitcher]]
                    hm_data = pd.concat(all_outing_dfs, ignore_index=True) if all_outing_dfs else pd.DataFrame()

                    if not hm_data.empty:
                        avail_types = hm_data["PitchType"].value_counts()
                        avail_types = avail_types[avail_types >= 5].index.tolist()

                        if avail_types:
                            hm_pitch_type = st.selectbox("Select Pitch Type", avail_types, key="hm_pt_select")
                            hm_metric = st.selectbox("Select Metric", ["Location", "Run Value", "Whiff Rate", "xwOBA"], key="hm_metric_select")

                            metric_map = {"Location": "location", "Run Value": "run_value", "Whiff Rate": "whiff", "xwOBA": "xwoba"}

                            if st.button("🔥 Generate Heatmap", type="primary", use_container_width=True, key="btn_gen_hm"):
                                with st.spinner(f"Generating {hm_metric} heatmap..."):
                                    fig = generate_heatmap(hm_data, hm_pitch_type, metric_map[hm_metric])
                                if fig:
                                    st.session_state["_hm_fig_bytes"] = None
                                    buf = io.BytesIO()
                                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor=BG_COLOR)
                                    buf.seek(0)
                                    st.session_state["_hm_fig_bytes"] = buf.getvalue()
                                    st.session_state["_hm_fig_label"] = f"{hm_metric}_{hm_pitcher}_{hm_pitch_type}"
                                    st.pyplot(fig, use_container_width=True)
                                    plt.close(fig)
                                else:
                                    st.warning(f"Not enough {hm_pitch_type} data for heatmap")

                            # Persist download button across reruns
                            if st.session_state.get("_hm_fig_bytes"):
                                st.download_button(
                                    f"📥 Download Heatmap",
                                    data=st.session_state["_hm_fig_bytes"],
                                    file_name=f"Heatmap_{st.session_state.get('_hm_fig_label', 'heatmap')}.png",
                                    mime="image/png", use_container_width=True)
                        else:
                            st.warning("No pitch types with enough data for heatmaps (need 5+)")
                    else:
                        st.warning("No pitch data available")
                elif st.session_state.get("_hm_loaded") and st.session_state.get("_hm_loaded") != hm_pitcher:
                    st.info("Click **Load Pitch Data** to load data for this pitcher")

            # ========== TAB 4: DEBUG ==========
            with tab_debug:
                st.caption("This tab shows raw PA-level data so you can verify IP/ER calculations. Remove this tab once stats are confirmed correct.")
                if st.button("🔧 Load & Show Debug Data", type="primary", use_container_width=True, key="btn_debug"):
                    with st.spinner("Loading data..."):
                        load_full_pitcher_data(selected_names)

                    for pname in selected_names:
                        if pname not in st.session_state.get("pitcher_outings", {}):
                            st.warning(f"No data for {pname}")
                            continue
                        st.subheader(f"🔍 {pname}")
                        for p_df, gdate, opp in st.session_state["pitcher_outings"][pname]:
                            st.write(f"**{gdate} vs {opp}** — {len(p_df)} pitches")

                            # Show PA-by-PA breakdown
                            pa_rows = []
                            debug_outs = 0
                            reached_results = ("Single", "Double", "Triple", "HomeRun",
                                               "Error", "FieldersChoice", "CaughtStealing",
                                               "ReachedOnError")
                            for (inn, pa_num), grp in p_df.groupby(["Inning", "PAofInning"]):
                                last = grp.loc[grp["PitchNo"].idxmax()]
                                oop = last.get("OutsOnPlay", "")
                                korbb = last.get("KorBB", "")
                                result = last.get("PlayResult", "")
                                out_src = ""
                                if pd.notna(oop) and float(oop) > 0:
                                    debug_outs += int(float(oop))
                                    out_src = f"+{int(float(oop))} (OutsOnPlay)"
                                elif korbb == "Strikeout":
                                    if result in reached_results:
                                        out_src = f"K but reached ({result}) → NO out"
                                    else:
                                        debug_outs += 1
                                        out_src = "+1 (K)"

                                pa_rows.append({
                                    "Inn": inn,
                                    "PA#": pa_num,
                                    "#P": len(grp),
                                    "Batter": last.get("Batter", ""),
                                    "LastCall": last.get("PitchCall", ""),
                                    "KorBB": korbb,
                                    "PlayResult": result,
                                    "OutsOnPlay": oop,
                                    "RunsScored": last.get("RunsScored", ""),
                                    "OutCredit": out_src,
                                })
                            pa_table = pd.DataFrame(pa_rows)
                            st.dataframe(pa_table, use_container_width=True, hide_index=True)

                            # Computed stats
                            ip_s = calc_ip(p_df)
                            er_s = calc_er(p_df)
                            pa_s = calc_pa(p_df)
                            k_s = int((p_df["KorBB"] == "Strikeout").sum())
                            bb_s = int((p_df["KorBB"] == "Walk").sum())
                            st.write(f"**Computed:** IP={ip_s}, ER={er_s}, PA={pa_s}, K={k_s}, BB={bb_s}, Debug outs total={debug_outs}")

                            # Show raw OutsOnPlay and RunsScored distributions
                            st.write("**Raw OutsOnPlay values in data:**", p_df["OutsOnPlay"].value_counts().to_dict())
                            st.write("**Raw RunsScored values in data:**", p_df["RunsScored"].value_counts().to_dict())
                            st.divider()

    elif "pitcher_names" in st.session_state:
        st.warning("No pitchers found for this team in the selected games")
else:
    st.info("👈 Select a date range to get started — sessions load automatically")

# ===========================================================================
# STANDALONE PITCH MIX SECTION
# ===========================================================================
st.divider()
st.title("🎯 Pitch Mix Analysis")

if "selected_team" in st.session_state:
    pm_team = st.session_state["selected_team"]

    # Date range
    pm_col1, pm_col2 = st.columns(2)
    with pm_col1:
        pm_from = st.date_input("From", value=date(2026, 1, 1), key="pm_from")
    with pm_col2:
        pm_to = st.date_input("To", value=date.today(), key="pm_to")

    # Load all pitcher names for this team + date range
    pm_cache_key = f"_pm_data_{pm_team}_{pm_from}_{pm_to}"

    if st.button("📂 Load Pitchers", type="secondary", use_container_width=True, key="btn_pm_load"):
        with st.spinner("Fetching data for pitch mix..."):
            pm_start = f"{pm_from}T00:00:00Z"
            pm_end = f"{pm_to + timedelta(days=1)}T00:00:00Z"

            pm_sessions = fetch_sessions(pm_start, pm_end)
            pm_team_sessions = get_sessions_for_team(pm_sessions, pm_team) if pm_sessions else []

            pm_all_outings = {}
            if pm_team_sessions:
                def fetch_pm_session(session):
                    sid = session["sessionId"]
                    plays_raw, balls_raw = fetch_game_data(sid)
                    if not plays_raw: return None
                    return (session, plays_raw, balls_raw)

                results = []
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = {executor.submit(fetch_pm_session, s): s for s in pm_team_sessions}
                    for future in as_completed(futures):
                        r = future.result()
                        if r: results.append(r)

                for session, plays_raw, balls_raw in results:
                    ht = session.get("homeTeam", {}).get("name", "")
                    at = session.get("awayTeam", {}).get("name", "")
                    opp = at if pm_team.lower() in ht.lower() else (ht if pm_team.lower() in at.lower() else "Opponent")
                    gdate = parse_session_date(session, date.today())

                    df = flatten_game(plays_raw, balls_raw)
                    if df.empty: continue
                    tc = identify_team_code(df, pm_team, ht, at)
                    if not tc: continue
                    tdf = df[df["PitcherTeam"] == tc].copy()
                    if tdf.empty: continue

                    tdf["PitchType"] = tdf.apply(resolve_pt, axis=1)
                    tdf, _ = auto_correct_pitch_types(tdf)
                    tdf["InZone"] = in_zone(tdf)

                    for pn in tdf["Pitcher"].unique():
                        p_df = tdf[tdf["Pitcher"] == pn].copy().sort_values("PitchNo").reset_index(drop=True)
                        if pn not in pm_all_outings:
                            pm_all_outings[pn] = []
                        pm_all_outings[pn].append((p_df, gdate, opp))

            st.session_state[pm_cache_key] = pm_all_outings
            st.session_state["_pm_pitchers"] = sorted(pm_all_outings.keys())
            st.session_state["_pm_active_key"] = pm_cache_key

    # Show controls if data loaded
    if st.session_state.get("_pm_active_key") == pm_cache_key and "_pm_pitchers" in st.session_state:
        pm_outings = st.session_state.get(pm_cache_key, {})

        if not st.session_state["_pm_pitchers"]:
            st.warning("No pitchers found")
        else:
            pm_pitcher = st.selectbox("Select Pitcher", st.session_state["_pm_pitchers"], key="pm_pitcher")

            fc1, fc2 = st.columns(2)
            with fc1:
                pm_hand = st.selectbox("Batter Hand", ["All", "vs RHH", "vs LHH"], key="pm_hand")
            with fc2:
                pm_tto = st.selectbox("Time Through Order",
                                      ["All", "1st Time", "2nd Time", "3rd Time", "4th+ Time"], key="pm_tto")

            if st.button("🎯 Generate Pitch Mix", type="primary", use_container_width=True, key="btn_pitchmix"):
                if pm_pitcher not in pm_outings or not pm_outings[pm_pitcher]:
                    st.warning(f"No data found for {pm_pitcher}")
                else:
                    # Combine all outings
                    all_dfs = []
                    for p_df, gdate, opp in pm_outings[pm_pitcher]:
                        df_copy = p_df.copy()
                        df_copy["_game_date"] = gdate
                        df_copy["_opp"] = opp
                        all_dfs.append(df_copy)
                    pitch_data = pd.concat(all_dfs, ignore_index=True)

                    # Compute TTO per game
                    pitch_data["_TTO"] = 0
                    for gd in pitch_data["_game_date"].unique():
                        game_mask = pitch_data["_game_date"] == gd
                        game_df = pitch_data[game_mask].copy()
                        pa_order = game_df.groupby(["Inning", "PAofInning"]).first().reset_index()
                        pa_order = pa_order.sort_values("PitchNo")
                        batter_count = {}
                        pa_tto = {}
                        for _, pa_row in pa_order.iterrows():
                            batter = pa_row["Batter"]
                            if batter not in batter_count:
                                batter_count[batter] = 0
                            batter_count[batter] += 1
                            pa_key = (pa_row["Inning"], pa_row["PAofInning"])
                            pa_tto[pa_key] = batter_count[batter]

                        for idx in pitch_data[game_mask].index:
                            row = pitch_data.loc[idx]
                            pa_key = (row["Inning"], row["PAofInning"])
                            pitch_data.loc[idx, "_TTO"] = pa_tto.get(pa_key, 1)

                    # Apply filters
                    filtered = pitch_data.copy()
                    if pm_hand == "vs RHH":
                        filtered = filtered[filtered["BatterSide"] == "Right"]
                    elif pm_hand == "vs LHH":
                        filtered = filtered[filtered["BatterSide"] == "Left"]

                    if pm_tto == "1st Time":
                        filtered = filtered[filtered["_TTO"] == 1]
                    elif pm_tto == "2nd Time":
                        filtered = filtered[filtered["_TTO"] == 2]
                    elif pm_tto == "3rd Time":
                        filtered = filtered[filtered["_TTO"] == 3]
                    elif pm_tto == "4th+ Time":
                        filtered = filtered[filtered["_TTO"] >= 4]

                    if len(filtered) == 0:
                        st.warning("No pitches match these filters")
                    else:
                        N = len(filtered)
                        pts = filtered["PitchType"].value_counts().index.tolist()

                        def get_count_cat(balls, strikes):
                            cats = []
                            if (balls, strikes) in [(0, 0), (1, 0), (0, 1)]:
                                cats.append("early")
                            if strikes > balls and strikes >= 1:
                                cats.append("ahead")
                            if balls > strikes:
                                cats.append("behind")
                            if strikes < 2:
                                cats.append("pre2k")
                            if strikes == 2:
                                cats.append("twok")
                            return cats

                        filtered["_count_cats"] = filtered.apply(
                            lambda r: get_count_cat(
                                int(r["Balls"]) if pd.notna(r["Balls"]) else 0,
                                int(r["Strikes"]) if pd.notna(r["Strikes"]) else 0
                            ), axis=1)

                        situations = {
                            "All Counts": filtered,
                            "Early Count": filtered[filtered["_count_cats"].apply(lambda x: "early" in x)],
                            "Pitcher Ahead": filtered[filtered["_count_cats"].apply(lambda x: "ahead" in x)],
                            "Pitcher Behind": filtered[filtered["_count_cats"].apply(lambda x: "behind" in x)],
                            "Pre Two Strikes": filtered[filtered["_count_cats"].apply(lambda x: "pre2k" in x)],
                            "Two Strikes": filtered[filtered["_count_cats"].apply(lambda x: "twok" in x)],
                        }

                        # Build table data
                        table_data = []
                        all_counts_pcts = {}
                        for pt in pts:
                            all_pct = len(filtered[filtered["PitchType"] == pt]) / N * 100 if N > 0 else 0
                            all_counts_pcts[pt] = all_pct

                        for pt in pts:
                            row_d = {"Pitch Type": pt}
                            for sit_name, sit_df in situations.items():
                                sit_n = len(sit_df)
                                pct = len(sit_df[sit_df["PitchType"] == pt]) / sit_n * 100 if sit_n > 0 else 0
                                row_d[sit_name] = pct
                            table_data.append(row_d)

                        # Header
                        hand_label = pm_hand if pm_hand != "All" else "vs All"
                        tto_label = pm_tto if pm_tto != "All" else "All ABs"
                        st.markdown(f"### {pm_pitcher}")
                        st.caption(f"Pitch Mix {hand_label} · {tto_label} · {pm_from} to {pm_to} · {N} pitches")

                        # Dark-themed table
                        fig, ax = plt.subplots(figsize=(14, max(2.5, 0.6 * len(pts) + 1.2)),
                                               facecolor="#1a1d23")
                        ax.axis("off")

                        col_labels = list(situations.keys())
                        cell_text = []
                        for rd in table_data:
                            cell_text.append([f"{rd[c]:.0f}%" for c in col_labels])

                        tbl = ax.table(
                            cellText=cell_text,
                            rowLabels=[r["Pitch Type"] for r in table_data],
                            colLabels=[c.upper().replace(" ", "\n") for c in col_labels],
                            loc="center", cellLoc="center"
                        )
                        tbl.auto_set_font_size(False)
                        tbl.set_fontsize(10)
                        tbl.scale(1, 2.0)

                        for (row, col), cell in tbl.get_celld().items():
                            cell.set_edgecolor("#2a2d35")
                            cell.set_linewidth(0.5)

                            if row == 0:
                                cell.set_facecolor("#2a2d35")
                                cell.set_text_props(fontweight="bold", color="#8890a0",
                                                    fontfamily="monospace", fontsize=8)
                            elif col == -1:
                                pt_name = cell.get_text().get_text()
                                cell.set_facecolor("#1a1d23")
                                cell.get_text().set_text(f"● {pt_name}")
                                cell.get_text().set_color(pc(pt_name))
                                cell.set_text_props(fontweight="bold", fontfamily="monospace", fontsize=10)
                            else:
                                cell.set_facecolor("#1e2128")
                                pt_name = table_data[row - 1]["Pitch Type"]
                                col_name = col_labels[col]
                                val = table_data[row - 1][col_name]
                                base = all_counts_pcts.get(pt_name, 0)
                                diff = val - base

                                if col_name != "All Counts" and abs(diff) >= 5:
                                    if diff >= 5:
                                        cell.set_facecolor("#1a3a2a")
                                        cell.set_text_props(color="#4ade80", fontweight="bold",
                                                            fontfamily="monospace", fontsize=10)
                                    else:
                                        cell.set_facecolor("#3a1a1a")
                                        cell.set_text_props(color="#f87171", fontweight="bold",
                                                            fontfamily="monospace", fontsize=10)
                                else:
                                    cell.set_text_props(color="white", fontfamily="monospace",
                                                        fontsize=10)

                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)

                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="#1a1d23")
                        buf.seek(0)
                        st.download_button("📥 Download Pitch Mix", data=buf,
                                           file_name=f"PitchMix_{pm_pitcher}_{pm_hand}_{pm_tto}.png",
                                           mime="image/png", use_container_width=True)
                        plt.close(fig)

                        with st.expander("📋 Sample sizes per situation"):
                            size_data = {name: len(sdf) for name, sdf in situations.items()}
                            st.write(size_data)
else:
    st.info("Select a team above to use Pitch Mix")

