"""
fetch_data.py
=============
Runs once daily to pull all 2026 D1 college baseball data from the
TrackMan API and write it to data/pitches.parquet.

The Streamlit app (trackman_app.py) reads from that file and never
touches the API directly.

Setup (Windows Task Scheduler):
  Program: python
  Arguments: C:\\path\\to\\fetch_data.py
  Trigger: Daily at 6:00 AM
  Settings: Check "Wake the computer to run this task"

Manual run:
  python fetch_data.py

Output:
  data/pitches.parquet       — all pitches with metadata
  data/sessions.parquet      — session/game metadata
  data/last_updated.json     — timestamp of last successful run
"""

import requests, json, os, sys, time
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===========================================================================
# CREDENTIALS
# ===========================================================================
CLIENT_ID = st.secrets["CLIENT_ID"]
CLIENT_SECRET = st.secrets["CLIENT_SECRET"]
BASE_URL      = "https://dataapi.trackmanbaseball.com"
TOKEN_URL     = "https://login.trackman.com/connect/token"

SEASON_START  = date(2026, 2, 1)   # adjust each year
DATA_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ===========================================================================
# AUTH
# ===========================================================================
_token_cache = {"token": None, "expires": 0}

def get_token():
    if time.time() < _token_cache["expires"] - 60:
        return _token_cache["token"]
    resp = requests.post(TOKEN_URL, data={
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }, timeout=30)
    if resp.status_code != 200:
        print(f"  Auth failed: {resp.status_code} — {resp.text[:200]}")
        return None
    data = resp.json()
    _token_cache["token"] = data["access_token"]
    _token_cache["expires"] = time.time() + data.get("expires_in", 3600)
    return _token_cache["token"]

def get_headers():
    token = get_token()
    if not token: return None
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

# ===========================================================================
# SAFE HELPERS
# ===========================================================================
def sg(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d

def sf(v):
    if v is None: return np.nan
    try: return float(v)
    except: return np.nan

# ===========================================================================
# D1 FILTER
# ===========================================================================
def is_d1_session(s):
    lv = s.get("level", "")
    lv_text = " ".join(str(v) for v in lv.values()).upper() if isinstance(lv, dict) else str(lv).upper()
    return any(kw in lv_text for kw in ["D1", "NCAA-D1", "DIVISION 1", "DIV1", "DIV-1"])

# ===========================================================================
# API FETCHERS  (polite — 1.5s sleep, retries on failure)
# ===========================================================================
def fetch_sessions(date_from_str, date_to_str):
    headers = get_headers()
    if not headers: return []
    resp = requests.post(
        f"{BASE_URL}/api/v1/discovery/game/sessions",
        headers=headers,
        json={"sessionType": "All", "utcDateFrom": date_from_str, "utcDateTo": date_to_str},
        timeout=60,
    )
    if not resp.ok:
        print(f"  Session fetch failed: {resp.status_code} — {resp.text[:100]}")
        return []
    data = resp.json()
    return data if isinstance(data, list) else data.get("sessions", [])

def fetch_game_data(session_id):
    for attempt in range(4):
        try:
            headers = get_headers()
            if not headers: return [], []

            resp = requests.get(
                f"{BASE_URL}/api/v1/data/game/plays/{session_id}",
                headers=headers, timeout=30,
            )
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 30))
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code != 200: return [], []
            plays = resp.json()
            if not isinstance(plays, list): return [], []

            resp = requests.get(
                f"{BASE_URL}/api/v1/data/game/balls/{session_id}",
                headers=headers, timeout=30,
            )
            if resp.status_code != 200: return plays, []
            return plays, resp.json()

        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"  Connection error (attempt {attempt+1}/4), retrying in {wait}s: {e}")
            time.sleep(wait)
    return [], []

# ===========================================================================
# DATA FLATTENING  (matches trackman_app.py exactly)
# ===========================================================================
def flatten_game(plays_raw, balls_raw, session):
    ht = session.get("homeTeam", {}).get("name", "")
    at = session.get("awayTeam", {}).get("name", "")
    game_date = session.get("gameDateLocal", session.get("gameDateUtc", ""))[:10] if session.get("gameDateLocal") or session.get("gameDateUtc") else ""
    session_id = session.get("sessionId", "")

    rows = []
    for p in plays_raw:
        rows.append({
            "PlayID":          p.get("playID"),
            "SessionID":       session_id,
            "GameDate":        game_date,
            "HomeTeam":        ht,
            "AwayTeam":        at,
            "PitchNo":         sg(p, "taggerBehavior", "pitchNo"),
            "PAofInning":      sg(p, "taggerBehavior", "pAofinning"),
            "PitchofPA":       sg(p, "taggerBehavior", "pitchofPA"),
            "Pitcher":         sg(p, "pitcher", "name", default=""),
            "PitcherTeam":     sg(p, "pitcher", "team", default=""),
            "PitcherThrows":   sg(p, "pitcher", "throwHand", default=""),
            "Batter":          sg(p, "batter", "name", default=""),
            "BatterSide":      sg(p, "batter", "side", default=""),
            "Inning":          sg(p, "gameState", "inning"),
            "TopBottom":       sg(p, "gameState", "topBottom"),
            "Outs":            sg(p, "gameState", "outs"),
            "Balls":           sg(p, "gameState", "balls"),
            "Strikes":         sg(p, "gameState", "strikes"),
            "TaggedPitchType": sg(p, "pitchTag", "taggedPitchType", default=""),
            "AutoPitchType":   sg(p, "pitchTag", "autoPitchType", default=""),
            "PitchCall":       sg(p, "pitchTag", "pitchCall", default=""),
            "KorBB":           p.get("korBB", ""),
            "PlayResult":      sg(p, "playResult", "playResult", default=""),
            "OutsOnPlay":      sf(sg(p, "playResult", "outsOnPlay", default=0)),
            "RunsScored":      sf(sg(p, "playResult", "runsScored", default=0)),
        })

    plays_df = pd.DataFrame(rows)
    if plays_df.empty: return plays_df
    plays_df["PitchNo"] = pd.to_numeric(plays_df["PitchNo"], errors="coerce")
    plays_df = plays_df.sort_values("PitchNo").reset_index(drop=True)

    pr, hr_ = [], []
    for b in balls_raw:
        kind = b.get("kind", ""); pid = b.get("playId")
        if kind == "Pitch":
            pr.append({
                "PlayID":           pid,
                "RelSpeed":         sf(sg(b, "pitch", "release", "relSpeed")),
                "SpinRate":         sf(sg(b, "pitch", "release", "spinRate")),
                "Extension":        sf(sg(b, "pitch", "release", "extension")),
                "RelHeight":        sf(sg(b, "pitch", "release", "relHeight")),
                "RelSide":          sf(sg(b, "pitch", "release", "relSide")),
                "HorzBreak":        sf(sg(b, "pitch", "movement", "horzBreak")),
                "InducedVertBreak": sf(sg(b, "pitch", "movement", "inducedVertBreak")),
                "PlateLocHeight":   sf(sg(b, "pitch", "location", "plateLocHeight")),
                "PlateLocSide":     sf(sg(b, "pitch", "location", "plateLocSide")),
                "VertApprAngle":    sf(sg(b, "pitch", "location", "vertApprAngle")),
            })
        elif kind == "Hit":
            hr_.append({
                "PlayID":      pid,
                "ExitSpeed":   sf(sg(b, "hit", "launch", "exitSpeed")),
                "LaunchAngle": sf(sg(b, "hit", "launch", "angle")),
            })

    pbdf = pd.DataFrame(pr).drop_duplicates("PlayID", keep="first") if pr else pd.DataFrame()
    hbdf = pd.DataFrame(hr_).drop_duplicates("PlayID", keep="first") if hr_ else pd.DataFrame()

    df = plays_df
    if not pbdf.empty:
        df = df.merge(pbdf, on="PlayID", how="left")
    else:
        for c in ["RelSpeed","SpinRate","Extension","RelHeight","RelSide",
                   "HorzBreak","InducedVertBreak","PlateLocHeight","PlateLocSide","VertApprAngle"]:
            df[c] = np.nan
    if not hbdf.empty:
        df = df.merge(hbdf, on="PlayID", how="left")
    else:
        for c in ["ExitSpeed", "LaunchAngle"]:
            df[c] = np.nan

    return df.drop_duplicates("PlayID", keep="first")

def resolve_pt(row):
    t = row.get("TaggedPitchType", "") or ""
    a = row.get("AutoPitchType", "") or ""
    if t in ("Fastball", "FourSeamFastBall"):    return "Fastball"
    if t in ("Sinker", "TwoSeamFastBall"):        return "Sinker"
    if t == "Cutter":                              return "Cutter"
    if t in ("Slider", "Sweeper"):                return "Slider"
    if t in ("Curveball", "CurveBall"):           return "Curveball"
    if t in ("ChangeUp", "Changeup", "Splitter"): return "ChangeUp"
    if a in ("Fastball", "FourSeamFastBall"):    return "Fastball"
    if a in ("Sinker", "TwoSeamFastBall"):        return "Sinker"
    if a == "Cutter":                              return "Cutter"
    if a in ("Slider", "Sweeper"):                return "Slider"
    if a in ("Curveball", "CurveBall"):           return "Curveball"
    if a in ("ChangeUp", "Changeup", "Splitter"): return "ChangeUp"
    return "Other"

# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 60)
    print(f"FETCH DATA — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    # ---- Step 1: Load existing data to find already-fetched sessions ----
    pitches_path  = os.path.join(DATA_DIR, "pitches.parquet")
    sessions_path = os.path.join(DATA_DIR, "sessions.parquet")

    existing_session_ids = set()
    existing_df = None

    if os.path.exists(pitches_path):
        existing_df = pd.read_parquet(pitches_path)
        existing_session_ids = set(existing_df["SessionID"].dropna().unique())
        print(f"\n  Existing data: {len(existing_df)} pitches, {len(existing_session_ids)} sessions")
    else:
        print("\n  No existing data — full season fetch")

    # ---- Step 2: Fetch all sessions for the 2026 season ----
    print(f"\n[1/3] Fetching session list from {SEASON_START} → today...")
    all_sessions = []
    chunk_start = SEASON_START
    today = date.today()

    while chunk_start <= today:
        chunk_end = min(chunk_start + timedelta(days=14), today + timedelta(days=1))
        sessions = fetch_sessions(f"{chunk_start}T00:00:00Z", f"{chunk_end}T00:00:00Z")
        if sessions:
            all_sessions.extend(sessions)
            print(f"  {chunk_start} → {chunk_end}: {len(sessions)} sessions")
        time.sleep(1.0)
        chunk_start = chunk_end

    # Deduplicate
    seen, unique = set(), []
    for s in all_sessions:
        sid = s.get("sessionId", "")
        if sid and sid not in seen:
            seen.add(sid); unique.append(s)

    # Filter D1
    d1 = [s for s in unique if is_d1_session(s)]
    if not d1:
        print(f"\n  WARNING: D1 filter returned 0 — using all {len(unique)} sessions")
        d1 = unique
    else:
        print(f"\n  D1 sessions: {len(d1)} of {len(unique)} total")

    # Only fetch sessions we don't already have
    new_sessions = [s for s in d1 if s.get("sessionId") not in existing_session_ids]
    print(f"  New sessions to fetch: {len(new_sessions)}")

    if not new_sessions:
        print("\n  Already up to date — nothing new to fetch.")
        _write_timestamp()
        return

    # ---- Step 3: Fetch pitch data for new sessions ----
    print(f"\n[2/3] Fetching pitch data ({len(new_sessions)} games)...")
    new_dfs = []

    for i, session in enumerate(new_sessions):
        sid = session.get("sessionId")
        plays, balls = fetch_game_data(sid)
        if plays:
            df = flatten_game(plays, balls, session)
            if not df.empty:
                df["PitchType"] = df.apply(resolve_pt, axis=1)
                new_dfs.append(df)

        time.sleep(1.5)  # polite — no rush at 6 AM

        if (i + 1) % 10 == 0 or (i + 1) == len(new_sessions):
            print(f"  {i+1}/{len(new_sessions)} sessions — {sum(len(d) for d in new_dfs)} new pitches")

    if not new_dfs:
        print("\n  No new pitch data found.")
        _write_timestamp()
        return

    # ---- Step 4: Merge with existing and save ----
    print(f"\n[3/3] Saving data...")
    new_df = pd.concat(new_dfs, ignore_index=True)

    if existing_df is not None:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates("PlayID", keep="first")
    else:
        combined = new_df

    combined.to_parquet(pitches_path, index=False)
    print(f"  pitches.parquet — {len(combined)} total pitches")

    # Save session metadata
    session_rows = []
    for s in d1:
        session_rows.append({
            "SessionID": s.get("sessionId"),
            "GameDate":  (s.get("gameDateLocal") or s.get("gameDateUtc") or "")[:10],
            "HomeTeam":  s.get("homeTeam", {}).get("name", ""),
            "AwayTeam":  s.get("awayTeam", {}).get("name", ""),
            "Level":     str(s.get("level", "")),
        })
    pd.DataFrame(session_rows).to_parquet(sessions_path, index=False)
    print(f"  sessions.parquet — {len(session_rows)} sessions")

    _write_timestamp()

    print(f"\n  Done! Added {len(new_df)} pitches from {len(new_dfs)} new games.")
    print(f"  Total dataset: {len(combined)} pitches")

def _write_timestamp():
    ts_path = os.path.join(DATA_DIR, "last_updated.json")
    with open(ts_path, "w") as f:
        json.dump({
            "last_updated": datetime.now().isoformat(),
            "last_updated_date": date.today().isoformat(),
        }, f, indent=2)
    print(f"  Timestamp written → {ts_path}")

if __name__ == "__main__":

    main()
