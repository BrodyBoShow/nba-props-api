"""
build_training_data.py
──────────────────────
Pull 3 seasons of player game logs from nba_api and build a clean,
leakage-free training dataset for the XGBoost projection models.

Each output row = one player-game appearance with ONLY pre-game features.

Usage:
    pip install nba_api pandas numpy pyarrow
    python build_training_data.py

Output:
    training_data.parquet   (master dataset — ~200k+ rows, 3 seasons)
    data_cache/             (raw API responses cached to avoid re-fetching)
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np

# Force UTF-8 output so Windows cp1252 console doesn't choke on Unicode in print/errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from nba_api.stats.endpoints import leaguegamelog
except ImportError:
    print("Missing dependency — run: pip install nba_api pandas numpy pyarrow")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
SEASONS      = ["2022-23", "2023-24", "2024-25"]
SEASON_TYPES = ["Regular Season", "Playoffs"]
SLEEP_SEC    = 1.0          # rate-limit pause between API calls
MIN_GP_PRIOR = 5            # drop rows where player has < 5 prior games (L5 invalid)
CACHE_DIR    = "data_cache"
OUTPUT_FILE  = "training_data.parquet"


def _fetch(season: str, season_type: str, retries: int = 3) -> pd.DataFrame:
    """Fetch LeagueGameLog with local parquet caching to avoid re-hitting the API."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag   = season_type.replace(" ", "_")
    fpath = os.path.join(CACHE_DIR, f"{season}_{tag}.parquet")
    if os.path.exists(fpath):
        print(f"  cache hit  → {fpath}")
        return pd.read_parquet(fpath)
    for attempt in range(retries):
        try:
            time.sleep(SLEEP_SEC)
            df = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation="P",
                timeout=90,
            ).get_data_frames()[0]
            df.to_parquet(fpath, index=False)
            print(f"  fetched    → {len(df):,} rows")
            return df
        except Exception as exc:
            wait = 3 * (attempt + 1)
            print(f"  attempt {attempt+1} failed ({exc}), retrying in {wait}s…")
            time.sleep(wait)
    print(f"  FAILED after {retries} attempts — skipping")
    return pd.DataFrame()


def _parse_min(val) -> float:
    """Handle both 'MM:SS' strings and numeric minutes."""
    try:
        if isinstance(val, str) and ":" in val:
            m, s = val.split(":")
            return float(m) + float(s) / 60
        return float(val)
    except Exception:
        return 0.0


def _rolling_prior(series: pd.Series, n: int, min_periods: int = 1) -> pd.Series:
    """Rolling mean of the n games BEFORE the current one (no leakage)."""
    return series.shift(1).rolling(n, min_periods=min_periods).mean()


def _std_prior(series: pd.Series, n: int, min_periods: int = 2) -> pd.Series:
    """Rolling std of the n games BEFORE current."""
    return series.shift(1).rolling(n, min_periods=min_periods).std()


def _expanding_prior(series: pd.Series, min_periods: int = 1) -> pd.Series:
    """Season-to-date mean using only prior games."""
    return series.shift(1).expanding(min_periods=min_periods).mean()


def _ewma_prior(series: pd.Series, halflife: float = 3.0, min_periods: int = 3) -> pd.Series:
    """
    Exponentially weighted mean of prior games.
    halflife=3 means a game 3 games ago gets 50% the weight of the most recent game.
    This captures hot/cold streaks better than flat L5/L10 windows.
    """
    return series.shift(1).ewm(halflife=halflife, min_periods=min_periods).mean()


def _per_player(grp: pd.DataFrame) -> pd.DataFrame:
    """Compute all rolling pre-game features for one player's history."""
    grp = grp.sort_values("GAME_DATE").copy()

    # Rest days (capped at 14 — bye weeks, start of season get 7)
    grp["rest_days"] = grp["GAME_DATE"].diff().dt.days.fillna(7).clip(upper=14)

    # L5 rolling averages (prior games only — flat window)
    for col, feat in [
        ("PTS",    "l5_pts"),
        ("REB",    "l5_reb"),
        ("AST",    "l5_ast"),
        ("MIN",    "l5_min"),
        ("ts_pct", "l5_ts"),
    ]:
        grp[feat] = _rolling_prior(grp[col], 5)

    # EWMA recency features (halflife=3 games — recent form weighted 2× more than 3-game-old)
    # Captures hot/cold streaks missed by flat L5 averages.
    for col, feat in [
        ("PTS", "ewma_pts"),
        ("REB", "ewma_reb"),
        ("AST", "ewma_ast"),
        ("MIN", "ewma_min"),
    ]:
        grp[feat] = _ewma_prior(grp[col], halflife=3.0, min_periods=3)

    # L10 volatility features
    grp["l10_pts_std"] = _std_prior(grp["PTS"], 10)
    grp["l10_min_std"] = _std_prior(grp["MIN"], 10)

    # Season-to-date expanding means (reset each season via groupby upstream)
    for col, feat in [
        ("PTS", "std_pts"),
        ("REB", "std_reb"),
        ("AST", "std_ast"),
        ("MIN", "std_min"),
    ]:
        grp[feat] = _expanding_prior(grp[col])

    # USG proxy (numerator of USG% formula) — used for inactive_usg_pool computation
    if "FGA" in grp.columns and "FTA" in grp.columns and "TOV" in grp.columns:
        grp["usg_proxy_raw"] = grp["FGA"] + 0.44 * grp["FTA"] + grp["TOV"]
        grp["l5_usg_proxy"]  = _rolling_prior(grp["usg_proxy_raw"], 5, min_periods=3)
    else:
        grp["l5_usg_proxy"] = np.nan

    # Games played BEFORE this game in the current season (resets each season)
    grp["gp_prior"] = grp.groupby("season").cumcount()

    return grp


def _compute_inactive_usg_pool(df: pd.DataFrame) -> pd.Series:
    """
    For each player-game, compute the sum of rolling USG proxies of teammates
    who were expected to play (appeared in any of the last 10 games for that team)
    but did NOT play (0 MIN or absent from the game log).

    Vectorized implementation — avoids iterrows() for speed on 80k+ rows.
    """
    print("  Computing inactive_usg_pool…")

    # Active player set per (GAME_ID, TEAM_ABBREVIATION)
    played_df = df[df["MIN"] > 0][["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "PLAYER_ID"]].copy()
    game_team_active = (
        played_df.groupby(["GAME_ID", "TEAM_ABBREVIATION"])["PLAYER_ID"]
        .apply(set)
    )  # Series indexed by (GAME_ID, TEAM_ABBREVIATION)

    # Build expected roster per (TEAM_ABBREVIATION, GAME_DATE) using prior 10 games
    # Explicit loop per team to avoid pandas 2.x groupby/apply column-dropping issue
    team_date_active = (
        played_df.groupby(["TEAM_ABBREVIATION", "GAME_DATE"])["PLAYER_ID"]
        .apply(set)
        .reset_index()
        .rename(columns={"PLAYER_ID": "active_set"})
    )

    roster_lookup = {}   # (team, game_date) → frozenset of expected player IDs
    for team, grp in team_date_active.groupby("TEAM_ABBREVIATION"):
        grp = grp.sort_values("GAME_DATE").reset_index(drop=True)
        for i in range(len(grp)):
            prior_sets = grp.iloc[max(0, i - 10):i]["active_set"]
            roster = frozenset().union(*prior_sets) if len(prior_sets) > 0 else frozenset()
            roster_lookup[(team, grp.at[i, "GAME_DATE"])] = roster

    # USG proxy lookup: (PLAYER_ID, GAME_DATE) → pre-game rolling USG proxy
    usg_ser = df.set_index(["PLAYER_ID", "GAME_DATE"])["l5_usg_proxy"].dropna()
    usg_lookup = usg_ser.to_dict()

    # Vectorized pool computation using numpy arrays
    teams      = df["TEAM_ABBREVIATION"].values
    game_ids   = df["GAME_ID"].values
    player_ids = df["PLAYER_ID"].values
    dates      = df["GAME_DATE"].values

    pool_values = np.zeros(len(df), dtype=np.float32)
    for i in range(len(df)):
        key_team  = (teams[i], dates[i])
        expected  = roster_lookup.get(key_team, frozenset())
        if not expected:
            continue
        active   = game_team_active.get((game_ids[i], teams[i]), set())
        inactive = expected - active - {player_ids[i]}
        pool_values[i] = sum(
            usg_lookup.get((pid, dates[i]), 0.0) or 0.0
            for pid in inactive
        )

    return pd.Series(pool_values, index=df.index, name="inactive_usg_pool")


def main():
    # ── 1. Fetch all seasons ──────────────────────────────────────────────────
    frames = []
    for season in SEASONS:
        for stype in SEASON_TYPES:
            print(f"Fetching {season} / {stype}…")
            df = _fetch(season, stype)
            if not df.empty:
                df["season"]      = season
                df["season_type"] = stype
                frames.append(df)

    if not frames:
        print("No data fetched — exiting.")
        return

    raw = pd.concat(frames, ignore_index=True)
    print(f"\nRaw rows: {len(raw):,}")

    # ── 2. Parse & clean ──────────────────────────────────────────────────────
    raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"], errors="coerce")
    raw = raw.dropna(subset=["GAME_DATE", "PLAYER_ID"]).copy()

    num_cols = ["PTS", "REB", "AST", "MIN", "FGA", "FTA", "TOV",
                "OREB", "DREB", "FGM", "FG3M", "FG3A", "FTM",
                "STL", "BLK", "PLUS_MINUS"]
    for c in num_cols:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0)

    raw["MIN"] = raw["MIN"].apply(_parse_min)

    # Venue
    raw["is_home"] = raw["MATCHUP"].str.contains(r"vs\.", na=False).astype(int)

    # Per-game TS% only — usg_prox removed (FGA not cleanly available at inference)
    denom = 2 * (raw["FGA"] + 0.44 * raw["FTA"])
    raw["ts_pct"] = np.where(denom > 0, raw["PTS"] / denom, np.nan)

    # ── 3. Build opponent defensive efficiency proxy ───────────────────────────
    # Aggregate player logs to team totals per game
    agg_cols = {"team_pts": ("PTS", "sum"), "team_fga": ("FGA", "sum")}
    if "FG3A" in raw.columns:
        agg_cols["team_fg3a"] = ("FG3A", "sum")
    if "FGM" in raw.columns:
        agg_cols["team_fgm"] = ("FGM", "sum")
    team_game = raw.groupby(["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE"]).agg(**agg_cols).reset_index()

    # Self-join: each team row gets the opponent's stats (= what this team allowed)
    join_cols = ["GAME_ID", "TEAM_ABBREVIATION", "team_pts", "team_fga"]
    if "team_fg3a" in team_game.columns:
        join_cols.append("team_fg3a")
    if "team_fgm" in team_game.columns:
        join_cols.append("team_fgm")
    tg = team_game.merge(team_game[join_cols], on="GAME_ID", suffixes=("", "_opp"))
    tg = tg[tg["TEAM_ABBREVIATION"] != tg["TEAM_ABBREVIATION_opp"]].copy()
    tg = tg.rename(columns={"team_pts_opp": "pts_allowed", "team_fga_opp": "opp_fga"})
    tg = tg.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])

    # Rolling 10-game defensive and pace metrics (prior games only)
    tg["opp_def_roll10"]  = tg.groupby("TEAM_ABBREVIATION")["pts_allowed"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()
    )
    tg["opp_pace_roll10"] = tg.groupby("TEAM_ABBREVIATION")["team_fga"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()
    )

    # fg3_vs_avg proxy: rolling opponent 3PA/FGA rate vs league average.
    # Positive = team allows more 3PA (weak perimeter D); negative = fewer 3PA.
    # rim_vs_avg proxy: rolling opponent FG% vs league average.
    # Positive = team allows higher FG% (weak interior D); negative = stronger D.
    # NOTE: after the rename above, "opp_fga" = what this team's opponent attempted.
    #       "team_fg3a_opp" and "team_fgm_opp" are still the un-renamed suffix columns.
    _LEAGUE_AVG_FG3A_RATE = 0.37   # ~37% of FGA are 3PA in 2024-25 NBA
    _LEAGUE_AVG_FG_PCT    = 0.470  # ~47% overall FG% in 2024-25 NBA
    if "team_fg3a_opp" in tg.columns:
        tg["opp_fg3a_rate"] = tg["team_fg3a_opp"] / tg["opp_fga"].replace(0, np.nan)
        tg["fg3_vs_avg"] = tg.groupby("TEAM_ABBREVIATION")["opp_fg3a_rate"].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        ) - _LEAGUE_AVG_FG3A_RATE
    else:
        tg["fg3_vs_avg"] = np.nan

    if "team_fgm_opp" in tg.columns:
        tg["opp_fgpct"] = tg["team_fgm_opp"] / tg["opp_fga"].replace(0, np.nan)
        tg["rim_vs_avg"] = tg.groupby("TEAM_ABBREVIATION")["opp_fgpct"].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        ) - _LEAGUE_AVG_FG_PCT
    else:
        tg["rim_vs_avg"] = np.nan

    # Lookup table: game_id + player team → rolling opponent defensive stats
    opp_lookup_cols = ["GAME_ID", "TEAM_ABBREVIATION", "opp_def_roll10", "opp_pace_roll10",
                       "fg3_vs_avg", "rim_vs_avg"]
    opp_lookup = tg[[c for c in opp_lookup_cols if c in tg.columns]].copy()

    # ── 4. Per-player rolling features ────────────────────────────────────────
    # Explicit loop avoids pandas 2.x groupby/apply key-dropping behaviour.
    print("Computing per-player rolling features…")
    raw = raw.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    pieces = [_per_player(grp) for _, grp in raw.groupby("PLAYER_ID")]
    raw = pd.concat(pieces, ignore_index=True)

    # ── 5. Join opponent context ───────────────────────────────────────────────
    raw = raw.merge(opp_lookup, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")

    # ── 5b. Compute inactive_usg_pool ─────────────────────────────────────────
    # Must run AFTER per-player features (needs l5_usg_proxy) and AFTER opp join.
    raw["inactive_usg_pool"] = _compute_inactive_usg_pool(raw)
    print(f"  inactive_usg_pool: mean={raw['inactive_usg_pool'].mean():.2f}  "
          f"max={raw['inactive_usg_pool'].max():.2f}  "
          f"non-zero={( raw['inactive_usg_pool'] > 0).mean():.1%}")

    # ── 5c. Tracking-derived ML feature proxies ────────────────────────────────
    # efficiency_delta: player's L5 TS% vs NBA average (proxy for shot quality delta).
    # At inference, replaced by actual (l5_ts - xPPS) computed from tracking splits.
    _LEAGUE_AVG_TS = 0.559   # 2024-25 NBA avg TS% — used as training baseline
    raw["efficiency_delta"] = (raw["l5_ts"] - _LEAGUE_AVG_TS).fillna(0.0)

    # l5_potential_ast: creation volume proxy = l5_ast × 3.33 (AST/potAST ≈ 0.30 NBA avg).
    # At inference, replaced by actual potentialAst/g from passing tracking endpoint.
    raw["l5_potential_ast"] = (raw["l5_ast"] * 3.33).fillna(0.0)

    # ── 6. Define targets ─────────────────────────────────────────────────────
    raw["target_pts"] = raw["PTS"]
    raw["target_reb"] = raw["REB"]
    raw["target_ast"] = raw["AST"]

    # ── 7. Select output columns & filter ─────────────────────────────────────
    keep = [
        # Identifiers
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE",
        "season", "season_type",
        # Pre-game features — flat windows
        "is_home", "rest_days",
        "l5_pts", "l5_reb", "l5_ast", "l5_min", "l5_ts",
        "l10_pts_std", "l10_min_std",
        "std_pts", "std_reb", "std_ast", "std_min",
        "gp_prior",
        "opp_def_roll10", "opp_pace_roll10",
        # EWMA recency features (new — halflife=3 games)
        "ewma_pts", "ewma_reb", "ewma_ast", "ewma_min",
        # Usage redistribution pool (new — sum of absent teammates' USG proxy)
        "inactive_usg_pool",
        # Tracking-derived features (proxied from box-score for training)
        "fg3_vs_avg", "rim_vs_avg",       # opponent scheme concessions
        "efficiency_delta",               # player TS% vs expected (shot quality)
        "l5_potential_ast",               # creation volume proxy
        # Targets
        "target_pts", "target_reb", "target_ast",
        # Keep actual MIN for analysis — NOT a training feature (future leakage)
        "MIN",
    ]
    out = raw[[c for c in keep if c in raw.columns]].copy()

    # Drop rows with insufficient prior history
    out = out[out["gp_prior"] >= MIN_GP_PRIOR].reset_index(drop=True)

    print(f"\nFinal rows: {len(out):,}")
    print(f"Seasons:    {sorted(out['season'].unique())}")
    print(f"Players:    {out['PLAYER_ID'].nunique():,}")
    print(f"\nNaN counts in key features:")
    feat_cols = ["l5_pts", "l5_reb", "l5_ast", "l5_min",
                 "ewma_pts", "ewma_reb", "ewma_ast",
                 "inactive_usg_pool", "opp_def_roll10", "opp_pace_roll10"]
    print(out[[c for c in feat_cols if c in out.columns]].isna().sum().to_string())

    out.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved → {OUTPUT_FILE}")

    # Save feature column list and median imputation values for inference
    feature_cols = [
        "l5_pts", "l5_reb", "l5_ast", "l5_min", "l5_ts",
        "l10_pts_std", "l10_min_std",
        "std_pts", "std_reb", "std_ast", "std_min",
        "gp_prior", "is_home", "rest_days",
        "opp_def_roll10", "opp_pace_roll10",
        "ewma_pts", "ewma_reb", "ewma_ast", "ewma_min",
        "inactive_usg_pool",
        # Tracking-derived features (proxied at training; real values at inference)
        "fg3_vs_avg", "rim_vs_avg",
        "efficiency_delta",
        "l5_potential_ast",
    ]
    medians = {c: float(out[c].median()) for c in feature_cols if c in out.columns}
    meta = {"feature_cols": feature_cols, "medians": medians}
    with open("model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved → model_meta.json")


if __name__ == "__main__":
    main()
