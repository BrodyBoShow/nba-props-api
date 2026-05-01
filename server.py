from flask import Flask, jsonify
from flask_cors import CORS
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashteamstats,
    leaguedashptteamdefend,
    playergamelog,
)
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
import pandas as pd
import time
import logging
import threading

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app)

SEASON = "2025-26"
_CACHE_TTL = 3600  # 1 hour

# ── In-memory cache ───────────────────────────────────────────────────────────
_cache: dict = {}
_cache_lock = threading.Lock()


def _cache_get(key):
    with _cache_lock:
        entry = _cache.get(key)
        if entry and (time.time() - entry["ts"]) < _CACHE_TTL:
            logging.info("Cache HIT: %s (age %.0fs)", key, time.time() - entry["ts"])
            return entry["data"]
    return None


def _cache_set(key, data):
    with _cache_lock:
        _cache[key] = {"data": data, "ts": time.time()}
    logging.info("Cache SET: %s", key)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _sleep():
    time.sleep(0.8)


def _pct(val):
    try:
        v = float(val)
        return round(v * 100, 1) if v else 0.0
    except (TypeError, ValueError):
        return 0.0


def _f(val, d=1):
    try:
        return round(float(val), d)
    except (TypeError, ValueError):
        return 0.0


def _i(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _stat_row(r):
    """Convert a row dict (from nba_api DataFrame) to a standardised stat object."""
    if not r:
        return None
    return {
        "ppg":  _f(r.get("PTS", 0)),
        "rpg":  _f(r.get("REB", 0)),
        "apg":  _f(r.get("AST", 0)),
        "spg":  _f(r.get("STL", 0)),
        "bpg":  _f(r.get("BLK", 0)),
        "topg": _f(r.get("TOV", 0)),
        "fg":   _pct(r.get("FG_PCT")),
        "fg3":  _pct(r.get("FG3_PCT")),
        "ft":   _pct(r.get("FT_PCT")),
        "min":  _f(r.get("MIN", 0)),
        "gp":   _i(r.get("GP", 0)),
    }


def _fetch_player_stats(season_type, measure="Base", location=None):
    kwargs = dict(
        season=SEASON,
        season_type_all_star=season_type,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense=measure,
    )
    if location:
        kwargs["location_nullable"] = location
    return leaguedashplayerstats.LeagueDashPlayerStats(**kwargs).get_data_frames()[0]


# ── Core data builders ────────────────────────────────────────────────────────
def _build_players():
    logging.info("Fetching PO base stats...")
    po_base = _fetch_player_stats("Playoffs")
    _sleep()
    logging.info("Fetching PO advanced stats...")
    po_adv = _fetch_player_stats("Playoffs", "Advanced")
    _sleep()
    logging.info("Fetching RS base stats...")
    rs_base = _fetch_player_stats("Regular Season")
    _sleep()
    logging.info("Fetching RS advanced stats...")
    rs_adv = _fetch_player_stats("Regular Season", "Advanced")

    po_adv_idx = po_adv.set_index("PLAYER_ID").to_dict("index")
    rs_adv_idx = rs_adv.set_index("PLAYER_ID").to_dict("index")
    rs_base_idx = rs_base.set_index("PLAYER_ID").to_dict("index")

    players = {}
    for _, row in po_base.iterrows():
        pid = int(row["PLAYER_ID"])
        name = row["PLAYER_NAME"].lower()
        team = row["TEAM_ABBREVIATION"]
        pa = po_adv_idx.get(pid, {})
        ra = rs_adv_idx.get(pid, {})
        rb = rs_base_idx.get(pid, {})

        po = {
            "ppg": _f(row["PTS"]), "rpg": _f(row["REB"]), "apg": _f(row["AST"]),
            "spg": _f(row["STL"]), "bpg": _f(row["BLK"]), "topg": _f(row["TOV"]),
            "fg": _pct(row.get("FG_PCT")), "fg3": _pct(row.get("FG3_PCT")),
            "ft": _pct(row.get("FT_PCT")), "min": _f(row["MIN"]), "gp": _i(row["GP"]),
            "usg": _pct(pa.get("USG_PCT")) if pa else None,
            "ts":  _pct(pa.get("TS_PCT"))  if pa else None,
        }
        if rb:
            rs = {
                "ppg": _f(rb.get("PTS", 0)), "rpg": _f(rb.get("REB", 0)),
                "apg": _f(rb.get("AST", 0)), "spg": _f(rb.get("STL", 0)),
                "bpg": _f(rb.get("BLK", 0)), "topg": _f(rb.get("TOV", 0)),
                "fg": _pct(rb.get("FG_PCT")), "fg3": _pct(rb.get("FG3_PCT")),
                "ft": _pct(rb.get("FT_PCT")), "min": _f(rb.get("MIN", 0)),
                "gp": _i(rb.get("GP", 0)),
                "usg": _pct(ra.get("USG_PCT")) if ra else None,
                "ts":  _pct(ra.get("TS_PCT"))  if ra else None,
            }
        else:
            rs = po
        players[name] = {"team": team, "pid": pid, "rs": rs, "po": po}

    logging.info("Built %d playoff players", len(players))
    return players


def _build_teams():
    logging.info("Fetching RS team advanced stats...")
    rs_adv = leaguedashteamstats.LeagueDashTeamStats(
        season=SEASON, season_type_all_star="Regular Season",
        per_mode_detailed="PerGame", measure_type_detailed_defense="Advanced",
    ).get_data_frames()[0]
    _sleep()
    logging.info("Fetching PO team advanced stats...")
    po_adv = leaguedashteamstats.LeagueDashTeamStats(
        season=SEASON, season_type_all_star="Playoffs",
        per_mode_detailed="PerGame", measure_type_detailed_defense="Advanced",
    ).get_data_frames()[0]

    po_idx = po_adv.set_index("TEAM_ABBREVIATION").to_dict("index")
    teams_data = {}
    for _, row in rs_adv.iterrows():
        abbr = row["TEAM_ABBREVIATION"]
        po = po_idx.get(abbr, {})
        teams_data[abbr] = {
            "fullName": row["TEAM_NAME"],
            "rsPace": _f(row.get("PACE", 100.0)),
            "oEFF": _f(po.get("OFF_RATING")) if po else None,
            "dEFF": _f(po.get("DEF_RATING")) if po else None,
            "eDIFF": _f(po.get("NET_RATING")) if po else None,
        }
    logging.info("Built %d teams", len(teams_data))
    return teams_data


def _build_splits():
    """
    Per-player home/road PO splits via location_nullable parameter.
    Returns {player_name: {pid, home: statRow, road: statRow}}.
    """
    logging.info("Fetching PO home splits...")
    home_df = _fetch_player_stats("Playoffs", location="Home")
    _sleep()
    logging.info("Fetching PO road splits...")
    road_df = _fetch_player_stats("Playoffs", location="Road")

    home_idx = home_df.set_index("PLAYER_ID").to_dict("index")
    road_idx = road_df.set_index("PLAYER_ID").to_dict("index")

    # Collect all player names + pids from both DataFrames
    all_players: dict = {}
    for df in (home_df, road_df):
        for _, row in df.iterrows():
            pid = int(row["PLAYER_ID"])
            name = row["PLAYER_NAME"].lower()
            if name not in all_players:
                all_players[name] = pid

    splits = {}
    for name, pid in all_players.items():
        splits[name] = {
            "pid": pid,
            "home": _stat_row(home_idx.get(pid)),
            "road": _stat_row(road_idx.get(pid)),
        }

    logging.info("Built splits for %d players", len(splits))
    return splits


def _build_team_defense():
    """
    Per-team opponent shooting %, by zone, in Playoffs.
    PCT_PLUSMINUS = opp_FG% - league_avg_FG% in that zone (decimal, e.g. -0.030).
    Negative = team defends this zone well; positive = team gives up more than avg.
    """
    logging.info("Fetching PO 3pt team defense...")
    fg3_df = leaguedashptteamdefend.LeagueDashPtTeamDefend(
        season=SEASON,
        season_type_all_star="Playoffs",
        defense_category="3 Pointers",
        per_mode_simple="PerGame",
    ).get_data_frames()[0]
    _sleep()

    logging.info("Fetching PO 2pt (<6ft rim) team defense...")
    rim_df = leaguedashptteamdefend.LeagueDashPtTeamDefend(
        season=SEASON,
        season_type_all_star="Playoffs",
        defense_category="Less Than 6Ft",
        per_mode_simple="PerGame",
    ).get_data_frames()[0]

    fg3_idx = fg3_df.set_index("TEAM_ABBREVIATION").to_dict("index")
    rim_idx  = rim_df.set_index("TEAM_ABBREVIATION").to_dict("index")

    all_abbrs = set(fg3_df["TEAM_ABBREVIATION"]) | set(rim_df["TEAM_ABBREVIATION"])
    team_def = {}
    for abbr in all_abbrs:
        fg3 = fg3_idx.get(abbr, {})
        rim = rim_idx.get(abbr, {})
        team_def[abbr] = {
            # How much better/worse opponents shoot on 3s vs league avg (decimal)
            "fg3VsAvg": _f(fg3.get("PCT_PLUSMINUS", 0), 4),
            "fg3OppPct": _f(fg3.get("D_FG_PCT", 0), 4),
            # Same for rim (< 6ft)
            "rimVsAvg": _f(rim.get("PCT_PLUSMINUS", 0), 4),
            "rimOppPct": _f(rim.get("D_FG_PCT", 0), 4),
        }
    logging.info("Built team defense for %d teams", len(team_def))
    return team_def


# ── Background warm-up on startup ─────────────────────────────────────────────
def _warmup():
    logging.info("Background warm-up starting...")
    try:
        players = _build_players()
        _cache_set("players", {"success": True, "players": players, "count": len(players)})
        _sleep()
        teams = _build_teams()
        _cache_set("teams", {"success": True, "teams": teams})
        _sleep()
        splits = _build_splits()
        _cache_set("splits", {"success": True, "splits": splits})
        _sleep()
        team_def = _build_team_defense()
        _cache_set("team_defense", {"success": True, "teamDefense": team_def})
        logging.info("Warm-up complete.")
    except Exception as e:
        logging.error("Warm-up failed: %s", e)


threading.Thread(target=_warmup, daemon=True).start()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/api/players")
def get_players():
    cached = _cache_get("players")
    if cached:
        return jsonify(cached)
    try:
        players = _build_players()
        result = {"success": True, "players": players, "count": len(players)}
        _cache_set("players", result)
        return jsonify(result)
    except Exception as e:
        logging.error("Error fetching players: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/teams")
def get_teams():
    cached = _cache_get("teams")
    if cached:
        return jsonify(cached)
    try:
        teams = _build_teams()
        result = {"success": True, "teams": teams}
        _cache_set("teams", result)
        return jsonify(result)
    except Exception as e:
        logging.error("Error fetching teams: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/splits")
def get_splits():
    """Home/road per-player playoff splits from NBA.com (location_nullable)."""
    cached = _cache_get("splits")
    if cached:
        return jsonify(cached)
    try:
        splits = _build_splits()
        result = {"success": True, "splits": splits}
        _cache_set("splits", result)
        return jsonify(result)
    except Exception as e:
        logging.error("Error fetching splits: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/team-defense")
def get_team_defense():
    """
    Per-team zone defense (3pt + rim) from NBA.com LeagueDashPtTeamDefend.
    PCT_PLUSMINUS = opp FG% minus league avg (decimal) — negative = good defense.
    """
    cached = _cache_get("team_defense")
    if cached:
        return jsonify(cached)
    try:
        team_def = _build_team_defense()
        result = {"success": True, "teamDefense": team_def}
        _cache_set("team_defense", result)
        return jsonify(result)
    except Exception as e:
        logging.error("Error fetching team defense: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/schedule")
def get_schedule():
    try:
        board = live_scoreboard.ScoreBoard()
        games_raw = board.games.get_dict()
        games = []
        for g in games_raw:
            away = g.get("awayTeam", {})
            home = g.get("homeTeam", {})
            games.append({
                "id": g.get("gameId"),
                "away": away.get("teamTricode"),
                "home": home.get("teamTricode"),
                "awayTeam": away.get("teamName"),
                "homeTeam": home.get("teamName"),
                "status": g.get("gameStatusText"),
                "awayScore": away.get("score"),
                "homeScore": home.get("score"),
                "period": g.get("period"),
            })
        return jsonify({"success": True, "games": games})
    except Exception as e:
        logging.error("Error fetching schedule: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/cache-status")
def cache_status():
    with _cache_lock:
        status = {}
        for k, v in _cache.items():
            age = time.time() - v["ts"]
            status[k] = {
                "age_seconds": round(age),
                "expires_in": round(_CACHE_TTL - age),
                "valid": age < _CACHE_TTL,
            }
    return jsonify(status)


# ── Game log endpoints (not cached — player-specific, called on demand) ───────
def _parse_min(val):
    try:
        if isinstance(val, str) and ":" in val:
            parts = val.split(":")
            return round(int(parts[0]) + int(parts[1]) / 60, 1)
        return _f(val)
    except (ValueError, IndexError):
        return 0.0


def _game_log_avg(df):
    def safe_pct(col):
        val = df[col].mean()
        return round(float(val) * 100, 1) if not pd.isna(val) else 0.0

    return {
        "ppg":  _f(df["PTS"].mean()),
        "rpg":  _f(df["REB"].mean()),
        "apg":  _f(df["AST"].mean()),
        "spg":  _f(df["STL"].mean()),
        "bpg":  _f(df["BLK"].mean()),
        "topg": _f(df["TOV"].mean()),
        "fg":   safe_pct("FG_PCT"),
        "fg3":  safe_pct("FG3_PCT"),
        "ft":   safe_pct("FT_PCT"),
        "min":  _f(df["MIN"].apply(_parse_min).mean()),
        "gp":   len(df),
    }


@app.route("/api/recent/<int:player_id>")
def get_recent(player_id):
    try:
        logs = playergamelog.PlayerGameLog(
            player_id=player_id, season=SEASON,
            season_type_all_star="Playoffs",
        ).get_data_frames()[0]
        if logs.empty:
            return jsonify({"success": True, "recent": None, "gp": 0})
        recent = logs.head(5)
        return jsonify({"success": True, "recent": _game_log_avg(recent), "gp": len(recent)})
    except Exception as e:
        logging.error("Error fetching recent stats: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/vs-opponent/<int:player_id>/<opp_abbr>")
def get_vs_opponent(player_id, opp_abbr):
    try:
        po_logs = playergamelog.PlayerGameLog(
            player_id=player_id, season=SEASON,
            season_type_all_star="Playoffs",
        ).get_data_frames()[0]
        _sleep()

        vs = po_logs[po_logs["MATCHUP"].str.contains(opp_abbr, na=False, case=False)]
        source = "Playoffs"

        if len(vs) < 2:
            rs_logs = playergamelog.PlayerGameLog(
                player_id=player_id, season=SEASON,
                season_type_all_star="Regular Season",
            ).get_data_frames()[0]
            rs_vs = rs_logs[rs_logs["MATCHUP"].str.contains(opp_abbr, na=False, case=False)]
            vs = pd.concat([vs, rs_vs])
            source = "PO+RS" if not po_logs.empty else "RS"

        if vs.empty:
            return jsonify({"success": True, "vsOpponent": None, "gp": 0, "source": None})

        return jsonify({"success": True, "vsOpponent": _game_log_avg(vs), "gp": len(vs), "source": source})
    except Exception as e:
        logging.error("Error fetching vs-opponent stats: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
