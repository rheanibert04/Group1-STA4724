
import numpy as np
import pandas as pd

# final columns we keep for modeling
KEEP_COLS = [
    "Team","Season","Wins","PointDiff","FG%","3P%","FT%","3PAr","FTr","eFG%","TS%",
    "REB","OREB%","DREB%","AST/TOV","TOV%","ORtg","DRtg","NetRtg","PACE"
]

def safe_div(n, d):
    n = np.asarray(n, dtype="float64")
    d = np.asarray(d, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = n / d
    out[~np.isfinite(out)] = np.nan
    return out

# 2019-10-22 to most recent in file

def _parse_game_date(df: pd.DataFrame) -> pd.Series:
    """Try multiple date parsing patterns to handle inconsistent formats."""
    s = df["gameDate"].astype(str).str.strip()

    # try ISO or YYYY-MM-DD HH:MM:SS
    dt = pd.to_datetime(s, errors="coerce", utc=True)

    # fill NaTs by trying MM/DD/YYYY and similar US-style formats
    mask = dt.isna()
    if mask.any():
        dt2 = pd.to_datetime(s[mask], format="%m/%d/%Y %I:%M %p", errors="coerce", utc=True)
        dt.loc[mask] = dt2

    # try fallback parse
    mask = dt.isna()
    if mask.any():
        dt3 = pd.to_datetime(s[mask], errors="coerce", utc=True, infer_datetime_format=True)
        dt.loc[mask] = dt3

    return dt


def _label_nba_season(dt: pd.Series) -> pd.Series:
    # July 1 boundary: months < 7 to previous start year
    y = dt.dt.year
    m = dt.dt.month
    start_year = y.where(m >= 7, y - 1)
    return (start_year.astype(int).astype(str) + "-" +
            ((start_year + 1) % 100).astype(str).str.zfill(2))

def filter_past_games(df: pd.DataFrame, start_str: str = "2019-10-22") -> pd.DataFrame:
    d = df.copy()
    d.columns = d.columns.str.strip()
    if "gameDate" not in d.columns:
        raise KeyError("Expected 'gameDate' column in TeamStatistics CSV.")
    d["game_dt"] = _parse_game_date(d)
    start = pd.Timestamp(start_str, tz="UTC")
    end = d["game_dt"].max()  # most recent in THIS file
    d = d.loc[d["game_dt"].between(start, end)].copy()
    return d

def aggregate_team_season(past_filtered: pd.DataFrame) -> pd.DataFrame:
    d = past_filtered.copy()

    # season label 
    d["Season"] = _label_nba_season(d["game_dt"])

    # normalize team column
    team_col = "teamName" if "teamName" in d.columns else ("Team" if "Team" in d.columns else None)
    if not team_col:
        raise KeyError("Could not find a team column (expected 'teamName' or 'Team').")
    d = d.rename(columns={team_col: "Team"})

    # possessions (team side)
    d["_poss"] = (
        pd.to_numeric(d.get("fieldGoalsAttempted"), errors="coerce")
        - pd.to_numeric(d.get("reboundsOffensive"), errors="coerce")
        + pd.to_numeric(d.get("turnovers"), errors="coerce")
        + 0.44 * pd.to_numeric(d.get("freeThrowsAttempted"), errors="coerce")
    )

    # numeric aliases
    d["_pts"] = pd.to_numeric(d.get("teamScore"), errors="coerce")
    d["_opp"] = pd.to_numeric(d.get("opponentScore"), errors="coerce")
    d["_fga"] = pd.to_numeric(d.get("fieldGoalsAttempted"), errors="coerce")
    d["_fgm"] = pd.to_numeric(d.get("fieldGoalsMade"), errors="coerce")
    d["_3pa"] = pd.to_numeric(d.get("threePointersAttempted"), errors="coerce")
    d["_3pm"] = pd.to_numeric(d.get("threePointersMade"), errors="coerce")
    d["_fta"] = pd.to_numeric(d.get("freeThrowsAttempted"), errors="coerce")
    d["_or"]  = pd.to_numeric(d.get("reboundsOffensive"), errors="coerce")
    d["_dr"]  = pd.to_numeric(d.get("reboundsDefensive"), errors="coerce")
    d["_reb"] = pd.to_numeric(d.get("reboundsTotal"), errors="coerce")
    d["_ast"] = pd.to_numeric(d.get("assists"), errors="coerce")
    d["_tov"] = pd.to_numeric(d.get("turnovers"), errors="coerce")

    # optional source columns
    has_ft_pct = "freeThrowsPercentage" in d.columns
    has_wins   = "seasonWins" in d.columns

    # aggregate to Team × Season
    agg_dict = {
        "_pts":"mean","_opp":"mean","_fga":"mean","_fgm":"mean","_3pa":"mean","_3pm":"mean","_fta":"mean",
        "_or":"mean","_dr":"mean","_reb":"mean","_ast":"mean","_tov":"mean","_poss":"mean"
    }
    if has_ft_pct:
        agg_dict["freeThrowsPercentage"] = "mean"
    if has_wins:
        agg_dict["seasonWins"] = "max"

    agg = d.groupby(["Team","Season"]).agg(agg_dict).reset_index()

    # standardize names
    if has_ft_pct:
        agg = agg.rename(columns={"freeThrowsPercentage":"FT_pct_src"})
    else:
        agg["FT_pct_src"] = np.nan
    if has_wins:
        agg = agg.rename(columns={"seasonWins":"Wins"})
    else:
        agg["Wins"] = np.nan

    # feature engineering
    agg["PointDiff"] = agg["_pts"] - agg["_opp"]
    agg["FG%"]  = safe_div(agg["_fgm"], agg["_fga"]) * 100
    agg["3P%"]  = safe_div(agg["_3pm"], agg["_3pa"]) * 100
    agg["FT%"]  = pd.to_numeric(agg["FT_pct_src"], errors="coerce")
    agg["3PAr"] = safe_div(agg["_3pa"], agg["_fga"])
    agg["FTr"]  = safe_div(agg["_fta"], agg["_fga"])
    agg["eFG%"] = safe_div(agg["_fgm"] + 0.5*agg["_3pm"], agg["_fga"])
    agg["TS%"]  = safe_div(agg["_pts"], 2*(agg["_fga"] + 0.44*agg["_fta"]))
    agg["REB"]   = agg["_reb"]
    agg["OREB%"] = safe_div(agg["_or"], agg["_reb"])
    agg["DREB%"] = safe_div(agg["_dr"], agg["_reb"])
    agg["AST/TOV"] = safe_div(agg["_ast"], agg["_tov"])
    agg["TOV%"]    = safe_div(agg["_tov"], agg["_poss"])
    agg["ORtg"]   = safe_div(agg["_pts"], agg["_poss"]) * 100
    agg["DRtg"]   = safe_div(agg["_opp"], agg["_poss"]) * 100
    agg["NetRtg"] = agg["ORtg"] - agg["DRtg"]
    agg["PACE"]   = agg["_poss"]

    out = agg[KEEP_COLS].sort_values(["Season","Team"]).reset_index(drop=True)
    return out

# 2025–26 CSV to same columns

def _norm_header(s: str) -> str:
    return s.replace("\n", " ").strip()

def build_current_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [_norm_header(c) for c in d.columns]
    d = d.rename(columns={c: c.lower() for c in d.columns})

    if "team" not in d.columns:
        # attempt a few alternatives
        for cand in ["teamname","team_name","TEAM","Team"]:
            if cand in df.columns:
                d["team"] = df[cand]
                break
        else:
            raise KeyError("Expected a TEAM column in the 2025–26 CSV ('team' or similar).")

    # common aliases
    if "fg" in d.columns and "fgm" not in d.columns: d = d.rename(columns={"fg":"fgm"})
    if "3p" in d.columns and "3pm" not in d.columns: d = d.rename(columns={"3p":"3pm"})

    # possessions if possible
    if all(c in d.columns for c in ["fga","or","to","fta"]):
        d["_poss"] = d["fga"] - d["or"] + d["to"] + 0.44 * d["fta"]
    else:
        d["_poss"] = np.nan

    mean_cols = [c for c in [
        "pts","oeff","deff","pace","or","dr","tot","a","st","to","bl",
        "fga","fgm","3pa","3pm","fta","ft%","_poss"
    ] if c in d.columns]

    curr = d.groupby("team")[mean_cols].mean(numeric_only=True).reset_index().rename(columns={"team":"Team"})
    curr["Season"] = "2025-26"

    # variable features
    curr["FG%"]  = safe_div(curr.get("fgm", np.nan), curr.get("fga", np.nan)) * 100
    curr["3P%"]  = safe_div(curr.get("3pm", np.nan), curr.get("3pa", np.nan)) * 100
    curr["3PAr"] = safe_div(curr.get("3pa", np.nan), curr.get("fga", np.nan))
    curr["FTr"]  = safe_div(curr.get("fta", np.nan), curr.get("fga", np.nan))
    curr["eFG%"] = safe_div(curr.get("fgm", np.nan) + 0.5*curr.get("3pm", np.nan), curr.get("fga", np.nan))
    if all(c in curr.columns for c in ["pts","fga","fta"]):
        curr["TS%"] = safe_div(curr["pts"], 2*(curr["fga"] + 0.44*curr["fta"]))
    else:
        curr["TS%"] = np.nan

    curr["REB"]   = curr.get("tot", np.nan)
    curr["OREB%"] = safe_div(curr.get("or", np.nan), curr["REB"])
    curr["DREB%"] = safe_div(curr.get("dr", np.nan), curr["REB"])
    curr["AST/TOV"] = safe_div(curr.get("a", np.nan), curr.get("to", np.nan))
    curr["TOV%"]    = safe_div(curr.get("to", np.nan), curr.get("_poss", np.nan))

    curr["ORtg"]   = curr["oeff"] if "oeff" in curr.columns else safe_div(curr.get("pts", np.nan), curr.get("_poss", np.nan)) * 100
    curr["DRtg"]   = curr["deff"] if "deff" in curr.columns else np.nan
    curr["NetRtg"] = curr["ORtg"] - curr["DRtg"]
    curr["PACE"]   = curr["pace"] if "pace" in curr.columns else curr.get("_poss", np.nan)

    # current file won't have reliable season Wins / PointDiff
    curr["FT%"]       = curr["ft%"] if "ft%" in curr.columns else np.nan
    curr["PointDiff"] = np.nan
    curr["Wins"]      = np.nan

    out = curr[KEEP_COLS].sort_values("Team").reset_index(drop=True)
    return out

# merge datasets

def merge_all(past_csv: str, current_csv: str) -> pd.DataFrame:
    """
    1) Filter TeamStatistics to 2019-10-22 to most-recent date present
    2) Aggregate to Team × Season features.
    3) Build 2025–26 features from the current-season CSV
    4) Concatenate: keep past seasons up to last completed season, and use
       the current-season rows only from the current file (avoid duplicates)
    """
    past_raw = pd.read_csv(past_csv, low_memory=False)
    curr_raw = pd.read_csv(current_csv, low_memory=False)

    # build pieces
    past_filtered = filter_past_games(past_raw)
    past_features = aggregate_team_season(past_filtered)
    curr_features = build_current_features(curr_raw)  # Season == "2025-26"

    # drop the current season from the past piece to avoid double-counting
    last_dt = past_filtered["game_dt"].max()
    last_season_year = last_dt.year if last_dt.month >= 7 else last_dt.year - 1
    last_season_label = f"{last_season_year}-{str((last_season_year + 1) % 100).zfill(2)}"

    past_no_current = past_features[past_features["Season"] != last_season_label]

    # merge the two parts
    df = pd.concat([past_no_current, curr_features], ignore_index=True)

    # final tidy
    df["Team"] = df["Team"].astype(str).str.strip()
    df["Season"] = df["Season"].astype(str)

    # enforce a single row per Team×Season
    num_cols = df.select_dtypes(include="number").columns.tolist()
    agg_dict = {c: "mean" for c in num_cols}
    
    # keep the first occurrence for non-numerics
    agg_dict.update({"Team": "first", "Season": "first"})

    df = (
        df.groupby(["Team", "Season"], as_index=False)
          .agg(agg_dict)
          .reindex(columns=KEEP_COLS)
          .sort_values(["Season", "Team"])
          .reset_index(drop=True)
    )

    return df
