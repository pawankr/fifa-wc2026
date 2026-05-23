import pandas as pd
import numpy as np

RECENT_MATCHES_WINDOW = 10
FORM_MATCHES_WINDOW = 5


def compute_rolling_features(matches_df, ratings_df):
    features = []
    matches_with_ratings = matches_df.copy()
    matches_with_ratings["home_elo"] = ratings_df["home_elo_before"]
    matches_with_ratings["away_elo"] = ratings_df["away_elo_before"]
    matches_with_ratings["elo_diff"] = matches_with_ratings["home_elo"] - matches_with_ratings["away_elo"]

    team_match_log = {}

    for idx, row in matches_df.iterrows():
        home, away = row["home_team"], row["away_team"]
        date = row["date"]
        home_g, away_g = row["home_score"], row["away_score"]

        home_features = _get_team_features(team_match_log, home, date)
        away_features = _get_team_features(team_match_log, away, date)

        features.append({
            "date": date,
            "home_team": home,
            "away_team": away,
            "home_score": home_g,
            "away_score": away_g,
            "tournament": row["tournament"],
            "neutral": row["neutral"],
            "elo_home": ratings_df.loc[idx, "home_elo_before"],
            "elo_away": ratings_df.loc[idx, "away_elo_before"],
            "elo_difference": ratings_df.loc[idx, "home_elo_before"] - ratings_df.loc[idx, "away_elo_before"],
            "home_goals_scored_avg": home_features["goals_scored_avg"],
            "home_goals_conceded_avg": home_features["goals_conceded_avg"],
            "away_goals_scored_avg": away_features["goals_scored_avg"],
            "away_goals_conceded_avg": away_features["goals_conceded_avg"],
            "home_form_points": home_features["form_points"],
            "away_form_points": away_features["form_points"],
            "home_matches_played": home_features["matches_played"],
            "away_matches_played": away_features["matches_played"],
            "is_world_cup": int("World Cup" in str(row["tournament"]) and "qualification" not in str(row["tournament"]).lower()),
        })

        _update_team_log(team_match_log, home, date, home_g, away_g)
        _update_team_log(team_match_log, away, date, away_g, home_g)

    return pd.DataFrame(features)


def _get_team_features(team_log, team, date):
    matches = team_log.get(team, [])
    recent = [m for m in matches if m["date"] < date]
    recent = recent[-RECENT_MATCHES_WINDOW:] if len(recent) > RECENT_MATCHES_WINDOW else recent
    form = [m for m in recent if m["date"] >= date - pd.Timedelta(days=730)]
    form = form[-FORM_MATCHES_WINDOW:] if len(form) > FORM_MATCHES_WINDOW else form

    if not recent:
        return {
            "goals_scored_avg": 1.0,
            "goals_conceded_avg": 1.0,
            "form_points": 1.0,
            "matches_played": 0,
        }

    return {
        "goals_scored_avg": np.mean([m["goals_for"] for m in recent]) if recent else 1.0,
        "goals_conceded_avg": np.mean([m["goals_against"] for m in recent]) if recent else 1.0,
        "form_points": np.mean([m["points"] for m in form]) if form else 1.0,
        "matches_played": len(recent),
    }


def _update_team_log(team_log, team, date, goals_for, goals_against):
    if team not in team_log:
        team_log[team] = []
    if goals_for > goals_against:
        points = 3
    elif goals_for == goals_against:
        points = 1
    else:
        points = 0
    team_log[team].append({
        "date": date,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "points": points,
    })
