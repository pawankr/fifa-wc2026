import pandas as pd
import numpy as np
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.load_data import download_historical_data, load_historical_data, load_fixtures
from src.features.elo import EloSystem
from src.features.build_features import compute_rolling_features
from src.models.poisson import train_poisson_model
from src.models.xgboost_model import XGBoostPredictor
from src.simulation.group_stage import (
    run_group_stage, get_group_qualifiers, predict_group_match, THIRD_PLACE_SLOTS
)
from src.simulation.knockout import run_full_knockout

np.random.seed(42)

# Actual playoff qualifiers confirmed March 31, 2026
PLAYOFF_MAP = {
    "UEFA Playoff A": "Sweden",
    "UEFA Playoff B": "Turkey",
    "UEFA Playoff C": "Bosnia and Herzegovina",
    "UEFA Playoff D": "Czech Republic",
    "FIFA Playoff 1": "Iraq",
    "FIFA Playoff 2": "DR Congo",
}
# Historical data uses "Czech Republic" not "Czechia"
NAME_MAP = {"Czechia": "Czech Republic", "USA": "United States"}


def resolve_playoff_name(name):
    if name in PLAYOFF_MAP:
        return PLAYOFF_MAP[name]
    return NAME_MAP.get(name, name)


def build_team_recent_stats(matches_df):
    stats = {}
    for _, row in matches_df.iterrows():
        for team, gf, ga in [
            (row["home_team"], row["home_score"], row["away_score"]),
            (row["away_team"], row["away_score"], row["home_score"]),
        ]:
            stats.setdefault(team, {"goals_for": [], "goals_against": [], "points": []})
            stats[team]["goals_for"].append(gf)
            stats[team]["goals_against"].append(ga)
            stats[team]["points"].append(3 if gf > ga else 1 if gf == ga else 0)

    return {
        team: {
            "goals_scored_avg": float(np.mean(d["goals_for"][-10:])) if d["goals_for"] else 1.5,
            "goals_conceded_avg": float(np.mean(d["goals_against"][-10:])) if d["goals_against"] else 1.5,
            "form_points": float(np.mean(d["points"][-5:])) if d["points"] else 1.5,
            "matches_played": min(len(d["goals_for"]), 10),
        }
        for team, d in stats.items()
    }


def resolve_slot(slot_str, winners, runners_up):
    for g in ["A","B","C","D","E","F","G","H","I","J","K","L"]:
        if re.search(rf"Winner\s+Group\s+{g}", slot_str, re.IGNORECASE):
            return winners.get(g, "TBD")
        if re.search(rf"Runner-up\s+Group\s+{g}", slot_str, re.IGNORECASE):
            return runners_up.get(g, "TBD")
    return "TBD"


def main():
    print("=" * 60)
    print("FIFA World Cup 2026 — Match Predictor")
    print("=" * 60)

    print("\n[1/6] Loading data...")
    download_historical_data()
    matches_df = load_historical_data(min_year=2000)
    group_fixtures, knockout_slots = load_fixtures()
    print(f"  Historical: {len(matches_df)} matches")
    print(f"  Group fixtures: {len(group_fixtures)}")
    print(f"  Knockout slots: {len(knockout_slots)}")

    print("\n[2/6] Computing Elo ratings...")
    elo = EloSystem(initial_rating=1500, k_factor_base=40)
    training_matches = matches_df[matches_df["date"] < "2026-01-01"].copy()
    ratings_df = elo.compute_historical_ratings(training_matches)
    print(f"  Elo: {len(ratings_df)} matches, {len(elo.ratings)} teams")
    top_15 = sorted(elo.ratings.items(), key=lambda x: x[1], reverse=True)[:15]
    print(f"  Top 15: {[(t, int(r)) for t, r in top_15]}")

    print("\n[3/6] Computing features...")
    features_df = compute_rolling_features(training_matches, ratings_df)
    team_stats = build_team_recent_stats(training_matches)
    print(f"  Features: {len(features_df)} rows")

    features_fn = lambda h, a: {
        "elo_difference": elo.get_rating(resolve_playoff_name(h)) - elo.get_rating(resolve_playoff_name(a)),
        "home_goals_scored_avg": team_stats.get(resolve_playoff_name(h), {}).get("goals_scored_avg", 1.5),
        "home_goals_conceded_avg": team_stats.get(resolve_playoff_name(h), {}).get("goals_conceded_avg", 1.3),
        "away_goals_scored_avg": team_stats.get(resolve_playoff_name(a), {}).get("goals_scored_avg", 1.3),
        "away_goals_conceded_avg": team_stats.get(resolve_playoff_name(a), {}).get("goals_conceded_avg", 1.5),
        "home_form_points": team_stats.get(resolve_playoff_name(h), {}).get("form_points", 1.5),
        "away_form_points": team_stats.get(resolve_playoff_name(a), {}).get("form_points", 1.4),
        "home_matches_played": team_stats.get(resolve_playoff_name(h), {}).get("matches_played", 20),
        "away_matches_played": team_stats.get(resolve_playoff_name(a), {}).get("matches_played", 20),
        "is_world_cup": 1, "neutral": 1,
    }

    print("\n[4/6] Training Poisson model...")
    poisson_model = train_poisson_model(training_matches)
    print(f"  Teams: {len(poisson_model.teams)}, avg home: {poisson_model.avg_home_goals:.3f}")

    print("\n[5/6] Training XGBoost model...")
    xgb_model = XGBoostPredictor()
    xgb_model.train_outcome(features_df)
    xgb_model.train_goals_regressors(features_df)

    print("\n[6/6] Simulations + Output...")
    print("\n  --- Group Stage ---")
    all_results, all_standings = run_group_stage(group_fixtures, poisson_model, xgb_model, features_fn)

    for g in sorted(all_standings.keys()):
        standings = all_standings[g]
        print(f"  Group {g}:")
        for i, (team, s) in enumerate(standings, 1):
            print(f"    {i}. {team:20s} {s['points']}pts ({s['gd']:+d} GD)")

    winners, runners_up, thirds_map = get_group_qualifiers(all_standings)

    # Resolve playoff placeholders to actual team names
    winners = {g: resolve_playoff_name(t) for g, t in winners.items()}
    runners_up = {g: resolve_playoff_name(t) for g, t in runners_up.items()}
    thirds_map = {g: resolve_playoff_name(t) for g, t in thirds_map.items()}

    print(f"\n  Winners:  {[winners[g] for g in sorted(winners.keys())]}")
    print(f"  Runners:  {[runners_up[g] for g in sorted(runners_up.keys())]}")
    print(f"  3rds:     {list(thirds_map.values())}")

    # Build Round of 32 with one 3rd place team per slot
    r32 = knockout_slots[knockout_slots["round"] == "Round of 32"].copy()
    slot_3rd_teams = list(thirds_map.values())
    slot_idx = 0
    for i in r32.index:
        r32.at[i, "home_team"] = resolve_slot(str(r32.at[i, "slot_home"]), winners, runners_up)
        r32.at[i, "away_team"] = resolve_slot(str(r32.at[i, "slot_away"]), winners, runners_up)
        for col in ["slot_home", "slot_away"]:
            if "Best 3rd" in str(r32.at[i, col]) and slot_idx < len(slot_3rd_teams):
                r32.at[i, "home_team" if col == "slot_home" else "away_team"] = slot_3rd_teams[slot_idx]
                slot_idx += 1

    print("\n  --- Knockout Stage ---")
    knockout_predictions = run_full_knockout(knockout_slots, poisson_model, xgb_model, features_fn, r32)

    if len(knockout_predictions) > 0:
        for _, m in knockout_predictions.iterrows():
            rnd = m.get("round", "")
            print(f"  {m['match_id']:3d} {rnd:20s} {m['home_team']:18s} {m['home_goals']}-{m['away_goals']}  {m['away_team']:18s}{' (P)' if m.get('penalties') else ''}")

    print("\n  --- Saving ---")
    # Group predictions
    group_preds = group_fixtures.copy()
    for col in ["predicted_home_goals", "predicted_away_goals", "corners", "yellow_cards", "red_cards"]:
        group_preds[col] = 0
    group_preds["winning_team"] = ""

    for idx in group_fixtures.index:
        h, a = group_fixtures.at[idx, "home_team"], group_fixtures.at[idx, "away_team"]
        p = predict_group_match(h, a, poisson_model, xgb_model, features_fn)
        group_preds.at[idx, "predicted_home_goals"] = p["home_goals"]
        group_preds.at[idx, "predicted_away_goals"] = p["away_goals"]
        group_preds.at[idx, "corners"] = p["corners"]
        group_preds.at[idx, "yellow_cards"] = p["yellow_cards"]
        group_preds.at[idx, "red_cards"] = p["red_cards"]
        group_preds.at[idx, "winning_team"] = p["winning_team"]

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    group_preds.to_csv(out / "group_predictions.csv", index=False)
    print(f"  {out / 'group_predictions.csv'} ({len(group_preds)} matches)")

    if len(knockout_predictions) > 0:
        cols = ["match_id", "predicted_home_team", "predicted_away_team",
                "predicted_home_goals", "predicted_away_goals",
                "corners", "yellow_cards", "red_cards", "match_winner", "penalties"]
        kp = knockout_predictions.rename(columns={
            "home_team": "predicted_home_team", "away_team": "predicted_away_team",
            "home_goals": "predicted_home_goals", "away_goals": "predicted_away_goals",
        })
        kp = kp[[c for c in cols if c in kp.columns]].drop_duplicates(subset="match_id").sort_values("match_id")
        kp.to_csv(out / "knockout_predictions.csv", index=False)
        print(f"  {out / 'knockout_predictions.csv'} ({len(kp)} matches)")

    print("\nDone! outputs/group_predictions.csv & outputs/knockout_predictions.csv")


if __name__ == "__main__":
    main()
