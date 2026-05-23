import pandas as pd
import numpy as np
import sys
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
from src.simulation.knockout import resolve_first_round, run_full_knockout

np.random.seed(42)

RECENT_MATCHES_WINDOW = 10
FORM_MATCHES_WINDOW = 5


def build_team_recent_stats(matches_df):
    stats = {}
    for _, row in matches_df.iterrows():
        for team, gf, ga in [
            (row["home_team"], row["home_score"], row["away_score"]),
            (row["away_team"], row["away_score"], row["home_score"]),
        ]:
            if team not in stats:
                stats[team] = {"goals_for": [], "goals_against": [], "points": []}
            stats[team]["goals_for"].append(gf)
            stats[team]["goals_against"].append(ga)
            if gf > ga:
                stats[team]["points"].append(3)
            elif gf == ga:
                stats[team]["points"].append(1)
            else:
                stats[team]["points"].append(0)

    team_features = {}
    for team, data in stats.items():
        recent_gf = data["goals_for"][-RECENT_MATCHES_WINDOW:] if len(data["goals_for"]) > RECENT_MATCHES_WINDOW else data["goals_for"]
        recent_ga = data["goals_against"][-RECENT_MATCHES_WINDOW:] if len(data["goals_against"]) > RECENT_MATCHES_WINDOW else data["goals_against"]
        recent_pts = data["points"][-FORM_MATCHES_WINDOW:] if len(data["points"]) > FORM_MATCHES_WINDOW else data["points"]
        team_features[team] = {
            "goals_scored_avg": float(np.mean(recent_gf)) if recent_gf else 1.5,
            "goals_conceded_avg": float(np.mean(recent_ga)) if recent_ga else 1.5,
            "form_points": float(np.mean(recent_pts)) if recent_pts else 1.5,
            "matches_played": len(recent_gf),
        }
    return team_features


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
    print(f"  Elo computed across {len(ratings_df)} matches")
    print(f"  Teams tracked: {len(elo.ratings)}")
    top_15 = sorted(elo.ratings.items(), key=lambda x: x[1], reverse=True)[:15]
    print(f"  Top 15 Elo: {[(t, int(r)) for t, r in top_15]}")

    print("\n[3/6] Computing features and team stats...")
    features_df = compute_rolling_features(training_matches, ratings_df)
    team_stats = build_team_recent_stats(training_matches)
    print(f"  Features: {len(features_df)} rows")
    print(f"  Teams with stats: {len(team_stats)}")

    features_fn = lambda h, a: {
        "elo_difference": elo.get_rating(h) - elo.get_rating(a),
        "home_goals_scored_avg": team_stats.get(h, {}).get("goals_scored_avg", 1.5),
        "home_goals_conceded_avg": team_stats.get(h, {}).get("goals_conceded_avg", 1.3),
        "away_goals_scored_avg": team_stats.get(a, {}).get("goals_scored_avg", 1.3),
        "away_goals_conceded_avg": team_stats.get(a, {}).get("goals_conceded_avg", 1.5),
        "home_form_points": team_stats.get(h, {}).get("form_points", 1.5),
        "away_form_points": team_stats.get(a, {}).get("form_points", 1.4),
        "home_matches_played": team_stats.get(h, {}).get("matches_played", 20),
        "away_matches_played": team_stats.get(a, {}).get("matches_played", 20),
        "is_world_cup": 1,
        "neutral": 1,
    }

    print("\n[4/6] Training Poisson model...")
    poisson_model = train_poisson_model(training_matches)
    print(f"  Teams: {len(poisson_model.teams)}")
    print(f"  Avg home goals: {poisson_model.avg_home_goals:.3f}, away: {poisson_model.avg_away_goals:.3f}")

    print("\n[5/6] Training XGBoost model...")
    xgb_model = XGBoostPredictor()
    xgb_model.train_outcome(features_df)
    xgb_model.train_goals_regressors(features_df)

    print("\n[6/6] Running simulations...")
    print("\n  --- Group Stage ---")
    all_results, all_standings = run_group_stage(group_fixtures, poisson_model, xgb_model, features_fn)

    for g in sorted(all_standings.keys()):
        standings = all_standings[g]
        print(f"  Group {g}:")
        for i, (team, stats) in enumerate(standings, 1):
            print(f"    {i}. {team:20s} {stats['points']}pts ({stats['gd']:+d} GD)")

    winners, runners_up, third_qualifiers = get_group_qualifiers(all_standings)
    print(f"\n  Group winners: {[winners[g] for g in sorted(winners.keys())]}")
    print(f"  Runners-up:    {[runners_up[g] for g in sorted(runners_up.keys())]}")
    print(f"  Best 3rd:      {list(third_qualifiers.values())}")

    # Pre-assign 3rd place teams to specific bracket slots (each team used once)
    third_place_slot_map = {}
    used_thirds = set()
    for slot_name, eligible_groups in THIRD_PLACE_SLOTS.items():
        for group in eligible_groups:
            if group in third_qualifiers and third_qualifiers[group] not in used_thirds:
                third_place_slot_map[tuple(eligible_groups)] = third_qualifiers[group]
                used_thirds.add(third_qualifiers[group])
                break

    # Build a per-match-id lookup for 3rd place assignments
    round_32 = knockout_slots[knockout_slots["round"] == "Round of 32"]
    match_id_third = {}
    for _, match in round_32.iterrows():
        for slot_col in ["slot_home", "slot_away"]:
            slot = str(match[slot_col])
            if "Best 3rd" in slot:
                import re
                m = re.search(r"Best 3rd\s*\(Groups\s+([A-Za-z/]+)\)", slot)
                if m:
                    group_set = tuple(m.group(1).split("/"))
                    if group_set in third_place_slot_map:
                        match_id_third[match["match_id"]] = third_place_slot_map[group_set]

    def resolve_slot_with_3rd(slot_str, winners, runners_up, match_3rd):
        import re
        groups = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
        for g in groups:
            if re.search(rf"Winner\s+Group\s+{g}", slot_str, re.IGNORECASE):
                return winners.get(g, "TBD")
            if re.search(rf"Runner-up\s+Group\s+{g}", slot_str, re.IGNORECASE):
                return runners_up.get(g, "TBD")
        return "TBD"

    def resolve_first_round_custom():
        r32 = knockout_slots[knockout_slots["round"] == "Round of 32"].copy()
        for idx, match in r32.iterrows():
            r32.at[idx, "home_team"] = resolve_slot_with_3rd(str(match["slot_home"]), winners, runners_up, match_id_third)
            r32.at[idx, "away_team"] = resolve_slot_with_3rd(str(match["slot_away"]), winners, runners_up, match_id_third)

        # Fill 3rd place teams into their assigned slots
        for match_id, team in match_id_third.items():
            mask = r32["match_id"] == match_id
            for idx in r32[mask].index:
                slot = str(r32.at[idx, "slot_home"] if "Best 3rd" in str(r32.at[idx, "slot_home"]) else r32.at[idx, "slot_away"])
                import re
                m = re.search(r"Best 3rd\s*\(Groups\s+([A-Za-z/]+)\)", slot)
                if m:
                    group_set = tuple(m.group(1).split("/"))
                    if tuple(group_set) in third_place_slot_map:
                        fill_team = third_place_slot_map[tuple(group_set)]
                        if "Best 3rd" in str(r32.at[idx, "slot_home"]):
                            r32.at[idx, "home_team"] = fill_team
                        else:
                            r32.at[idx, "away_team"] = fill_team
        return r32

    resolved_r32 = resolve_first_round_custom()
    knockout_predictions = run_full_knockout(knockout_slots, poisson_model, xgb_model, features_fn, resolved_r32)

    print("\n  --- Group Match Predictions ---")
    for idx in group_fixtures.index:
        home, away = group_fixtures.at[idx, "home_team"], group_fixtures.at[idx, "away_team"]
        pred = predict_group_match(home, away, poisson_model, xgb_model, features_fn)
        print(f"  {home:20s} {pred['home_goals']}-{pred['away_goals']}  {away:20s} ({pred['winning_team']})")

    if len(knockout_predictions) > 0:
        print("\n  --- Knockout Bracket ---")
        for _, m in knockout_predictions.iterrows():
            rnd = m.get("round", "")
            print(f"  {rnd:20s} {m['home_team']:18s} {m['home_goals']}-{m['away_goals']}  {m['away_team']:18s}{' (P)' if m['penalties'] else ''}")

    print("\n  --- Saving Outputs ---")
    group_predictions = group_fixtures.copy()
    for col in ["predicted_home_goals", "predicted_away_goals", "corners", "yellow_cards", "red_cards"]:
        group_predictions[col] = 0
    group_predictions["winning_team"] = ""

    for idx in group_fixtures.index:
        home, away = group_fixtures.at[idx, "home_team"], group_fixtures.at[idx, "away_team"]
        pred = predict_group_match(home, away, poisson_model, xgb_model, features_fn)
        group_predictions.at[idx, "predicted_home_goals"] = pred["home_goals"]
        group_predictions.at[idx, "predicted_away_goals"] = pred["away_goals"]
        group_predictions.at[idx, "corners"] = pred["corners"]
        group_predictions.at[idx, "yellow_cards"] = pred["yellow_cards"]
        group_predictions.at[idx, "red_cards"] = pred["red_cards"]
        group_predictions.at[idx, "winning_team"] = pred["winning_team"]

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    group_save = group_predictions[[
        "match_id", "group", "home_team", "away_team",
        "predicted_home_goals", "predicted_away_goals",
        "corners", "yellow_cards", "red_cards", "winning_team"
    ]].copy()
    group_save.to_csv(output_dir / "group_predictions.csv", index=False)
    print(f"  {output_dir / 'group_predictions.csv'} ({len(group_save)} matches)")

    if len(knockout_predictions) > 0:
        cols = ["match_id", "predicted_home_team", "predicted_away_team",
                "predicted_home_goals", "predicted_away_goals",
                "corners", "yellow_cards", "red_cards", "match_winner", "penalties"]
        kp = knockout_predictions.rename(columns={
            "home_team": "predicted_home_team",
            "away_team": "predicted_away_team",
            "home_goals": "predicted_home_goals",
            "away_goals": "predicted_away_goals",
        })
        kp = kp[[c for c in cols if c in kp.columns]].copy()
        kp = kp.drop_duplicates(subset=["match_id"]).sort_values("match_id")
        kp.to_csv(output_dir / "knockout_predictions.csv", index=False)
        print(f"  {output_dir / 'knockout_predictions.csv'} ({len(kp)} matches)")
    else:
        print("  No knockout predictions generated")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
