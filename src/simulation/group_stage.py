import pandas as pd
import numpy as np

GROUP_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]


def build_group_teams(fixtures_df):
    groups = {}
    for _, row in fixtures_df.iterrows():
        g = row["group"]
        if g not in groups:
            groups[g] = set()
        groups[g].add(row["home_team"])
        groups[g].add(row["away_team"])
    return groups


def predict_group_match(home, away, poisson_model, xgb_model, features_fn):
    p_probs = poisson_model.match_probabilities(home, away)
    xgb_probs = xgb_model.predict_outcome(features_fn(home, away))
    xgb_hg, xgb_ag = xgb_model.predict_goals(features_fn(home, away))

    w_poisson, w_xgb = 0.3, 0.7
    combined_hw = w_poisson * p_probs["home_win"] + w_xgb * xgb_probs["home_win"]
    combined_d = w_poisson * p_probs["draw"] + w_xgb * xgb_probs["draw"]
    combined_aw = w_poisson * p_probs["away_win"] + w_xgb * xgb_probs["away_win"]
    total = combined_hw + combined_d + combined_aw
    combined_hw, combined_d, combined_aw = combined_hw / total, combined_d / total, combined_aw / total

    exp_h, exp_a = p_probs["home_exp"], p_probs["away_exp"]
    avg_h = int(round((exp_h + xgb_hg) / 2))
    avg_a = int(round((exp_a + xgb_ag) / 2))

    r = np.random.random()
    if r < combined_hw:
        result = "home"
        if avg_h <= avg_a:
            avg_h = avg_a + 1
    elif r < combined_hw + combined_d:
        result = "draw"
        if avg_h != avg_a:
            m = max(avg_h, avg_a)
            avg_h = avg_a = m
    else:
        result = "away"
        if avg_a <= avg_h:
            avg_a = avg_h + 1

    corners_h = int(np.random.poisson(5 + 0.3 * max(0, exp_h - 1)))
    corners_a = int(np.random.poisson(4 + 0.3 * max(0, exp_a - 1)))

    return {
        "home_goals": max(0, avg_h),
        "away_goals": max(0, avg_a),
        "corners": corners_h + corners_a,
        "yellow_cards": int(np.random.poisson(3.5)),
        "red_cards": int(np.random.poisson(0.15)),
        "winning_team": result,
    }


def simulate_group_matches(group_teams, fixtures_df, poisson_model, xgb_model, features_fn):
    group_g = fixtures_df[fixtures_df["group"] == group_teams["group_name"]].copy()
    standings = {}
    for t in group_teams["teams"]:
        standings[t] = {"points": 0, "gd": 0, "gf": 0, "ga": 0}
    results = []

    for _, match in group_g.iterrows():
        home, away = match["home_team"], match["away_team"]
        pred = predict_group_match(home, away, poisson_model, xgb_model, features_fn)
        hg, ag = pred["home_goals"], pred["away_goals"]
        results.append({**match.to_dict(), **pred})

        standings[home]["gf"] += hg
        standings[home]["ga"] += ag
        standings[home]["gd"] += hg - ag
        standings[away]["gf"] += ag
        standings[away]["ga"] += hg
        standings[away]["gd"] += ag - hg

        if hg > ag:
            standings[home]["points"] += 3
        elif hg == ag:
            standings[home]["points"] += 1
            standings[away]["points"] += 1
        else:
            standings[away]["points"] += 3

    sorted_teams = sorted(
        standings.items(),
        key=lambda x: (x[1]["points"], x[1]["gd"], x[1]["gf"]),
        reverse=True,
    )
    return sorted_teams, results


THIRD_PLACE_SLOTS = {
    "Match 75": ["A", "B", "C", "D", "F"],
    "Match 78": ["C", "D", "F", "G", "H"],
    "Match 79": ["C", "E", "F", "H", "I"],
    "Match 80": ["E", "H", "I", "J", "K"],
    "Match 81": ["A", "E", "H", "I", "J"],
    "Match 82": ["B", "E", "F", "I", "J"],
    "Match 85": ["E", "F", "G", "I", "J"],
    "Match 88": ["D", "E", "I", "J", "L"],
}


def run_group_stage(fixtures_df, poisson_model, xgb_model, features_fn):
    groups = build_group_teams(fixtures_df)
    all_standings = {}
    all_results = {}

    for g in GROUP_NAMES:
        if g not in groups:
            continue
        group_data = {"group_name": g, "teams": groups[g]}
        standings, results = simulate_group_matches(group_data, fixtures_df, poisson_model, xgb_model, features_fn)
        all_standings[g] = standings
        all_results[g] = results

    return all_results, all_standings


def get_group_qualifiers(all_standings):
    winners = {}
    runners_up = {}
    thirds = {}

    for g, standings in all_standings.items():
        teams = [t for t, s in standings]
        winners[g] = teams[0]
        runners_up[g] = teams[1]
        thirds[g] = (teams[2], g, standings[2][1]["points"], standings[2][1]["gd"], standings[2][1]["gf"])

    ranked_thirds = sorted(
        list(thirds.values()),
        key=lambda x: (x[2], x[3], x[4]),
        reverse=True,
    )[:8]

    third_qualifiers = {}
    used_groups = set()
    # Assign best 3rds to specific bracket slots
    for slot_name, eligible_groups in THIRD_PLACE_SLOTS.items():
        for rt in ranked_thirds:
            team, group, pts, gd, gf = rt
            if group in eligible_groups and group not in used_groups and group not in third_qualifiers:
                third_qualifiers[group] = team
                used_groups.add(group)
                break

    # Fill any remaining with unassigned best thirds
    for rt in ranked_thirds:
        team, group, pts, gd, gf = rt
        if group not in third_qualifiers:
            third_qualifiers[group] = team

    return winners, runners_up, third_qualifiers
