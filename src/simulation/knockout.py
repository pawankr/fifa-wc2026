import pandas as pd
import numpy as np
import re


def resolve_slot(slot_str, winners, runners_up, third_qualifiers):
    groups = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

    for g in groups:
        m = re.search(rf"Winner\s+Group\s+{g}", slot_str, re.IGNORECASE)
        if m:
            return winners.get(g, "TBD")
        m = re.search(rf"Runner-up\s+Group\s+{g}", slot_str, re.IGNORECASE)
        if m:
            return runners_up.get(g, "TBD")

    m = re.search(r"Best 3rd\s*\(Groups\s+([A-Za-z/]+)\)", slot_str)
    if m:
        group_set = m.group(1).split("/")
        for g in group_set:
            if g in third_qualifiers:
                return third_qualifiers[g]
        return "TBD"

    return slot_str


def resolve_first_round(knockout_slots_df, winners, runners_up, third_qualifiers):
    resolved = knockout_slots_df[knockout_slots_df["round"] == "Round of 32"].copy()
    resolved["home_team"] = resolved["slot_home"].apply(
        lambda s: resolve_slot(str(s), winners, runners_up, third_qualifiers)
    )
    resolved["away_team"] = resolved["slot_away"].apply(
        lambda s: resolve_slot(str(s), winners, runners_up, third_qualifiers)
    )
    return resolved


def predict_knockout_match(home, away, poisson_model, xgb_model, features_fn):
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
    penalties = False

    if r < combined_hw:
        result = "home"
        if avg_h <= avg_a:
            avg_h = avg_a + 1
    elif r < combined_hw + combined_d:
        penalties = True
        avg_h = max(avg_h, 1)
        avg_a = avg_h
        result = "home" if np.random.random() < 0.5 else "away"
    else:
        result = "away"
        if avg_a <= avg_h:
            avg_a = avg_h + 1

    att_h = poisson_model.attack.get(home, 1.0)
    att_a = poisson_model.attack.get(away, 1.0)
    avg_att = sum(poisson_model.attack.values()) / max(len(poisson_model.attack), 1)

    corners = int(round(5 * att_h / avg_att + 4 * att_a / avg_att))
    yellows = int(round(2.5 * att_h / avg_att + 2.0 * att_a / avg_att))
    reds = int(np.random.poisson(0.12))

    return {
        "home_team": home,
        "away_team": away,
        "home_goals": max(0, avg_h),
        "away_goals": max(0, avg_a),
        "winner": home if result == "home" else away,
        "loser": away if result == "home" else home,
        "corners": corners,
        "yellow_cards": yellows,
        "red_cards": reds,
        "penalties": penalties,
    }


def run_full_knockout(knockout_slots_df, poisson_model, xgb_model, features_fn, resolved_first_round):
    round_order = ["Round of 32", "Round of 16", "Quarter-final", "Semi-final"]
    match_winners = {}
    all_predictions = []

    def resolve_round_matches(round_name, bracket_df, match_winners):
        round_df = bracket_df[bracket_df["round"] == round_name].copy()
        if round_name != "Round of 32":
            for idx, match in round_df.iterrows():
                for slot_col, team_col in [("slot_home", "home_team"), ("slot_away", "away_team")]:
                    slot = str(match[slot_col])
                    m = re.search(r"Winner Match (\d+)", slot)
                    if m:
                        mid = int(m.group(1))
                        round_df.at[idx, team_col] = match_winners.get(mid, "TBD")
        return round_df

    for round_name in round_order:
        if round_name == "Round of 32":
            round_matches = resolved_first_round
        else:
            round_matches = resolve_round_matches(round_name, knockout_slots_df, match_winners)

        for _, match in round_matches.iterrows():
            home, away = match["home_team"], match["away_team"]
            mid = match["match_id"]
            if home == "TBD" or away == "TBD":
                continue
            pred = predict_knockout_match(home, away, poisson_model, xgb_model, features_fn)
            pred["match_id"] = mid
            pred["round"] = round_name
            all_predictions.append(pred)
            match_winners[mid] = pred["winner"]

    third_place_teams = {}
    for p in all_predictions:
        if p.get("round") == "Semi-final":
            mid = p["match_id"]
            third_place_teams[mid] = p["loser"]

    tp_df = knockout_slots_df[knockout_slots_df["round"] == "Third-place playoff"].copy()
    for idx, match in tp_df.iterrows():
        for slot_col, team_col in [("slot_home", "home_team"), ("slot_away", "away_team")]:
            slot = str(match[slot_col])
            m = re.search(r"Loser Match (\d+)", slot)
            if m:
                mid = int(m.group(1))
                tp_df.at[idx, team_col] = third_place_teams.get(mid, "TBD")

    for _, match in tp_df.iterrows():
        home, away = match["home_team"], match["away_team"]
        if home != "TBD" and away != "TBD":
            pred = predict_knockout_match(home, away, poisson_model, xgb_model, features_fn)
            pred["match_id"] = match["match_id"]
            pred["round"] = "Third-place playoff"
            all_predictions.append(pred)

    final_df = knockout_slots_df[knockout_slots_df["round"] == "Final"].copy()
    for idx, match in final_df.iterrows():
        for slot_col, team_col in [("slot_home", "home_team"), ("slot_away", "away_team")]:
            slot = str(match[slot_col])
            m = re.search(r"Winner Match (\d+)", slot)
            if m:
                mid = int(m.group(1))
                final_df.at[idx, team_col] = match_winners.get(mid, "TBD")

    for _, match in final_df.iterrows():
        home, away = match["home_team"], match["away_team"]
        if home != "TBD" and away != "TBD":
            pred = predict_knockout_match(home, away, poisson_model, xgb_model, features_fn)
            pred["match_id"] = match["match_id"]
            pred["round"] = "Final"
            all_predictions.append(pred)

    result_df = pd.DataFrame(all_predictions)
    if len(result_df) == 0:
        return result_df

    result_df["match_winner"] = result_df.apply(
        lambda r: "home" if r["winner"] == r["home_team"] else "away", axis=1
    )
    intensity_bonus = {"Round of 32": 1, "Round of 16": 1.2, "Quarter-final": 1.3, "Semi-final": 1.4, "Third-place playoff": 1.2, "Final": 1.5}
    result_df["corners"] = result_df.apply(
        lambda r: int(round(r["corners"] * intensity_bonus.get(r.get("round", ""), 1))), axis=1
    )
    result_df["yellow_cards"] = result_df.apply(
        lambda r: int(round(r["yellow_cards"] * intensity_bonus.get(r.get("round", ""), 1))), axis=1
    )

    return result_df
