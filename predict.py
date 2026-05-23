import pandas as pd
import numpy as np
import sys
import re
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

from src.data.load_data import download_historical_data, load_historical_data, load_fixtures
from src.features.elo import EloSystem
from src.features.build_features import compute_rolling_features
from src.models.poisson import train_poisson_model
from src.models.xgboost_model import XGBoostPredictor
from src.simulation.group_stage import run_group_stage, get_group_qualifiers, predict_group_match
from src.simulation.knockout import run_full_knockout

MC_ITERATIONS = 1_000

PLAYOFF_MAP = {
    "UEFA Playoff A": "Sweden", "UEFA Playoff B": "Turkey",
    "UEFA Playoff C": "Bosnia and Herzegovina", "UEFA Playoff D": "Czech Republic",
    "FIFA Playoff 1": "Iraq", "FIFA Playoff 2": "DR Congo",
}
NAME_MAP = {"Czechia": "Czech Republic", "USA": "United States"}

FIFA_RANKINGS_URL = "https://raw.githubusercontent.com/irieti/fifa/main/fifa_rankings_2026.csv"
FIFA_RANKINGS_CSV = Path("data") / "fifa_rankings_2026.csv"


def resolve_playoff_name(name):
    if name in PLAYOFF_MAP:
        return PLAYOFF_MAP[name]
    return NAME_MAP.get(name, name)


def load_fifa_rankings():
    if not FIFA_RANKINGS_CSV.exists():
        df = pd.read_csv(FIFA_RANKINGS_URL)
        df.to_csv(FIFA_RANKINGS_CSV, index=False)
    df = pd.read_csv(FIFA_RANKINGS_CSV)
    fix = {"United States": "USA", "Czechia": "Czech Republic", "Congo DR": "DR Congo", "Ivory Coast": "Côte d'Ivoire"}
    return {fix.get(r["team"], r["team"]): r["fifa_points"] for _, r in df.iterrows()}


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
            "goals_scored_avg": float(np.mean(d["goals_for"][-10:])) or 1.5,
            "goals_conceded_avg": float(np.mean(d["goals_against"][-10:])) or 1.5,
            "form_points": float(np.mean(d["points"][-5:])) or 1.5,
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


def build_r32(knockout_slots, winners, runners_up, thirds):
    r32 = knockout_slots[knockout_slots["round"] == "Round of 32"].copy()
    t3 = list(thirds.values())
    si = 0
    for i in r32.index:
        r32.at[i, "home_team"] = resolve_slot(str(r32.at[i, "slot_home"]), winners, runners_up)
        r32.at[i, "away_team"] = resolve_slot(str(r32.at[i, "slot_away"]), winners, runners_up)
        for col in ["slot_home", "slot_away"]:
            if "Best 3rd" in str(r32.at[i, col]) and si < len(t3):
                r32.at[i, "home_team" if col == "slot_home" else "away_team"] = t3[si]
                si += 1
    return r32


def resolve_knockout_winner_goals(home, away, poisson, xgb, fn):
    p = poisson.match_probabilities(home, away)
    xp = xgb.predict_outcome(fn(home, away))
    xhg, xag = xgb.predict_goals(fn(home, away))
    w = 0.3
    hw = w * p["home_win"] + (1 - w) * xp["home_win"]
    aw = w * p["away_win"] + (1 - w) * xp["away_win"]
    dr = w * p["draw"] + (1 - w) * xp["draw"]
    t = hw + dr + aw
    hw, dr, aw = hw / t, dr / t, aw / t
    eh, ea = p["home_exp"], p["away_exp"]
    hg = int(round((eh + xhg) / 2))
    ag = int(round((ea + xag) / 2))
    mw = "home" if hw >= aw else "away"
    pen = dr > 0.25
    if mw == "home":
        hg, ag = max(hg, ag + 1), min(ag, hg - 1)
    else:
        ag, hg = max(ag, hg + 1), min(hg, ag - 1)
    c = int(round(9 + 0.3 * (eh + ea - 2)))
    yc = int(round(3.5 + 0.3 * (hw + aw)))
    return hg, ag, c, yc, int(np.random.poisson(0.1)), mw, pen


def main():
    print("=" * 60)
    print("FIFA World Cup 2026 — Monte Carlo Predictor")
    print("=" * 60)

    print("\n[1/6] Loading data...")
    download_historical_data()
    matches_df = load_historical_data(min_year=2000)
    group_fixtures, knockout_slots = load_fixtures()
    print(f"  Historical: {len(matches_df)}  Group: {len(group_fixtures)}  KO: {len(knockout_slots)}")

    print("\n[2/6] Elo + FIFA rankings...")
    elo = EloSystem()
    training = matches_df[matches_df["date"] < "2026-01-01"].copy()
    ratings_df = elo.compute_historical_ratings(training)
    fifa_r = load_fifa_rankings()
    top = sorted(elo.ratings.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"  Elo: {len(elo.ratings)} teams — Top: {[(t, int(r)) for t, r in top]}")
    print(f"  FIFA: {len(fifa_r)} teams")

    print("\n[3/6] Features...")
    features_df = compute_rolling_features(training, ratings_df)
    team_stats = build_team_recent_stats(training)

    def features_fn(h, a):
        hr, ar = resolve_playoff_name(h), resolve_playoff_name(a)
        hs = team_stats.get(hr, {}); as_ = team_stats.get(ar, {})
        return {
            "elo_difference": elo.get_rating(hr) - elo.get_rating(ar),
            "home_goals_scored_avg": hs.get("goals_scored_avg", 1.5),
            "home_goals_conceded_avg": hs.get("goals_conceded_avg", 1.3),
            "away_goals_scored_avg": as_.get("goals_scored_avg", 1.3),
            "away_goals_conceded_avg": as_.get("goals_conceded_avg", 1.5),
            "home_form_points": hs.get("form_points", 1.5),
            "away_form_points": as_.get("form_points", 1.4),
            "home_matches_played": hs.get("matches_played", 20),
            "away_matches_played": as_.get("matches_played", 20),
            "is_world_cup": 1, "neutral": 1,
        }
    print(f"  {len(features_df)} feature rows, {len(team_stats)} teams")

    print("\n[4/6] Poisson...")
    poisson = train_poisson_model(training)
    print(f"  {len(poisson.teams)} teams, avg home {poisson.avg_home_goals:.3f}")

    print("\n[5/6] XGBoost...")
    xgb = XGBoostPredictor()
    xgb.train_outcome(features_df)
    xgb.train_goals_regressors(features_df)

    # ===== MONTE CARLO GROUP STAGE =====
    print(f"\n[6/6] Monte Carlo ({MC_ITERATIONS:,} iterations)...")
    mc_group = {i: {"hg": [], "ag": [], "wt": []}
                for i in range(len(group_fixtures))}

    for it in range(MC_ITERATIONS):
        if it % 200 == 0:
            print(f"  {it}/{MC_ITERATIONS}")
        np.random.seed(it * 7 + 13)
        _, standings = run_group_stage(group_fixtures, poisson, xgb, features_fn)
        for idx in range(len(group_fixtures)):
            h, a = group_fixtures.at[idx, "home_team"], group_fixtures.at[idx, "away_team"]
            np.random.seed(it * 1000 + idx)
            p = predict_group_match(h, a, poisson, xgb, features_fn)
            mc_group[idx]["hg"].append(p["home_goals"])
            mc_group[idx]["ag"].append(p["away_goals"])
            mc_group[idx]["wt"].append(p["winning_team"])

    # ===== AGGREGATE GROUP (goals/results from MC, cards/corners from team stats) =====
    print("\n  Aggregating group results (mode)...")
    group_preds = group_fixtures.copy()
    for col in ["predicted_home_goals", "predicted_away_goals", "corners", "yellow_cards", "red_cards"]:
        group_preds[col] = 0
    group_preds["winning_team"] = ""

    np.random.seed(42)
    for idx in range(len(group_fixtures)):
        r = mc_group[idx]
        hg_mode = Counter(r["hg"]).most_common(1)[0][0]
        ag_mode = Counter(r["ag"]).most_common(1)[0][0]
        group_preds.at[idx, "predicted_home_goals"] = hg_mode
        group_preds.at[idx, "predicted_away_goals"] = ag_mode
        group_preds.at[idx, "winning_team"] = "draw" if hg_mode == ag_mode else Counter(r["wt"]).most_common(1)[0][0]

        home = group_fixtures.at[idx, "home_team"]
        away = group_fixtures.at[idx, "away_team"]
        att_h = poisson.attack.get(resolve_playoff_name(home), 1.0)
        att_a = poisson.attack.get(resolve_playoff_name(away), 1.0)
        avg_att = sum(poisson.attack.values()) / max(len(poisson.attack), 1)
        group_preds.at[idx, "corners"] = int(round(5 * att_h / avg_att + 4 * att_a / avg_att))
        group_preds.at[idx, "yellow_cards"] = int(round(2.5 * att_h / avg_att + 2.0 * att_a / avg_att))
        group_preds.at[idx, "red_cards"] = 1 if np.random.random() < 0.10 * (att_h / avg_att + att_a / avg_att) / 2 else 0

    # Build consensus group standings
    print("\n  --- Consensus Group Standings ---")
    standings_data = {}
    for _, m in group_fixtures.iterrows():
        g = m["group"]
        standings_data.setdefault(g, {})
        for t in [m["home_team"], m["away_team"]]:
            standings_data[g].setdefault(t, {"points": 0, "gd": 0, "gf": 0, "ga": 0})

    for idx, m in group_fixtures.iterrows():
        g = m["group"]; h, a = m["home_team"], m["away_team"]
        hg = group_preds.at[idx, "predicted_home_goals"]
        ag = group_preds.at[idx, "predicted_away_goals"]
        for team, gf, ga in [(h, hg, ag), (a, ag, hg)]:
            standings_data[g][team]["gf"] += gf
            standings_data[g][team]["ga"] += ga
            standings_data[g][team]["gd"] += gf - ga
            if gf > ga:
                standings_data[g][team]["points"] += 3
            elif gf == ga:
                standings_data[g][team]["points"] += 1

    final_standings = {}
    for g in sorted(standings_data.keys()):
        sd = sorted(standings_data[g].items(), key=lambda x: (x[1]["points"], x[1]["gd"], x[1]["gf"]), reverse=True)
        final_standings[g] = sd
        print(f"  Group {g}:")
        for i, (t, s) in enumerate(sd, 1):
            print(f"    {i}. {t:20s} {s['points']}pts ({s['gd']:+d} GD)")

    # Qualifiers
    winners = {g: resolve_playoff_name(s[0][0]) for g, s in final_standings.items()}
    runners_up = {g: resolve_playoff_name(s[1][0]) for g, s in final_standings.items()}
    all_3rds = [(resolve_playoff_name(s[2][0]), g, s[2][1]["points"], s[2][1]["gd"], s[2][1]["gf"]) for g, s in final_standings.items() if len(s) > 2]
    best_3rds = sorted(all_3rds, key=lambda x: (x[2], x[3], x[4]), reverse=True)[:8]
    thirds = {g: team for team, g, _, _, _ in best_3rds}

    print(f"\n  Winners: {[winners[g] for g in sorted(winners.keys())]}")
    print(f"  Runners: {[runners_up[g] for g in sorted(runners_up.keys())]}")
    print(f"  3rds:    {list(thirds.values())}")

    # ===== DETERMINISTIC KNOCKOUT =====
    print("\n  --- Knockout Bracket ---")
    r32 = build_r32(knockout_slots, winners, runners_up, thirds)

    np.random.seed(42)
    bracket = run_full_knockout(knockout_slots, poisson, xgb, features_fn, r32)

    ko_list = []
    if len(bracket) > 0:
        for _, m in bracket.iterrows():
            hg = int(round(m["home_goals"]))
            ag = int(round(m["away_goals"]))
            mw = m["match_winner"]
            pen = m.get("penalties", False)
            if not pen:
                if mw == "home" and hg <= ag: hg = ag + 1
                if mw == "away" and ag <= hg: ag = hg + 1
            print(f"  {m['match_id']:3d} {m['home_team']:18s} {hg}-{ag}  {m['away_team']:18s}{' (P)' if m.get('penalties') else ''}")
            ko_list.append({
                "match_id": m["match_id"],
                "predicted_home_team": m["home_team"],
                "predicted_away_team": m["away_team"],
                "predicted_home_goals": hg,
                "predicted_away_goals": ag,
                "corners": int(round(m["corners"])),
                "yellow_cards": int(round(m["yellow_cards"])),
                "red_cards": int(round(m["red_cards"])),
                "match_winner": mw,
                "penalties": m.get("penalties", False),
            })

        champ = ko_list[-1]["predicted_home_team"] if ko_list[-1]["match_winner"] == "home" else ko_list[-1]["predicted_away_team"]
        print(f"\n  Champion: {champ}")

    # ===== SAVE =====
    print("\n  --- Saving ---")
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    group_preds.to_csv(out / "group_predictions.csv", index=False)
    print(f"  {out / 'group_predictions.csv'} ({len(group_preds)} matches)")

    kp = pd.DataFrame(ko_list)
    if len(kp) > 0:
        kp.to_csv(out / "knockout_predictions.csv", index=False)
        print(f"  {out / 'knockout_predictions.csv'} ({len(kp)} matches)")
    else:
        print("  No knockout predictions")

    print("\nDone!")


if __name__ == "__main__":
    main()
