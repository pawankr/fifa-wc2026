import pandas as pd
import numpy as np

class EloSystem:
    def __init__(self, initial_rating=1500, k_factor_base=40):
        self.initial_rating = initial_rating
        self.k_factor_base = k_factor_base
        self.ratings = {}

    def _k_factor(self, tournament, goal_diff, elo_diff):
        if "World Cup" in tournament and "qualification" not in tournament.lower():
            base = 60
        elif "World Cup" in tournament:
            base = 45
        elif any(name in tournament for name in ["Euro", "Copa América", "Africa Cup", "Asian Cup", "Gold Cup"]):
            base = 50 if "qualification" not in tournament.lower() else 40
        else:
            base = 30
        gd_weight = np.log(max(abs(goal_diff), 1) + 1)
        elo_factor = (2.2 / (2.2 + abs(elo_diff) / 1000))
        return base * gd_weight * elo_factor

    def _expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def get_rating(self, team, date=None):
        if team not in self.ratings:
            return self.initial_rating
        return self.ratings[team]

    def update_match(self, home, away, home_goals, away_goals, tournament, date):
        for team in [home, away]:
            if team not in self.ratings:
                self.ratings[team] = self.initial_rating

        elo_home = self.ratings[home]
        elo_away = self.ratings[away]

        exp_home = self._expected_score(elo_home, elo_away)
        exp_away = 1 - exp_home

        if home_goals > away_goals:
            actual_home, actual_away = 1, 0
            goal_diff = home_goals - away_goals
            elo_diff = elo_home - elo_away
        elif home_goals < away_goals:
            actual_home, actual_away = 0, 1
            goal_diff = away_goals - home_goals
            elo_diff = elo_away - elo_home
        else:
            actual_home, actual_away = 0.5, 0.5
            goal_diff = 0
            elo_diff = 0

        k = self._k_factor(tournament, goal_diff, elo_diff)

        self.ratings[home] = elo_home + k * (actual_home - exp_home)
        self.ratings[away] = elo_away + k * (actual_away - exp_away)

    def compute_historical_ratings(self, matches_df):
        all_ratings = []
        for _, row in matches_df.iterrows():
            home, away = row["home_team"], row["away_team"]
            home_g, away_g = row["home_score"], row["away_score"]
            tour = row["tournament"]
            date = row["date"]

            home_elo_before = self.get_rating(home, date)
            away_elo_before = self.get_rating(away, date)

            all_ratings.append({
                "date": date,
                "home_team": home,
                "away_team": away,
                "home_elo_before": home_elo_before,
                "away_elo_before": away_elo_before,
            })

            self.update_match(home, away, home_g, away_g, tour, date)

        ratings_df = pd.DataFrame(all_ratings)
        return ratings_df

    def get_current_ratings(self, teams=None):
        if teams:
            return {t: self.ratings.get(t, self.initial_rating) for t in teams}
        return dict(self.ratings)

    def get_rating_for_team(self, team, date, matches_df):
        sub = matches_df[matches_df["date"] < date]
        elo = EloSystem(initial_rating=self.initial_rating, k_factor_base=self.k_factor_base)
        elo.compute_historical_ratings(sub)
        return elo.get_rating(team)
