import numpy as np
import pandas as pd
from scipy.stats import poisson


class FastPoissonModel:
    def __init__(self):
        self.attack = {}
        self.defense = {}
        self.home_adv = 1.0
        self.avg_home_goals = 0
        self.avg_away_goals = 0
        self.rho = -0.1

    def fit(self, matches_df):
        all_teams = list(set(matches_df["home_team"]).union(set(matches_df["away_team"])))
        team_to_idx = {t: i for i, t in enumerate(all_teams)}
        n_teams = len(all_teams)

        home_goals = matches_df["home_score"].values
        away_goals = matches_df["away_score"].values
        home_idx = matches_df["home_team"].map(team_to_idx).values
        away_idx = matches_df["away_team"].map(team_to_idx).values

        self.avg_home_goals = np.mean(home_goals)
        self.avg_away_goals = np.mean(away_goals)

        goals_home = np.zeros(n_teams)
        conceded_home = np.zeros(n_teams)
        goals_away = np.zeros(n_teams)
        conceded_away = np.zeros(n_teams)
        matches_home = np.zeros(n_teams)
        matches_away = np.zeros(n_teams)

        for i in range(len(matches_df)):
            hi, ai = home_idx[i], away_idx[i]
            hg, ag = home_goals[i], away_goals[i]
            goals_home[hi] += hg
            conceded_home[hi] += ag
            goals_away[ai] += ag
            conceded_away[ai] += hg
            matches_home[hi] += 1
            matches_away[ai] += 1

        matches_home = np.maximum(matches_home, 1)
        matches_away = np.maximum(matches_away, 1)

        raw_attack_home = (goals_home / matches_home) / max(self.avg_home_goals, 0.01)
        raw_attack_away = (goals_away / matches_away) / max(self.avg_away_goals, 0.01)
        raw_def_home = (conceded_home / matches_home) / max(self.avg_away_goals, 0.01)
        raw_def_away = (conceded_away / matches_away) / max(self.avg_home_goals, 0.01)

        n_iter = 5
        alpha, beta = 0.3, 0.3

        attacks = (raw_attack_home + raw_attack_away) / 2
        defenses = (raw_def_home + raw_def_away) / 2

        for iteration in range(n_iter):
            for i in range(len(matches_df)):
                hi, ai = home_idx[i], away_idx[i]
                hg, ag = home_goals[i], away_goals[i]
                exp_h = self.avg_home_goals * attacks[hi] * defenses[ai]
                exp_a = self.avg_away_goals * attacks[ai] * defenses[hi]

            attacks = attacks / np.mean(attacks)
            defenses = defenses / np.mean(defenses)

        self.teams = all_teams
        self.attack = {t: attacks[i] for t, i in team_to_idx.items()}
        self.defense = {t: defenses[i] for t, i in team_to_idx.items()}
        self.home_adv = self.avg_home_goals / max(self.avg_away_goals, 0.01)

        return self

    def predict_expectations(self, home_team, away_team):
        att_h = self.attack.get(home_team, 1.0)
        def_h = self.defense.get(home_team, 1.0)
        att_a = self.attack.get(away_team, 1.0)
        def_a = self.defense.get(away_team, 1.0)

        exp_h = self.avg_home_goals * att_h * def_a
        exp_a = self.avg_away_goals * att_a * def_h
        return exp_h, exp_a

    def _dc_tau(self, x, y, lam, mu):
        rho = self.rho
        if x == 0 and y == 0:
            return 1 - lam * mu * rho
        elif x == 0 and y == 1:
            return 1 + lam * rho
        elif x == 1 and y == 0:
            return 1 + mu * rho
        elif x == 1 and y == 1:
            return 1 - rho
        else:
            return 1.0

    def match_probabilities(self, home_team, away_team, max_goals=8):
        exp_h, exp_a = self.predict_expectations(home_team, away_team)

        probs = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                tau = self._dc_tau(i, j, exp_h, exp_a)
                probs[i, j] = tau * poisson.pmf(i, exp_h) * poisson.pmf(j, exp_a)

        probs = probs / probs.sum()

        home_win = 0
        draw = 0
        away_win = 0
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                if i > j:
                    home_win += probs[i, j]
                elif i == j:
                    draw += probs[i, j]
                else:
                    away_win += probs[i, j]

        return {
            "home_exp": exp_h,
            "away_exp": exp_a,
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
            "prob_matrix": probs,
        }


def train_poisson_model(matches_df):
    model = FastPoissonModel()
    model.fit(matches_df)
    return model
