import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent.parent / "outputs" / "models"

FEATURE_COLS = [
    "elo_difference",
    "home_goals_scored_avg",
    "home_goals_conceded_avg",
    "away_goals_scored_avg",
    "away_goals_conceded_avg",
    "home_form_points",
    "away_form_points",
    "home_matches_played",
    "away_matches_played",
    "is_world_cup",
    "neutral",
]


class XGBoostPredictor:
    def __init__(self):
        self.outcome_model = None
        self.goals_home_model = None
        self.goals_away_model = None

    def _encode_outcome(self, row):
        if row["home_score"] > row["away_score"]:
            return 0
        elif row["home_score"] == row["away_score"]:
            return 1
        else:
            return 2

    def train_outcome(self, features_df, test_size=0.15):
        df = features_df.dropna(subset=FEATURE_COLS).copy()
        df["outcome"] = df.apply(self._encode_outcome, axis=1)

        X = df[FEATURE_COLS].values
        y = df["outcome"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )

        self.outcome_model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.1,
            eval_metric="mlogloss",
            random_state=42,
        )
        self.outcome_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = self.outcome_model.predict(X_test)
        y_prob = self.outcome_model.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, y_prob)
        print(f"  Outcome model — Accuracy: {acc:.3f}, Log Loss: {ll:.3f}")
        return acc, ll

    def train_goals_regressors(self, features_df, test_size=0.15):
        df = features_df.dropna(subset=FEATURE_COLS).copy()

        X = df[FEATURE_COLS].values

        for target, name in [(df["home_score"].values, "home"), (df["away_score"].values, "away")]:
            X_train, X_test, y_train, y_test = train_test_split(
                X, target, test_size=test_size, random_state=42, shuffle=False
            )

            model = XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=42,
            )
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            if name == "home":
                self.goals_home_model = model
            else:
                self.goals_away_model = model

            y_pred = model.predict(X_test)
            mae = np.mean(np.abs(y_pred - y_test))
            print(f"  Goals ({name}) model — MAE: {mae:.3f}")

    def predict_outcome(self, features_dict):
        x = np.array([[features_dict.get(c, 0) for c in FEATURE_COLS]])
        if self.outcome_model is None:
            return {"home_win": 0.45, "draw": 0.25, "away_win": 0.30}
        probs = self.outcome_model.predict_proba(x)[0]
        return {"home_win": probs[0], "draw": probs[1], "away_win": probs[2]}

    def predict_goals(self, features_dict):
        x = np.array([[features_dict.get(c, 0) for c in FEATURE_COLS]])
        home_g = self.goals_home_model.predict(x)[0] if self.goals_home_model else 1.5
        away_g = self.goals_away_model.predict(x)[0] if self.goals_away_model else 1.0
        return max(0, round(home_g)), max(0, round(away_g))

    def save(self, path=None):
        path = path or MODEL_DIR
        path.mkdir(parents=True, exist_ok=True)
        if self.outcome_model:
            joblib.dump(self.outcome_model, path / "outcome_model.joblib")
        if self.goals_home_model:
            joblib.dump(self.goals_home_model, path / "goals_home_model.joblib")
        if self.goals_away_model:
            joblib.dump(self.goals_away_model, path / "goals_away_model.joblib")

    def load(self, path=None):
        path = path or MODEL_DIR
        outcome_path = path / "outcome_model.joblib"
        if outcome_path.exists():
            self.outcome_model = joblib.load(outcome_path)
        gh_path = path / "goals_home_model.joblib"
        if gh_path.exists():
            self.goals_home_model = joblib.load(gh_path)
        ga_path = path / "goals_away_model.joblib"
        if ga_path.exists():
            self.goals_away_model = joblib.load(ga_path)
