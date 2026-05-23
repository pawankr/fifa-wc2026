import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"

ELO_CSV = DATA_DIR / "elo_ratings_2026.csv"

# Actual playoff qualifiers confirmed March 31, 2026
PLAYOFF_TEAMS_ACTUAL = {
    "UEFA Playoff A": ["Sweden"],
    "UEFA Playoff B": ["Turkey"],
    "UEFA Playoff C": ["Bosnia and Herzegovina"],
    "UEFA Playoff D": ["Czech Republic"],
    "FIFA Playoff 1": ["Iraq"],
    "FIFA Playoff 2": ["DR Congo"],
}

NAME_MAP = {
    "Côte d'Ivoire": "Ivory Coast",
    "Curaçao": "Curacao",
    "Cabo Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde",
    "United States": "USA",
    "Korea Republic": "South Korea",
    "Czechia": "Czech Republic",
}


def get_playoff_elo(playoff_name, ratings):
    candidates = PLAYOFF_TEAMS_ACTUAL.get(playoff_name, [])
    elos = [ratings.get(c) for c in candidates if ratings.get(c)]
    return int(np.mean(elos)) if elos else 1500


def get_team_elo(team, ratings):
    mapped = NAME_MAP.get(team, team)
    return ratings.get(mapped)
