import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_HISTORICAL = DATA_DIR / "results.csv"
PROCESSED_DIR = Path(__file__).parent.parent.parent / "outputs"

HISTORICAL_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 1.0,
    "UEFA Euro": 0.95,
    "Copa América": 0.90,
    "Africa Cup of Nations": 0.85,
    "Asian Cup": 0.80,
    "CONCACAF Gold Cup": 0.75,
    "OFC Nations Cup": 0.60,
    "FIFA World Cup qualification": 0.70,
    "UEFA Euro qualification": 0.65,
    "Africa Cup of Nations qualification": 0.60,
    "Asian Cup qualification": 0.55,
    "CONCACAF Gold Cup qualification": 0.50,
    "Copa América qualification": 0.50,
    "Confederations Cup": 0.80,
    "FIFA World Cup play-off": 0.70,
}

def download_historical_data():
    if RAW_HISTORICAL.exists():
        print(f"Historical data already exists at {RAW_HISTORICAL}")
        return
    print("Downloading historical match data...")
    df = pd.read_csv(HISTORICAL_URL)
    df.to_csv(RAW_HISTORICAL, index=False)
    print(f"Downloaded {len(df)} matches to {RAW_HISTORICAL}")

def load_historical_data(min_year=2000):
    if not RAW_HISTORICAL.exists():
        download_historical_data()
    df = pd.read_csv(RAW_HISTORICAL)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year >= min_year].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df

def load_fixtures():
    group_fixtures = pd.read_csv(DATA_DIR / "group_fixtures.csv")
    knockout_slots = pd.read_csv(DATA_DIR / "knockout_slots.csv")
    return group_fixtures, knockout_slots

def get_team_name_mapping():
    return {
        "Côte d'Ivoire": "Ivory Coast",
        "Curaçao": "Curacao",
        "Cabo Verde": "Cape Verde",
        "United States": "USA",
        "Korea Republic": "South Korea",
        "Iran": "IR Iran",
    }

TOURNAMENT_CATEGORIES = {
    "competitive": [
        "FIFA World Cup",
        "UEFA Euro",
        "Copa América",
        "Africa Cup of Nations",
        "Asian Cup",
        "CONCACAF Gold Cup",
        "OFC Nations Cup",
        "Confederations Cup",
    ],
}
