import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"

ELO_URL = "https://www.international-football.net/elo-ratings-table"
ELO_CSV = DATA_DIR / "elo_ratings_2026.csv"

FIXTURE_TEAMS = [
    "Mexico", "South Korea", "South Africa", "Canada", "Qatar", "Switzerland",
    "Brazil", "Morocco", "Haiti", "Scotland", "Germany", "Curacao",
    "Netherlands", "Japan", "Ivory Coast", "Ecuador", "Tunisia",
    "Spain", "Cape Verde", "Belgium", "Egypt", "Saudi Arabia", "Uruguay",
    "Iran", "New Zealand", "Austria", "Jordan", "France", "Senegal",
    "Norway", "Argentina", "Algeria", "Portugal", "England", "Croatia",
    "Ghana", "Panama", "Uzbekistan", "Colombia", "Paraguay", "Australia",
    "USA",
]

PLAYOFF_TEAMS = {
    "UEFA Playoff A": ["Turkey", "Greece", "Ukraine", "Poland"],
    "UEFA Playoff B": ["Norway", "Serbia", "Slovakia", "Hungary"],
    "UEFA Playoff C": ["Sweden", "Wales", "Finland", "Bosnia"],
    "UEFA Playoff D": ["Russia", "Czech Republic", "Slovenia", "Romania"],
    "FIFA Playoff 1": ["Panama", "Costa Rica", "Honduras", "Jamaica"],
    "FIFA Playoff 2": ["Iraq", "Vietnam", "Thailand", "Tajikistan"],
}

NAME_MAP = {
    "United States": "USA",
    "Korea Republic": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Curaçao": "Curacao",
    "Cabo Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde",
}

def scrape_elo_ratings():
    print("Scraping current Elo ratings...")
    resp = requests.get(ELO_URL, params={"year": "2026", "month": "05", "day": "01"}, timeout=30)
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if not table:
        print("Could not find Elo ratings table, trying alternative method...")
        return _parse_elo_from_text(resp.text)

    ratings = {}
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 3:
            rank_cell = cells[0].get_text(strip=True)
            name_cell = cells[1].get_text(strip=True) if len(cells) > 1 else ""
            elo_cell = cells[2].get_text(strip=True) if len(cells) > 2 else ""
            name = re.sub(r'^\d+\.\s*', '', name_cell).strip()
            if name and elo_cell.replace('.', '').replace(',', '').isdigit():
                ratings[name] = int(float(elo_cell))
    return ratings

def _parse_elo_from_text(text):
    ratings = {}
    lines = text.split('\n')
    for line in lines:
        match = re.search(r'^\s*(\d+)\.\s*\|\s*([A-Za-z\s\-]+)\s*\|\s*(\d+)', line)
        if match:
            name = match.group(2).strip()
            elo = int(match.group(3))
            ratings[name] = elo
    return ratings

def save_elo_ratings(ratings):
    df = pd.DataFrame(list(ratings.items()), columns=["team", "elo"])
    df.to_csv(ELO_CSV, index=False)
    print(f"Saved {len(df)} Elo ratings to {ELO_CSV}")

def load_elo_ratings():
    if ELO_CSV.exists():
        df = pd.read_csv(ELO_CSV)
        return dict(zip(df["team"], df["elo"]))
    ratings = scrape_elo_ratings()
    if ratings:
        save_elo_ratings(ratings)
    return ratings

def get_team_elo(team, ratings):
    mapped = NAME_MAP.get(team, team)
    return ratings.get(mapped)

def get_playoff_elo(playoff_name, ratings):
    candidates = PLAYOFF_TEAMS.get(playoff_name, [])
    elos = [ratings.get(c) for c in candidates if ratings.get(c)]
    return int(np.mean(elos)) if elos else 1500

import numpy as np

def resolve_playoff_teams(fixtures_df, knockout_slots_df, ratings):
    fixtures = fixtures_df.copy()
    knockouts = knockout_slots_df.copy()

    for col in ["home_team", "away_team"]:
        mask = fixtures[col].str.contains("Playoff|Play-off", na=False)
        fixtures.loc[mask, col + "_elo"] = fixtures.loc[mask, col].apply(
            lambda x: get_playoff_elo(x, ratings)
        )
        fixtures.loc[~mask, col + "_elo"] = fixtures.loc[~mask, col].apply(
            lambda x: get_team_elo(x, ratings) or 1500
        )

    return fixtures, knockouts
